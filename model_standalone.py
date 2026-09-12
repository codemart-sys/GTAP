import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MASKS = tuple(itertools.product((0, 1), repeat=3))
MASK_ID = {mask: i for i, mask in enumerate(MASKS)}


def exact_shapley(values):
    if values.ndim != 2 or values.size(1) != 8:
        raise ValueError('expected [B,8] coalition values')
    values = values if values.is_floating_point() else values.float()
    phi = values.new_zeros(values.size(0), 3)
    for player in range(3):
        for coalition in MASKS:
            if coalition[player]:
                continue
            k = sum(coalition)
            added = list(coalition)
            added[player] = 1
            phi[:, player] += math.factorial(k) * math.factorial(2 - k) / math.factorial(3) * (
                values[:, MASK_ID[tuple(added)]] - values[:, MASK_ID[coalition]])
    return phi


def random_sample(points, npoint):
    b, n, _ = points.shape
    npoint = min(npoint, n)
    if npoint >= n:
        return torch.arange(n, device=points.device).unsqueeze(0).expand(b, -1)
    return torch.stack([torch.randperm(n, device=points.device)[:npoint] for _ in range(b)])


def furthest_point_sample(points, npoint):
    b, n, _ = points.shape
    npoint = min(npoint, n)
    device = points.device
    centroids = torch.zeros(b, npoint, dtype=torch.long, device=device)
    distance = torch.full((b, n), 1e10, device=device)
    farthest = torch.zeros(b, dtype=torch.long, device=device)
    batch = torch.arange(b, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = points[batch, farthest].view(b, 1, 3)
        distance = torch.minimum(distance, torch.sum((points - centroid) ** 2, -1))
        farthest = torch.max(distance, -1)[1]
    return centroids


def knn(query_xyz, support_xyz, k):
    dist = torch.cdist(query_xyz, support_xyz)
    k = min(k, support_xyz.shape[1])
    idx = dist.topk(k, dim=-1, largest=False)[1]
    return dist.gather(-1, idx), idx


def three_interpolation(p1, p2, f2):
    b, c, _ = f2.shape
    k = min(3, p2.shape[1])
    dist, idx = knn(p1, p2, k)
    if k < 3:
        idx = idx.expand(-1, -1, 3).contiguous()
        dist = dist.expand(-1, -1, 3).contiguous()
    weight = 1.0 / (dist + 1e-8)
    weight = weight / weight.sum(-1, keepdim=True)
    gathered = f2.transpose(1, 2).gather(1, idx.reshape(b, -1, 1).expand(-1, -1, c)).reshape(b, p1.shape[1], 3, c)
    return (gathered * weight.unsqueeze(-1)).sum(2).transpose(1, 2).contiguous()


def make_norm(channels, norm, dim=1):
    if norm == 'bn':
        return nn.BatchNorm1d(channels) if dim == 1 else nn.BatchNorm2d(channels)
    if norm == 'gn':
        groups = 8 if channels % 8 == 0 else 1
        return nn.GroupNorm(groups, channels)
    return nn.Identity()


def conv_block1d(in_c, out_c, norm=None, act=True):
    layers = [nn.Conv1d(in_c, out_c, 1)]
    if norm:
        layers.append(make_norm(out_c, norm, dim=1))
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def conv_block2d(in_c, out_c, norm=None, act=True):
    layers = [nn.Conv2d(in_c, out_c, 1)]
    if norm:
        layers.append(make_norm(out_c, norm, dim=2))
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def local_aggregate(new_p, p, f, k):
    b, c, _ = f.shape
    _, idx = knn(new_p, p, k)
    kk = idx.shape[-1]
    grouped = f.gather(2, idx.reshape(b, -1).unsqueeze(1).expand(-1, c, -1)).reshape(b, c, new_p.shape[1], kk)
    dp = p.gather(1, idx.reshape(b, -1).unsqueeze(-1).expand(-1, -1, 3)).reshape(b, new_p.shape[1], kk, 3) - new_p.unsqueeze(2)
    return torch.cat((dp.permute(0, 3, 1, 2).contiguous(), grouped), 1)


class LocalAggregation(nn.Module):
    def __init__(self, in_channels, out_channels, layers=1, k=32, norm='bn', act=True):
        super().__init__()
        channels = [in_channels + 3] + [out_channels] * (layers - 1) + [out_channels]
        convs = [conv_block2d(channels[i], channels[i + 1], norm=norm,
                              act=act if i < len(channels) - 2 else False)
                 for i in range(len(channels) - 1)]
        self.convs = nn.Sequential(*convs)
        self.k = k

    def forward(self, p, f):
        return self.convs(local_aggregate(p, p, f, self.k)).max(-1)[0]


class SetAbstraction(nn.Module):
    def __init__(self, in_channels, out_channels, layers=1, stride=1, k=32, norm='bn', act=True,
                 is_head=False, use_res=False, sampler='random'):
        super().__init__()
        self.stride = stride
        self.is_head = is_head
        self.all_aggr = not is_head and stride == 1
        self.use_res = use_res and not self.all_aggr and not is_head
        self.sampler = sampler
        self.k = k
        mid = out_channels // 2 if stride > 1 else out_channels
        channels = [in_channels] + [mid] * (layers - 1) + [out_channels]
        channels[0] = in_channels if is_head else in_channels + 3
        if self.use_res:
            self.skipconv = conv_block1d(in_channels, channels[-1], norm=None, act=False) \
                if in_channels != channels[-1] else nn.Identity()
            self.act = nn.ReLU(inplace=True)
        create = conv_block1d if is_head else conv_block2d
        convs = [create(channels[i], channels[i + 1], norm=None if is_head else norm,
                        act=False if (i == len(channels) - 2 and (self.use_res or is_head))
                        else (True if act else False))
                 for i in range(len(channels) - 1)]
        self.convs = nn.Sequential(*convs)

    def forward(self, p, f):
        if self.is_head:
            return p, self.convs(f)
        if self.all_aggr:
            new_p, idx = p, None
        else:
            idx = furthest_point_sample(p, p.shape[1] // self.stride) if self.sampler == 'fps' \
                else random_sample(p, p.shape[1] // self.stride)
            new_p = p.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))
        identity = None
        if self.use_res:
            fi = f.gather(2, idx.unsqueeze(1).expand(-1, f.shape[1], -1))
            identity = self.skipconv(fi)
        out = self.convs(local_aggregate(new_p, p, f, self.k)).max(-1)[0]
        if self.use_res:
            out = self.act(out + identity)
        return new_p, out


class InvResMLP(nn.Module):
    def __init__(self, in_channels, k=32, norm='bn', act=True, expansion=1, use_res=True, num_posconvs=2):
        super().__init__()
        self.use_res = use_res
        mid_channels = int(in_channels * expansion)
        self.convs = LocalAggregation(in_channels, in_channels, layers=1, k=k, norm=norm, act=act)
        if num_posconvs < 1:
            channels = []
        elif num_posconvs == 1:
            channels = [in_channels, in_channels]
        else:
            channels = [in_channels, mid_channels, in_channels]
        pwconv = [conv_block1d(channels[i], channels[i + 1], norm=norm,
                               act=True if (i < len(channels) - 2 and act) else False)
                  for i in range(len(channels) - 1)]
        self.pwconv = nn.Sequential(*pwconv)
        self.act = nn.ReLU(inplace=True)

    def forward(self, p, f):
        identity = f
        f = self.pwconv(self.convs(p, f))
        if f.shape[-1] == identity.shape[-1] and self.use_res:
            f = f + identity
        return p, self.act(f)


class PointNextEncoder(nn.Module):
    def __init__(self, in_channels=3, width=32, blocks=(1, 1, 1, 1, 1), strides=(1, 4, 4, 4, 4),
                 layers=1, expansion=4, k=32, norm='bn', act=True, sa_use_res=True, sampler='random'):
        super().__init__()
        channels = []
        w = width
        for stride in strides:
            if stride != 1:
                w *= 2
            channels.append(w)
        self.channel_list = channels
        self.in_channels = in_channels
        encoder = []
        for i in range(len(blocks)):
            is_head = (i == 0 and strides[i] == 1)
            encoder.append(SetAbstraction(self.in_channels, channels[i], layers=layers, stride=strides[i],
                                          k=k, norm=norm, act=act, is_head=is_head,
                                          use_res=sa_use_res and not is_head, sampler=sampler))
            self.in_channels = channels[i]
            for _ in range(1, blocks[i]):
                encoder.append(InvResMLP(channels[i], k=k, norm=norm, act=act, expansion=expansion, use_res=True))
        self.encoder = nn.Sequential(*encoder)
        self.out_channels = channels[-1]

    def forward_seg_feat(self, data):
        p0, f0 = data['pos'], data.get('x', None)
        if f0 is None:
            f0 = p0.transpose(1, 2).contiguous()
        p, f = [p0], [f0]
        for layer in self.encoder:
            _p, _f = layer(p[-1], f[-1])
            p.append(_p)
            f.append(_f)
        return p, f


class FeaturePropagation(nn.Module):
    def __init__(self, mlp, norm='bn', act=True):
        super().__init__()
        self.convs = nn.Sequential(*[conv_block1d(mlp[i], mlp[i + 1], norm=norm, act=act)
                                     for i in range(len(mlp) - 1)])

    def forward(self, p1, f1, p2, f2):
        up = three_interpolation(p1, p2, f2)
        f = torch.cat((f1, up), dim=1) if f1 is not None else up
        return self.convs(f)


class PointNextDecoder(nn.Module):
    def __init__(self, encoder_channel_list, in_channels=3, decoder_layers=2, decoder_stages=4, norm='bn', act=True):
        super().__init__()
        self.in_channels = encoder_channel_list[-1]
        skip_channels = list(encoder_channel_list[:-1])
        if len(skip_channels) < decoder_stages:
            skip_channels.insert(0, in_channels)
        fp_channels = list(encoder_channel_list[:decoder_stages])
        decoders = [None] * len(fp_channels)
        for i in range(-1, -len(fp_channels) - 1, -1):
            mlp = [skip_channels[i] + self.in_channels] + [fp_channels[i]] * decoder_layers
            decoders[i] = FeaturePropagation(mlp, norm=norm, act=act)
            self.in_channels = fp_channels[i]
        self.decoder = nn.Sequential(*decoders)
        self.out_channels = fp_channels[-len(fp_channels)]

    def forward(self, p, f):
        f = list(f)
        for i in range(-1, -len(self.decoder) - 1, -1):
            f[i - 1] = self.decoder[i](p[i - 1], f[i - 1], p[i], f[i])
        return f[-len(self.decoder) - 1]


class SegHead(nn.Module):
    def __init__(self, num_classes, in_channels, mlps=None, norm='bn', act=True, dropout=0.5):
        super().__init__()
        mlps = [in_channels, in_channels, num_classes] if mlps is None else [in_channels] + mlps + [num_classes]
        heads = []
        for i in range(len(mlps) - 2):
            heads.append(conv_block1d(mlps[i], mlps[i + 1], norm=norm, act=act))
            if dropout:
                heads.append(nn.Dropout(dropout))
        heads.append(conv_block1d(mlps[-2], mlps[-1], norm=None, act=False))
        self.head = nn.Sequential(*heads)

    def forward(self, x):
        return self.head(x)


def replace_bn_with_gn(module, groups=8):
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            channels = child.num_features
            setattr(module, name, nn.GroupNorm(groups if channels % groups == 0 else 1, channels))
        else:
            replace_bn_with_gn(child, groups)
    return module


class ActorCritic(nn.Module):
    def __init__(self, pc, descriptor, hidden=128):
        super().__init__()
        self.state = nn.Sequential(nn.Linear(pc, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.candidate = nn.Sequential(nn.Linear(descriptor, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.actor = nn.Linear(hidden, 1)
        self.critic = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, global_feature, descriptors, sample):
        state, candidate = self.state(global_feature), self.candidate(descriptors)
        logits = self.actor(torch.tanh(state[:, None] + candidate)).squeeze(-1)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample() if sample else logits.argmax(-1)
        return action, (dist.log_prob(action) if sample else None), (dist.entropy() if sample else None), \
            self.critic(torch.cat((state, candidate.mean(1)), -1)).squeeze(-1)


class DepthEncoder(nn.Module):
    def __init__(self, c, stem=16):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, stem, 3, padding=1), nn.BatchNorm2d(stem), nn.GELU(),
                                 nn.Conv2d(stem, c, 3, 2, 1), nn.BatchNorm2d(c), nn.GELU(),
                                 nn.Conv2d(c, c, 3, 2, 1), nn.BatchNorm2d(c), nn.GELU())

    def forward(self, x):
        local = self.net(x)
        return local, local.mean((2, 3))


class ExactCoalitionGame(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.mask = nn.Embedding(8, c)
        self.fuse = nn.Sequential(nn.Conv1d(c, c, 1), nn.BatchNorm1d(c), nn.GELU(), nn.Conv1d(c, c, 1))
        self.value = nn.Sequential(nn.Linear(c * 2, c), nn.GELU(), nn.Linear(c, 1))
        self.residual = nn.Conv1d(c, c, 1)
        self.register_buffer('mask_tensor',
                             torch.tensor(MASKS, dtype=torch.float32).view(8, len(MASKS[0]), 1, 1), persistent=False)
        self.register_buffer('mask_ids', torch.arange(len(MASKS)), persistent=False)

    def all(self, effects):
        b, p, c, n = effects.shape
        masked = (effects[:, None] * self.mask_tensor.view(1, len(MASKS), p, 1, 1)).sum(2) + \
            self.mask.weight.view(1, len(MASKS), c, 1)
        f = self.fuse(masked.reshape(b * len(MASKS), c, n)).reshape(b, len(MASKS), c, n)
        ids = self.mask_ids.view(1, len(MASKS)).expand(b, len(MASKS)).reshape(b * len(MASKS))
        values = self.value(torch.cat((f.mean(-1).reshape(b * len(MASKS), c), self.mask(ids)), -1)) \
            .reshape(b, len(MASKS))
        return f, values

    def full(self, base, scale, phi, effects):
        return base + scale * self.residual((effects * phi[:, :, None, None]).sum(1))


class PointNextMultimodalRLShapley(nn.Module):
    def __init__(self, encoder_args=None, decoder_args=None, cls_args=None, text_dim=1024, image_channels=64,
                 descriptor_channels=8, group_norm=True, gn_groups=8, gate_init=0.0, norm_mode=None,
                 main_multitask=False, ablate_text=False, ablate_image=False, ablate_shapley=False,
                 ablate_rl=False, **kwargs):
        super().__init__()
        ea = dict(encoder_args or {})
        ea.pop('NAME', None)
        norm = (ea.pop('norm_args', {}) or {}).get('norm', 'bn')
        act = (ea.pop('act_args', {}) or {}).get('act', 'relu') == 'relu'
        ga = ea.pop('group_args', {}) or {}
        k = int(ea.pop('nsample', ga.get('nsample', 32)))
        sampler = ea.pop('sampler', 'random')
        ea.pop('radius', None)
        ea.pop('aggr_args', None)
        ea.pop('conv_args', None)
        self.encoder = PointNextEncoder(in_channels=int(ea.pop('in_channels', 3)),
                                        width=int(ea.pop('width', 32)),
                                        blocks=tuple(ea.pop('blocks', (1, 1, 1, 1, 1))),
                                        strides=tuple(ea.pop('strides', (1, 4, 4, 4, 4))),
                                        layers=int(ea.pop('sa_layers', 1)),
                                        expansion=int(ea.pop('expansion', 4)), k=k, norm=norm, act=act,
                                        sa_use_res=bool(ea.pop('sa_use_res', True)), sampler=sampler)
        da = dict(decoder_args or {})
        da.pop('NAME', None)
        self.decoder = PointNextDecoder(self.encoder.channel_list, in_channels=self.encoder.in_channels,
                                        decoder_layers=int(da.pop('decoder_layers', 2)),
                                        decoder_stages=int(da.pop('decoder_stages', 4)), norm=norm, act=act)
        self.c = self.decoder.out_channels
        ca = dict(cls_args or {})
        ca.pop('NAME', None)
        ca.pop('in_channels', None)
        cnorm = (ca.pop('norm_args', {}) or {}).get('norm', 'bn')
        cact = (ca.pop('act_args', {}) or {}).get('act', 'relu') == 'relu'
        cmlps = ca.pop('mlps', None)
        cdrop = ca.pop('dropout', 0.5)
        self.num_classes = int(ca.pop('num_classes', 2))
        self.head = SegHead(self.num_classes, self.c, mlps=cmlps, norm=cnorm, act=cact, dropout=cdrop)
        self.coalition_head = SegHead(self.num_classes, self.c, mlps=cmlps, norm=cnorm, act=cact, dropout=cdrop)
        deepest = self.encoder.channel_list[-1]
        self.agent = ActorCritic(deepest, descriptor_channels)
        self.image = DepthEncoder(image_channels)
        self.img_global = nn.Linear(image_channels, self.c)
        self.img_local = nn.Conv1d(image_channels, self.c, 1)
        self.text = nn.Sequential(nn.Linear(text_dim, self.c), nn.LayerNorm(self.c))
        self.film = nn.Linear(self.c, self.c * 2)
        self.game = ExactCoalitionGame(self.c)
        self.point_proj = nn.Linear(deepest, image_channels)
        self.image_proj = nn.Linear(image_channels, image_channels)
        self.text_proj = nn.Linear(self.c, image_channels)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.main_multitask = bool(main_multitask)
        self.ablate_text = bool(ablate_text)
        self.ablate_image = bool(ablate_image)
        self.ablate_shapley = bool(ablate_shapley)
        self.ablate_rl = bool(ablate_rl)
        self.inference_modalities = 'point' if self.main_multitask else 'point+image+text'
        mode = norm_mode if norm_mode is not None else ('gn' if group_norm else 'bn')
        self.norm_mode = mode
        if mode == 'gn':
            subs = (self.encoder, self.decoder, self.head, self.coalition_head, self.image, self.game)
        elif mode == 'bn_encoder':
            subs = (self.head, self.coalition_head, self.image, self.game)
        else:
            subs = ()
        for sub in subs:
            replace_bn_with_gn(sub, gn_groups)

    @staticmethod
    def _grid(pos, view):
        a, e = view[:, 0], view[:, 1]
        ca, sa = torch.cos(a), torch.sin(a)
        ce, se = torch.cos(e), torch.sin(e)
        ry = torch.stack((torch.stack((ca, torch.zeros_like(a), sa), -1),
                          torch.stack((torch.zeros_like(a), torch.ones_like(a), torch.zeros_like(a)), -1),
                          torch.stack((-sa, torch.zeros_like(a), ca), -1)), 1)
        rx = torch.stack((torch.stack((torch.ones_like(e), torch.zeros_like(e), torch.zeros_like(e)), -1),
                          torch.stack((torch.zeros_like(e), ce, -se), -1),
                          torch.stack((torch.zeros_like(e), se, ce), -1)), 1)
        p = torch.bmm(torch.bmm(rx, ry), pos.transpose(1, 2)).transpose(1, 2)
        return torch.stack((p[..., 0].clamp(-1, 1), (-p[..., 2]).clamp(-1, 1)), -1)

    def encode_and_select(self, data, sample_actions):
        positions, pyramid = self.encoder.forward_seg_feat(data)
        global_feature = pyramid[-1].mean(-1)
        if self.ablate_rl:
            action = torch.zeros(data['pos'].size(0), dtype=torch.long, device=data['pos'].device)
            logp = entropy = critic = None
        else:
            action, logp, entropy, critic = self.agent(global_feature, data['candidate_descriptors'], sample_actions)
        view = data['candidate_views'][torch.arange(action.size(0), device=action.device), action]
        return dict(positions=positions, pyramid=pyramid, global_feature=global_feature, view_index=action,
                    view_params=view, log_prob=logp, entropy=entropy, critic_value=critic)

    def forward_point_only(self, data):
        positions, pyramid = self.encoder.forward_seg_feat(data)
        return {'logits': self.head(self.decoder(positions, pyramid))}

    def forward_selected(self, data, state, depth, enable_modalities=True):
        decoded = self.decoder(state['positions'], state['pyramid'])
        image_map, image_global = self.image(depth)
        grid = self._grid(data['pos'], state['view_params'])
        local = F.grid_sample(image_map, grid.unsqueeze(2), align_corners=False).squeeze(-1)
        image_effect = self.img_local(local) + self.img_global(image_global).unsqueeze(-1)
        text = self.text(data['text_embedding'])
        gamma, beta = self.film(text).chunk(2, -1)
        text_effect = gamma.unsqueeze(-1) * decoded + beta.unsqueeze(-1)
        m_text = 0.0 if (self.ablate_text or not enable_modalities) else 1.0
        m_image = 0.0 if (self.ablate_image or not enable_modalities) else 1.0
        effects = torch.stack((decoded, text_effect * m_text, image_effect * m_image), 1)
        b, c, n = decoded.shape
        if self.ablate_shapley:
            k = len(MASKS)
            features = torch.zeros(b, k, c, n, device=decoded.device, dtype=decoded.dtype)
            values = torch.zeros(b, k, device=decoded.device, dtype=decoded.dtype)
            phi = torch.zeros(b, 3, device=decoded.device, dtype=decoded.dtype)
            full = decoded
        else:
            features, values = self.game.all(effects)
            phi = exact_shapley(values)
            full = self.game.full(decoded, torch.tanh(self.gate), phi, effects)
        k = features.size(1)
        if self.main_multitask:
            logits = self.head(decoded)
            point_logits = logits
            classes = logits.size(1)
        else:
            both = self.head(torch.cat((full.unsqueeze(1), decoded.unsqueeze(1)), 1).reshape(b * 2, c, n))
            both = both.reshape(b, 2, both.size(1), n)
            logits, point_logits = both[:, 0], both[:, 1]
            classes = both.size(2)
        if self.ablate_shapley:
            coalition = torch.zeros(b, k, classes, n, device=decoded.device, dtype=decoded.dtype)
        else:
            coalition = self.coalition_head(features.reshape(b * k, c, n)).reshape(b, k, classes, n)
        return {**state, 'logits': logits, 'point_logits': point_logits, 'coalition_logits': coalition,
                'coalition_utilities': values, 'shapley': phi, 'gate': torch.tanh(self.gate).detach(),
                'ablate': {'text': self.ablate_text, 'image': self.ablate_image,
                           'shapley': self.ablate_shapley, 'rl': self.ablate_rl},
                'point_embedding': F.normalize(self.point_proj(state['global_feature']), dim=-1),
                'image_embedding': F.normalize(self.image_proj(image_global), dim=-1),
                'text_embedding_projected': F.normalize(self.text_proj(text), dim=-1)}


class MultimodalLoss(nn.Module):
    def __init__(self, temperature=.07, contrastive_weight=.2, utility_weight=.1, actor_weight=.05,
                 entropy_weight=.001):
        super().__init__()
        self.temperature, self.cw, self.uw, self.aw, self.ew = \
            temperature, contrastive_weight, utility_weight, actor_weight, entropy_weight

    @staticmethod
    def per_sample_ce(logits, labels):
        return F.cross_entropy(logits, labels, reduction='none').mean(-1)

    def multi_positive(self, query, key, query_id, key_id):
        logits = query @ key.t() / self.temperature
        positive = query_id[:, None].eq(key_id[None])
        positive.fill_diagonal_(True)
        return -(torch.logsumexp(logits.masked_fill(~positive, float('-inf')), 1)
                 - torch.logsumexp(logits, 1)).mean()

    def forward(self, out, batch, rl_active, aux_scale=1.0):
        seg = F.cross_entropy(out['logits'], batch['y'])
        point_ce = self.per_sample_ce(out['point_logits'], batch['y'])
        full_ce = self.per_sample_ce(out['logits'], batch['y'])
        ids = batch['sample_index']
        zero = seg.new_zeros(())
        contrast, utility = zero, zero
        flags = out.get('ablate') or {}
        if aux_scale > 0:
            tid = batch['text_id']
            p, i, t = out['point_embedding'], out['image_embedding'], out['text_embedding_projected']
            pairs = []
            if not flags.get('image'):
                pairs.append(self.multi_positive(p, i, ids, ids))
            if not flags.get('text'):
                pairs.append(self.multi_positive(p, t, tid, tid))
            if not flags.get('text') and not flags.get('image'):
                pairs.append(self.multi_positive(i, t, tid, tid))
            if pairs:
                contrast = sum(pairs) / len(pairs)
            if not flags.get('shapley'):
                b, k, c, n = out['coalition_logits'].shape
                targets = batch['y'][:, None].expand(b, k, n).reshape(b * k, n)
                coalition_ce = F.cross_entropy(out['coalition_logits'].reshape(b * k, c, n), targets,
                                               reduction='none').mean(-1).reshape(b, k)
                utility = F.mse_loss(out['coalition_utilities'], -coalition_ce.detach())
        geometry = batch['candidate_descriptors'][torch.arange(ids.size(0), device=ids.device),
                                                  out['view_index'], 0]
        reward = (point_ce - full_ce).detach() + .25 * geometry.detach()
        actor = critic = zero
        if rl_active and out['log_prob'] is not None:
            advantage = reward - out['critic_value']
            actor = -(advantage.detach() * out['log_prob']).mean() - self.ew * out['entropy'].mean()
            critic = advantage.pow(2).mean()
        total = seg + aux_scale * (self.cw * contrast + self.uw * utility) + \
            (self.aw * (actor + critic) if rl_active else 0.)
        return total, {'loss': total.detach(), 'seg': seg.detach(), 'contrastive': contrast.detach(),
                       'utility': utility.detach(), 'actor': actor.detach(), 'reward': reward.mean().detach()}


def build_model(cfg):
    cfg = dict(cfg)
    cfg.pop('NAME', None)
    return PointNextMultimodalRLShapley(**cfg)


def replace_bn_with_gn_in(model, groups=8):
    return replace_bn_with_gn(model, groups)
