import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from openpoints.models.build import MODELS, build_model_from_cfg
from .attribution import MASKS, MASK_ID, MixController, ShapleyAmortizer, exact_shapley

def replace_bn_with_gn(module, groups=8):
    for (name, child) in module.named_children():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            channels = child.num_features
            group_count = groups if channels % groups == 0 else 1
            setattr(module, name, nn.GroupNorm(group_count, channels))
        else:
            replace_bn_with_gn(child, groups)
    return module

class ActorCritic(nn.Module):

    def __init__(self, point_dim, descriptor_dim, hidden=96):
        super().__init__()
        self.state = nn.Sequential(nn.Linear(point_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.candidate = nn.Sequential(nn.Linear(descriptor_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.actor = nn.Linear(hidden, 1)
        self.critic = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, global_feature, descriptors, sample=True):
        state = self.state(global_feature)
        candidates = self.candidate(descriptors)
        logits = self.actor(torch.tanh(state[:, None] + candidates)).squeeze(-1)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample() if sample else logits.argmax(-1)
        critic = self.critic(torch.cat((state, candidates.mean(1)), dim=-1)).squeeze(-1)
        return (action, distribution.log_prob(action) if sample else None, distribution.entropy() if sample else None, critic, logits)

class TinyDepthEncoder(nn.Module):

    def __init__(self, channels, stem=8, input_size=64):
        super().__init__()
        self.input_size = int(input_size)
        self.net = nn.Sequential(nn.Conv2d(2, stem, 3, padding=1), nn.BatchNorm2d(stem), nn.GELU(), nn.Conv2d(stem, channels, 3, stride=2, padding=1), nn.BatchNorm2d(channels), nn.GELU(), nn.Conv2d(channels, channels, 3, stride=2, padding=1), nn.BatchNorm2d(channels), nn.GELU())

    def forward(self, depth):
        if depth.shape[-2:] != (self.input_size, self.input_size):
            depth = F.interpolate(depth, size=(self.input_size, self.input_size), mode='bilinear', align_corners=True)
        local = self.net(depth)
        return (local, local.mean((2, 3)))

class FusionLogitAdapter(nn.Module):

    def __init__(self, channels, classes, groups=8, init_scale=0.03, max_delta=0.5):
        super().__init__()
        group_count = groups if channels % groups == 0 else 1
        in_channels = channels * 3 + 1
        self.trunk = nn.Sequential(nn.Conv1d(in_channels, channels, 1), nn.GroupNorm(group_count, channels), nn.GELU(), nn.Conv1d(channels, channels, 1), nn.GroupNorm(group_count, channels), nn.GELU())
        self.delta = nn.Conv1d(channels, classes, 1)
        self.reliability = nn.Conv1d(in_channels, 1, 1)
        self.max_delta = float(max_delta)
        if self.max_delta <= 0:
            raise ValueError('fusion_max_delta must be positive')
        ratio = min(max(float(init_scale) / self.max_delta, 0.0001), 1.0 - 0.0001)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio))))

    def forward(self, point, text, image, visibility, base_logits):
        raw = torch.cat((point, text, image, visibility), dim=1)
        hidden = self.trunk(raw)
        gate_logits = self.reliability(raw)
        gate = torch.sigmoid(gate_logits)
        scale = self.max_delta * torch.sigmoid(self.logit_scale)
        proposal = base_logits + scale * torch.tanh(self.delta(hidden))
        return (base_logits + gate * (proposal - base_logits), gate, proposal, scale, gate_logits)

class ExactCoalitionGame(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.mask_embedding = nn.Embedding(len(MASKS), channels)
        self.fuse = nn.Sequential(nn.Conv1d(channels, channels, 1), nn.BatchNorm1d(channels), nn.GELU(), nn.Conv1d(channels, channels, 1), nn.BatchNorm1d(channels), nn.GELU())
        self.register_buffer('mask_tensor', torch.tensor(MASKS, dtype=torch.float32).view(len(MASKS), 3, 1, 1), persistent=False)
        self.register_buffer('mask_ids', torch.arange(len(MASKS)), persistent=False)

    @staticmethod
    def landmark_indices(n, count, device):
        count = min(int(count), int(n))
        if count == n:
            return torch.arange(n, device=device)
        return torch.linspace(0, n - 1, count, device=device).round().long()

    def all(self, effects, landmark_count):
        (batch, players, channels, points) = effects.shape
        indices = self.landmark_indices(points, landmark_count, effects.device)
        sampled = effects[:, :, :, indices]
        masks = self.mask_tensor.to(dtype=effects.dtype)
        masked = (sampled[:, None] * masks.view(1, len(MASKS), players, 1, 1)).sum(2)
        masked = masked + self.mask_embedding.weight.view(1, len(MASKS), channels, 1)
        features = self.fuse(masked.reshape(batch * len(MASKS), channels, indices.numel()))
        return (features.reshape(batch, len(MASKS), channels, indices.numel()), indices)

@MODELS.register_module()
class PointNextMultimodalRLShapley(nn.Module):

    def __init__(self, encoder_args, decoder_args, cls_args, text_dim=1024, image_channels=32, descriptor_channels=8, norm_mode='bn_encoder', gn_groups=8, main_fusion=True, shapley_mode='mix', amort_hidden=64, amort_landmarks=24, amort_heads=4, amort_sigma=0.05, coalition_landmarks=256, fusion_init_scale=0.03, fusion_max_delta=0.5, c2_detach_backbone=True, ablate_text=False, ablate_image=False, ablate_shapley=False, ablate_rl=False, main_multitask=None, **_):
        super().__init__()
        self.encoder = build_model_from_cfg(encoder_args)
        decoder_cfg = copy.deepcopy(encoder_args)
        decoder_cfg.update(copy.deepcopy(decoder_args))
        decoder_cfg.encoder_channel_list = self.encoder.channel_list
        self.decoder = build_model_from_cfg(decoder_cfg)
        self.channels = self.decoder.out_channels
        self.num_classes = int(cls_args.num_classes)
        head_cfg = copy.deepcopy(cls_args)
        head_cfg.in_channels = self.channels
        self.head = build_model_from_cfg(head_cfg)
        coalition_head_cfg = copy.deepcopy(cls_args)
        coalition_head_cfg.in_channels = self.channels
        self.coalition_head = build_model_from_cfg(coalition_head_cfg)
        deepest = self.encoder.channel_list[-1]
        self.agent = ActorCritic(deepest, descriptor_channels)
        self.image = TinyDepthEncoder(image_channels)
        self.image_global = nn.Linear(image_channels, self.channels)
        self.image_local = nn.Conv1d(image_channels, self.channels, 1)
        self.text = nn.Sequential(nn.Linear(text_dim, self.channels), nn.LayerNorm(self.channels), nn.GELU())
        self.fusion = FusionLogitAdapter(self.channels, self.num_classes, gn_groups, fusion_init_scale, fusion_max_delta)
        self.game = ExactCoalitionGame(self.channels)
        self.point_proj = nn.Linear(deepest, image_channels)
        self.image_proj = nn.Linear(image_channels, image_channels)
        self.text_proj = nn.Linear(self.channels, image_channels)
        self.main_fusion = bool(main_fusion)
        if main_multitask is not None:
            self.main_fusion = not bool(main_multitask)
        self.ablate_text = bool(ablate_text)
        self.ablate_image = bool(ablate_image)
        self.ablate_shapley = bool(ablate_shapley)
        self.ablate_rl = bool(ablate_rl)
        self.shapley_mode = str(shapley_mode)
        if self.shapley_mode not in {'exact', 'amortized', 'mix'}:
            raise ValueError(f'Unsupported shapley_mode={self.shapley_mode!r}; expected exact, amortized, or mix')
        self.coalition_landmarks = int(coalition_landmarks)
        self.amort_sigma = float(amort_sigma)
        self.c2_detach_backbone = bool(c2_detach_backbone)
        self.amortizer = ShapleyAmortizer(c=self.channels, point_dim=deepest, image_dim=image_channels, text_dim=self.channels, hidden=amort_hidden, landmarks=amort_landmarks, heads=amort_heads) if self.shapley_mode in {'amortized', 'mix'} else None
        self.mix_controller = MixController(deepest, descriptor_channels) if self.shapley_mode == 'mix' else None
        self.inference_modalities = 'point+image+text' if self.main_fusion else 'point'
        if norm_mode == 'gn':
            modules = (self.encoder, self.decoder, self.head, self.coalition_head, self.image, self.game)
        elif norm_mode == 'bn_encoder':
            modules = (self.head, self.coalition_head, self.image, self.game)
        elif norm_mode == 'bn':
            modules = ()
        else:
            raise ValueError(f'Unknown norm_mode: {norm_mode}')
        for module in modules:
            replace_bn_with_gn(module, gn_groups)
        self.norm_mode = norm_mode

    @staticmethod
    def _grid(pos, view):
        (azimuth, elevation) = (view[:, 0], view[:, 1])
        (ca, sa) = (torch.cos(azimuth), torch.sin(azimuth))
        (ce, se) = (torch.cos(elevation), torch.sin(elevation))
        rotation_y = torch.stack((torch.stack((ca, torch.zeros_like(ca), sa), -1), torch.stack((torch.zeros_like(ca), torch.ones_like(ca), torch.zeros_like(ca)), -1), torch.stack((-sa, torch.zeros_like(ca), ca), -1)), 1)
        rotation_x = torch.stack((torch.stack((torch.ones_like(ce), torch.zeros_like(ce), torch.zeros_like(ce)), -1), torch.stack((torch.zeros_like(ce), ce, -se), -1), torch.stack((torch.zeros_like(ce), se, ce), -1)), 1)
        projected = torch.bmm(torch.bmm(rotation_x, rotation_y), pos.transpose(1, 2)).transpose(1, 2)
        return torch.stack((projected[..., 0].clamp(-1, 1), (-projected[..., 2]).clamp(-1, 1)), -1)

    @staticmethod
    def _as_per_sample(value, batch, reference):
        tensor = value if torch.is_tensor(value) else torch.as_tensor(float(value), device=reference.device)
        tensor = tensor.to(device=reference.device, dtype=reference.dtype).reshape(-1, 1)
        return tensor.expand(batch, 1) if tensor.size(0) == 1 else tensor

    @staticmethod
    def _audit_mask(batch, fraction, device):
        fraction = float(fraction)
        if fraction >= 1:
            return torch.ones(batch, dtype=torch.bool, device=device)
        if fraction <= 0:
            return torch.zeros(batch, dtype=torch.bool, device=device)
        count = max(1, int(round(batch * fraction)))
        indices = torch.randperm(batch, device=device)[:count]
        mask = torch.zeros(batch, dtype=torch.bool, device=device)
        mask[indices] = True
        return mask

    @staticmethod
    def geometry_quality(descriptors):
        return descriptors[..., 0] - descriptors[..., 1]

    def encode_and_select(self, data, sample_actions, geometry_warmup=False):
        (positions, pyramid) = self.encoder.forward_seg_feat(data)
        global_feature = pyramid[-1].mean(-1)
        descriptors = data['candidate_descriptors']
        if self.ablate_rl:
            action = torch.zeros(data['pos'].size(0), dtype=torch.long, device=data['pos'].device)
            log_prob = entropy = critic = None
            (_, _, _, _, policy_logits) = self.agent(global_feature, descriptors, sample=False)
        else:
            (sampled_action, log_prob, entropy, critic, policy_logits) = self.agent(global_feature, descriptors, sample_actions)
            action = self.geometry_quality(descriptors).argmax(-1) if geometry_warmup else sampled_action
            if geometry_warmup:
                log_prob = entropy = critic = None
        view = data['candidate_views'][torch.arange(action.size(0), device=action.device), action]
        return {'positions': positions, 'pyramid': pyramid, 'global_feature': global_feature, 'view_index': action, 'view_params': view, 'log_prob': log_prob, 'entropy': entropy, 'critic_value': critic, 'view_logits': policy_logits, 'geometry_quality': self.geometry_quality(descriptors)[torch.arange(action.size(0), device=action.device), action]}

    def select_mix(self, data, state, progress, attribution_error, exact_cost, sample=True):
        if self.mix_controller is None:
            return (0.0, None, None, None)
        return self.mix_controller(state['global_feature'], data['candidate_descriptors'], progress, attribution_error, exact_cost, sample)

    def forward_point_only(self, data):
        (positions, pyramid) = self.encoder.forward_seg_feat(data)
        decoded = self.decoder(positions, pyramid).squeeze(-1)
        return {'logits': self.head(decoded), 'decoded': decoded}

    def _image_text_effects(self, data, state, depth, decoded):
        if depth is None:
            raise ValueError('A selected two-channel depth map is required when modalities are enabled')
        depth = depth.to(dtype=decoded.dtype)
        (image_map, image_global) = self.image(depth)
        grid = self._grid(data.get('projection_pos', data['pos']), state['view_params'])
        local = F.grid_sample(image_map, grid.unsqueeze(2), align_corners=True).squeeze(-1)
        visibility = F.grid_sample(depth[:, 1:2], grid.unsqueeze(2), align_corners=True).squeeze(-1)
        image_effect = self.image_local(local) + self.image_global(image_global).unsqueeze(-1)
        text = self.text(data['text_embedding'])
        text_effect = text.unsqueeze(-1).expand_as(decoded)
        if self.ablate_image:
            image_effect = torch.zeros_like(image_effect)
            visibility = torch.zeros_like(visibility)
        if self.ablate_text:
            text_effect = torch.zeros_like(text_effect)
        return (image_effect, image_global, text, text_effect, visibility)

    def _sample_amortised_values(self, decoded, state, image_global, text, image_effect):
        if self.amortizer is None:
            return (None, None, None)
        mean = self.amortizer(decoded, state['global_feature'], image_global, text, image_effect)
        if self.training and self.amort_sigma > 0:
            distribution = torch.distributions.Normal(mean, self.amort_sigma)
            sample = distribution.sample()
            return (sample, mean, distribution.log_prob(sample).sum(-1))
        return (mean, mean, None)

    def _compact_coalitions(self, effects, batch_indices):
        if batch_indices.numel() == 0:
            return (None, None)
        (features, point_indices) = self.game.all(effects[batch_indices], self.coalition_landmarks)
        (b, coalitions, channels, points) = features.shape
        logits = self.coalition_head(features.reshape(b * coalitions, channels, points))
        return (logits.reshape(b, coalitions, self.num_classes, points), point_indices)

    def forward_selected(self, data, state, depth=None, enable_modalities=True, lam=0.0, audit_frac=1.0, exact_threshold=0.5, compute_shapley=True):
        decoded = self.decoder(state['positions'], state['pyramid']).squeeze(-1)
        point_logits = self.head(decoded)
        (batch, channels, points) = decoded.shape
        zeros_embedding = F.normalize(self.point_proj(state['global_feature']), dim=-1)
        empty = {'coalition_logits': None, 'coalition_batch_indices': None, 'coalition_point_indices': None, 'v_hat': None, 'v_hat_mu': None, 'v_logp': None, 'audit_mask': None, 'computed_mask': None, 'lam': self._as_per_sample(lam, batch, decoded), 'exact_fraction': decoded.new_zeros(()), 'amortized_choice': torch.zeros(batch, dtype=torch.bool, device=decoded.device)}
        if not enable_modalities:
            return {**state, **empty, 'logits': point_logits, 'point_logits': point_logits, 'fused_logits': point_logits, 'proposal_logits': point_logits, 'fusion_gate': decoded.new_zeros(()), 'fusion_gate_per_sample': decoded.new_zeros((batch,)), 'fusion_gate_logits_per_sample': decoded.new_zeros((batch,)), 'fusion_scale': decoded.new_zeros(()), 'point_embedding': zeros_embedding, 'image_embedding': zeros_embedding.detach() * 0, 'text_embedding_projected': zeros_embedding.detach() * 0, 'ablate': {'text': self.ablate_text, 'image': self.ablate_image, 'shapley': self.ablate_shapley, 'rl': self.ablate_rl}}
        (image_effect, image_global, text, text_effect, visibility) = self._image_text_effects(data, state, depth, decoded)
        (fused_logits, fusion_gate, proposal_logits, fusion_scale, fusion_gate_logits) = self.fusion(decoded, text_effect, image_effect, visibility, point_logits)
        logits = fused_logits if self.main_fusion else point_logits
        if self.c2_detach_backbone:
            c2_decoded = decoded.detach()
            c2_global = state['global_feature'].detach()
            c2_image_global = image_global.detach()
            c2_text = text.detach()
            c2_image_effect = image_effect.detach()
        else:
            c2_decoded = decoded
            c2_global = state['global_feature']
            c2_image_global = image_global
            c2_text = text
            c2_image_effect = image_effect
        effects = torch.stack((c2_decoded, text_effect.detach() if self.c2_detach_backbone else text_effect, c2_image_effect), dim=1)
        lam_t = self._as_per_sample(lam, batch, decoded).clamp(0, 1)
        c2_state = {**state, 'global_feature': c2_global}
        (v_hat, v_hat_mu, v_logp) = self._sample_amortised_values(c2_decoded, c2_state, c2_image_global, c2_text, c2_image_effect)
        coalition_logits = coalition_batch_indices = coalition_point_indices = None
        audit_mask = computed_mask = None
        if compute_shapley and (not self.ablate_shapley):
            if self.amortizer is None:
                computed_mask = torch.ones(batch, dtype=torch.bool, device=decoded.device)
                audit_mask = computed_mask
            else:
                audit_mask = self._audit_mask(batch, audit_frac, decoded.device)
                execution_mask = lam_t.reshape(-1).detach() < float(exact_threshold)
                computed_mask = execution_mask | audit_mask
            coalition_batch_indices = computed_mask.nonzero(as_tuple=False).flatten()
            (coalition_logits, coalition_point_indices) = self._compact_coalitions(effects, coalition_batch_indices)
        return {**state, **empty, 'logits': logits, 'point_logits': point_logits, 'fused_logits': fused_logits, 'proposal_logits': proposal_logits, 'fusion_gate': fusion_gate.detach().mean(), 'coalition_logits': coalition_logits, 'coalition_batch_indices': coalition_batch_indices, 'coalition_point_indices': coalition_point_indices, 'v_hat': v_hat, 'v_hat_mu': v_hat_mu, 'v_logp': v_logp, 'audit_mask': audit_mask, 'computed_mask': computed_mask, 'lam': lam_t, 'amortized_choice': lam_t.reshape(-1).detach() >= float(exact_threshold), 'exact_fraction': computed_mask.float().mean().detach() if computed_mask is not None else decoded.new_zeros(()), 'fusion_gate_per_sample': fusion_gate.mean((1, 2)), 'fusion_gate_logits_per_sample': fusion_gate_logits.mean((1, 2)), 'fusion_scale': fusion_scale.detach(), 'point_embedding': F.normalize(self.point_proj(state['global_feature']), dim=-1), 'image_embedding': F.normalize(self.image_proj(image_global), dim=-1), 'text_embedding_projected': F.normalize(self.text_proj(text), dim=-1), 'ablate': {'text': self.ablate_text, 'image': self.ablate_image, 'shapley': self.ablate_shapley, 'rl': self.ablate_rl}}

    def forward_point_with_attribution(self, data, state, depth):
        out = self.forward_selected(data, state, depth, enable_modalities=True, lam=1.0, audit_frac=0.0, compute_shapley=False)
        if out['v_hat'] is not None:
            values = out['v_hat_mu']
            return {'logits': out['logits'], 'coalition_utilities': values, 'shapley': exact_shapley(values), 'utility_type': 'amortised_calibrated'}
        return self.forward_exact_attribution(data, state, depth)

    def forward_exact_attribution(self, data, state, depth):
        out = self.forward_selected(data, state, depth, enable_modalities=True, lam=0.0, audit_frac=1.0, compute_shapley=True)
        confidence = F.log_softmax(out['coalition_logits'], dim=2).amax(dim=2).mean(-1)
        return {'logits': out['logits'], 'coalition_utilities': confidence, 'shapley': exact_shapley(confidence), 'utility_type': 'prediction_confidence'}
