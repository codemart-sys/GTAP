"""PointNeXt with frozen text, lazy depth views, Actor-Critic and exact Shapley.

v1.4, on top of v1.3. The three contributions keep their meaning -- (1) point / image / text
three-modal fusion, (2) exact eight-coalition Shapley attribution without softmax, (3)
Actor-Critic selection over the Fibonacci candidates -- and two switches were added to test where
the residual accuracy gap to the plain baseline came from:

1. `main_multitask`: true gives the main output as `head(decoded)`, a pure point branch shaped like
   the baseline's, with the multimodal terms purely auxiliary; false keeps v1.3's gated
   `decoded + Shapley residual`.
2. `norm_mode`: `gn` (v1.2/v1.3 behaviour), `bn_encoder` (encoder+decoder keep BatchNorm; heads,
   depth encoder and game get GroupNorm -- the current best) or `bn` (nothing replaced).

The trainer also gained `--seed`, and the module gained the ablation switches used by
`I:\\two\\ablation` (all default False, so the default behaviour is unchanged):

* `ablate_text`    -- text is zeroed out of `effects` and dropped from the contrastive pairs;
* `ablate_image`   -- image is zeroed out of `effects` and dropped from the contrastive pairs;
* `ablate_shapley` -- no coalition enumeration, no Shapley values, no utility calibration, no residual;
* `ablate_rl`      -- the view is fixed to candidate 0 and the Actor-Critic is not trained.

**Which modalities do inference need?** With `main_multitask: true` the logits are
`head(decoder(points))`, so the text embedding, the depth view and the Shapley game are all
*training-side only*: `self.inference_modalities == 'point'`. Use `forward_point_only(data)` for that
path — it needs no `assets/`, no BGE-M3, no view cache and no Actor-Critic (see
`infer_pointcloud.py`). With `main_multitask: false` the logits *do* depend on the fused
`decoded + Shapley residual`, hence `inference_modalities == 'point+image+text'`.

See `I:\\two\\md\\09_V14_CHANGES.md` and `I:\\two\\md\\10_MULTIDATASET_ABLATION.md` for the measured
comparisons.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from openpoints.models.build import MODELS, build_model_from_cfg
from .shapley import MASKS, MASK_ID, exact_shapley

def replace_bn_with_gn(module, groups=8):
    """Replace every BatchNorm* in `module` with a GroupNorm of the same channel count, in place."""
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            channels = child.num_features
            g = groups if channels % groups == 0 else 1
            setattr(module, name, nn.GroupNorm(g, channels))
        else:
            replace_bn_with_gn(child, groups)
    return module

class ActorCritic(nn.Module):
    def __init__(self, pc, descriptor, hidden=128):
        super().__init__()
        self.state = nn.Sequential(nn.Linear(pc,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.candidate = nn.Sequential(nn.Linear(descriptor,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.actor = nn.Linear(hidden,1); self.critic = nn.Sequential(nn.Linear(hidden*2,hidden),nn.GELU(),nn.Linear(hidden,1))
    def forward(self, global_feature, descriptors, sample):
        state, candidate = self.state(global_feature), self.candidate(descriptors)
        logits = self.actor(torch.tanh(state[:,None]+candidate)).squeeze(-1); dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample() if sample else logits.argmax(-1)
        return action, (dist.log_prob(action) if sample else None), (dist.entropy() if sample else None), self.critic(torch.cat((state,candidate.mean(1)),-1)).squeeze(-1)

class DepthEncoder(nn.Module):
    """16-channel stem; norm is decided by the owner module through `norm_mode`."""
    def __init__(self, c, stem=16):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(1,stem,3,padding=1),nn.BatchNorm2d(stem),nn.GELU(),nn.Conv2d(stem,c,3,2,1),nn.BatchNorm2d(c),nn.GELU(),nn.Conv2d(c,c,3,2,1),nn.BatchNorm2d(c),nn.GELU())
    def forward(self,x):
        local=self.net(x); return local, local.mean((2,3))

class ExactCoalitionGame(nn.Module):
    """Eight coalition characteristic values; no learned pseudo-Shapley vector.

    `all` produces the eight masked modality stacks and runs `fuse`/`value` once for the whole
    (batch x 8) group. `full` takes the base feature and a gate scale, so the caller can start
    from the decoder output and let the attribution term grow.
    """
    def __init__(self,c):
        super().__init__(); self.mask=nn.Embedding(8,c); self.fuse=nn.Sequential(nn.Conv1d(c,c,1),nn.BatchNorm1d(c),nn.GELU(),nn.Conv1d(c,c,1)); self.value=nn.Sequential(nn.Linear(c*2,c),nn.GELU(),nn.Linear(c,1)); self.residual=nn.Conv1d(c,c,1)
        self.register_buffer('mask_tensor',torch.tensor(MASKS,dtype=torch.float32).view(8,len(MASKS[0]),1,1),persistent=False)
        self.register_buffer('mask_ids',torch.arange(len(MASKS)),persistent=False)
    def all(self,effects):
        b,p,c,n=effects.shape
        masked=(effects[:,None]*self.mask_tensor.view(1,len(MASKS),p,1,1)).sum(2)+self.mask.weight.view(1,len(MASKS),c,1)
        f=self.fuse(masked.reshape(b*len(MASKS),c,n)).reshape(b,len(MASKS),c,n)
        ids=self.mask_ids.view(1,len(MASKS)).expand(b,len(MASKS)).reshape(b*len(MASKS))
        values=self.value(torch.cat((f.mean(-1).reshape(b*len(MASKS),c),self.mask(ids)),-1)).reshape(b,len(MASKS))
        return f,values
    def full(self,base,scale,phi,effects): return base+scale*self.residual((effects*phi[:,:,None,None]).sum(1))

@MODELS.register_module()
class PointNextMultimodalRLShapley(nn.Module):
    def __init__(self,encoder_args,decoder_args,cls_args,text_dim=1024,image_channels=64,descriptor_channels=8,group_norm=True,gn_groups=8,gate_init=0.0,norm_mode=None,main_multitask=False,ablate_text=False,ablate_image=False,ablate_shapley=False,ablate_rl=False,**_):
        super().__init__(); self.encoder=build_model_from_cfg(encoder_args); args=copy.deepcopy(encoder_args); args.update(copy.deepcopy(decoder_args)); args.encoder_channel_list=self.encoder.channel_list; self.decoder=build_model_from_cfg(args); self.c=self.decoder.out_channels
        head=copy.deepcopy(cls_args); head.in_channels=self.c; self.head=build_model_from_cfg(head)
        chead=copy.deepcopy(cls_args); chead.in_channels=self.c; self.coalition_head=build_model_from_cfg(chead)
        deepest=self.encoder.channel_list[-1]
        self.agent=ActorCritic(deepest,descriptor_channels); self.image=DepthEncoder(image_channels); self.img_global=nn.Linear(image_channels,self.c); self.img_local=nn.Conv1d(image_channels,self.c,1)
        self.text=nn.Sequential(nn.Linear(text_dim,self.c),nn.LayerNorm(self.c)); self.film=nn.Linear(self.c,self.c*2); self.game=ExactCoalitionGame(self.c)
        self.point_proj=nn.Linear(deepest,image_channels); self.image_proj=nn.Linear(image_channels,image_channels); self.text_proj=nn.Linear(self.c,image_channels)
        self.gate=nn.Parameter(torch.tensor(float(gate_init)))          # residual gate, starts closed
        self.main_multitask=bool(main_multitask)
        self.ablate_text=bool(ablate_text); self.ablate_image=bool(ablate_image); self.ablate_shapley=bool(ablate_shapley); self.ablate_rl=bool(ablate_rl)
        # what the segmentation output actually depends on at inference time
        self.inference_modalities='point' if self.main_multitask else 'point+image+text'
        mode=norm_mode if norm_mode is not None else ('gn' if group_norm else 'bn')
        self.norm_mode=mode
        if mode=='gn': subs=(self.encoder,self.decoder,self.head,self.coalition_head,self.image,self.game)
        elif mode=='bn_encoder': subs=(self.head,self.coalition_head,self.image,self.game)
        else: subs=()
        for sub in subs: replace_bn_with_gn(sub,gn_groups)
    @staticmethod
    def _grid(pos,view):
        a,e=view[:,0],view[:,1]; ca,sa=torch.cos(a),torch.sin(a); ce,se=torch.cos(e),torch.sin(e)
        ry=torch.stack((torch.stack((ca,torch.zeros_like(a),sa),-1),torch.stack((torch.zeros_like(a),torch.ones_like(a),torch.zeros_like(a)),-1),torch.stack((-sa,torch.zeros_like(a),ca),-1)),1)
        rx=torch.stack((torch.stack((torch.ones_like(e),torch.zeros_like(e),torch.zeros_like(e)),-1),torch.stack((torch.zeros_like(e),ce,-se),-1),torch.stack((torch.zeros_like(e),se,ce),-1)),1)
        p=torch.bmm(torch.bmm(rx,ry),pos.transpose(1,2)).transpose(1,2); return torch.stack((p[...,0].clamp(-1,1),(-p[...,2]).clamp(-1,1)),-1)
    def encode_and_select(self,data,sample_actions):
        positions,pyramid=self.encoder.forward_seg_feat(data); global_feature=pyramid[-1].mean(-1)
        if self.ablate_rl:
            action=torch.zeros(data['pos'].size(0),dtype=torch.long,device=data['pos'].device); logp=entropy=critic=None   # fixed view, no Actor-Critic
        else:
            action,logp,entropy,critic=self.agent(global_feature,data['candidate_descriptors'],sample_actions)
        view=data['candidate_views'][torch.arange(action.size(0),device=action.device),action]
        return dict(positions=positions,pyramid=pyramid,global_feature=global_feature,view_index=action,view_params=view,log_prob=logp,entropy=entropy,critic_value=critic)
    def forward_point_only(self,data):
        """Inference path when the logits depend on the point cloud only (`main_multitask: true`).

        Needs just `{'pos': (B,N,3), 'x': (B,3,N)}` — no text embedding, no depth view, no view cache,
        no Actor-Critic. This is the fastest and least-deployable-dependency way to run the model.
        """
        positions,pyramid=self.encoder.forward_seg_feat(data)
        decoded=self.decoder(positions,pyramid).squeeze(-1)
        return {'logits':self.head(decoded)}
    def forward_selected(self,data,state,depth,enable_modalities=True):
        decoded=self.decoder(state['positions'],state['pyramid']).squeeze(-1); image_map,image_global=self.image(depth); grid=self._grid(data['pos'],state['view_params']); local=F.grid_sample(image_map,grid.unsqueeze(2),align_corners=False).squeeze(-1)
        image_effect=self.img_local(local)+self.img_global(image_global).unsqueeze(-1); text=self.text(data['text_embedding']); gamma,beta=self.film(text).chunk(2,-1); text_effect=gamma.unsqueeze(-1)*decoded+beta.unsqueeze(-1)
        m_text=0.0 if (self.ablate_text or not enable_modalities) else 1.0
        m_image=0.0 if (self.ablate_image or not enable_modalities) else 1.0
        effects=torch.stack((decoded,text_effect*m_text,image_effect*m_image),1)
        b,c,n=decoded.shape
        if self.ablate_shapley:                                          # ablation: no coalition game at all
            k=len(MASKS); features=torch.zeros(b,k,c,n,device=decoded.device,dtype=decoded.dtype); values=torch.zeros(b,k,device=decoded.device,dtype=decoded.dtype); phi=torch.zeros(b,3,device=decoded.device,dtype=decoded.dtype); full=decoded
        else:
            features,values=self.game.all(effects); phi=exact_shapley(values)
            full=self.game.full(decoded,torch.tanh(self.gate),phi,effects)
        k=features.size(1)
        if self.main_multitask:
            logits=self.head(decoded); point_logits=logits; classes=logits.size(1)
        else:
            both=self.head(torch.cat((full.unsqueeze(1),decoded.unsqueeze(1)),1).reshape(b*2,c,n)).reshape(b,2,-1,n)
            logits,point_logits=both[:,0],both[:,1]; classes=both.size(2)
        if self.ablate_shapley:
            coalition=torch.zeros(b,k,classes,n,device=decoded.device,dtype=decoded.dtype)
        else:
            coalition=self.coalition_head(features.reshape(b*k,c,n)).reshape(b,k,classes,n)
        return {**state,'logits':logits,'point_logits':point_logits,'coalition_logits':coalition,'coalition_utilities':values,'shapley':phi,'gate':torch.tanh(self.gate).detach(),'ablate':{'text':self.ablate_text,'image':self.ablate_image,'shapley':self.ablate_shapley,'rl':self.ablate_rl},'point_embedding':F.normalize(self.point_proj(state['global_feature']),dim=-1),'image_embedding':F.normalize(self.image_proj(image_global),dim=-1),'text_embedding_projected':F.normalize(self.text_proj(text),dim=-1)}
