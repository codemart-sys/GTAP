"""Losses: multi-positive contrast, segmentation, coalition utility and A2C.

v1.2 switched the contrastive/utility terms off during the point-only warmup (`aux_active`).
v1.3 replaced that switch with a continuous `aux_scale` (ramped from epoch 31 to 50) and skips the
eight coalition cross-entropies when it is zero.

v1.4 makes the auxiliary terms *ablation-aware*: the model reports which modalities / shapley / rl
are switched off in `out['ablate']`, and this loss then
  * averages the contrastive objective over the modality pairs that still exist (zero if none do),
  * skips the coalition-utility calibration when the Shapley game is ablated,
  * skips the Actor-Critic terms when no `log_prob` is produced (fixed-view ablation).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def per_sample_ce(logits,labels): return F.cross_entropy(logits,labels,reduction='none').mean(-1)
def multi_positive(query,key,query_id,key_id,temperature=.07):
 logits=query@key.t()/temperature; positive=query_id[:,None].eq(key_id[None]); positive.fill_diagonal_(True)
 return -(torch.logsumexp(logits.masked_fill(~positive,float('-inf')),1)-torch.logsumexp(logits,1)).mean()

class MultimodalLoss(nn.Module):
 def __init__(self,temperature=.07,contrastive_weight=.2,utility_weight=.1,actor_weight=.05,entropy_weight=.001):
  super().__init__(); self.temperature,self.cw,self.uw,self.aw,self.ew=temperature,contrastive_weight,utility_weight,actor_weight,entropy_weight
 def forward(self,out,batch,rl_active,aux_scale=1.0):
  seg=F.cross_entropy(out['logits'],batch['y']); point_ce=per_sample_ce(out['point_logits'],batch['y']); full_ce=per_sample_ce(out['logits'],batch['y'])
  ids=batch['sample_index']; zero=seg.new_zeros(()); contrast=zero; utility=zero; flags=out.get('ablate') or {}
  if aux_scale>0:
   tid=batch['text_id']; p,i,t=out['point_embedding'],out['image_embedding'],out['text_embedding_projected']
   pairs=[]
   if not flags.get('image'): pairs.append(multi_positive(p,i,ids,ids,self.temperature))
   if not flags.get('text'): pairs.append(multi_positive(p,t,tid,tid,self.temperature))
   if not flags.get('text') and not flags.get('image'): pairs.append(multi_positive(i,t,tid,tid,self.temperature))
   if pairs: contrast=sum(pairs)/len(pairs)
   if not flags.get('shapley'):
    # Each of the eight characteristic values is calibrated to its actual coalition segmentation payoff.
    b,k,c,n=out['coalition_logits'].shape; coalition_targets=batch['y'][:,None].expand(b,k,n).reshape(b*k,n)
    coalition_ce=F.cross_entropy(out['coalition_logits'].reshape(b*k,c,n),coalition_targets,reduction='none').mean(-1).reshape(b,k)
    utility=F.mse_loss(out['coalition_utilities'],-coalition_ce.detach())
  geometry=batch['candidate_descriptors'][torch.arange(ids.size(0),device=ids.device),out['view_index'],0]
  reward=(point_ce-full_ce).detach()+.25*geometry.detach(); actor=zero; critic=zero
  if rl_active and out['log_prob'] is not None:
   advantage=reward-out['critic_value']; actor=-(advantage.detach()*out['log_prob']).mean()-self.ew*out['entropy'].mean(); critic=advantage.pow(2).mean()
  total=seg+aux_scale*(self.cw*contrast+self.uw*utility)+(self.aw*(actor+critic) if rl_active else 0.)
  return total,{'loss':total.detach(),'seg':seg.detach(),'contrastive':contrast.detach(),'utility':utility.detach(),'actor':actor.detach(),'reward':reward.mean().detach()}
