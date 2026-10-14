import torch
import torch.nn.functional as F
from .model import PointNextMultimodalRLShapley
from .shapley import MASKS, exact_shapley

class FullExactShapley(PointNextMultimodalRLShapley):

    def __init__(self, **kwargs):
        kwargs.pop('NAME', None)
        kwargs.update(shapley_mode='exact', c2_detach_backbone=False)
        super().__init__(**kwargs)
        del self.game
        del self.coalition_head
        self._game_visibility = None

    def _image_text_effects(self, *args, **kwargs):
        result = super()._image_text_effects(*args, **kwargs)
        self._game_visibility = result[-1]
        return result

    def _compact_coalitions(self, effects, batch_indices):
        if batch_indices.numel() == 0:
            return (None, None)
        features = effects[batch_indices]
        (b, _, c, n) = features.shape
        masks = features.new_tensor(MASKS)
        evidence = features[:, None] * masks.view(1, 8, 3, 1, 1)
        (point, text, image) = [evidence[:, :, k].reshape(b * 8, c, n) for k in range(3)]
        visibility = (self._game_visibility[batch_indices, None] * masks[:, 2].view(1, 8, 1, 1)).reshape(b * 8, 1, n)
        base_logits = self.head(point)
        logits = self.fusion(point, text, image, visibility, base_logits)[0]
        return (logits.reshape(b, 8, self.num_classes, n), torch.arange(n, device=effects.device))

def complete_attribution(out, labels, class_weights=None):
    compact = out.get('coalition_logits')
    indices = out.get('coalition_batch_indices')
    point_indices = out.get('coalition_point_indices')
    predicted = out.get('v_hat_mu')
    values = predicted.float().clone() if predicted is not None else None
    if compact is not None:
        (b, coalitions, classes, n) = compact.shape
        target = labels[indices][:, point_indices]
        target = target[:, None].expand(b, coalitions, n).reshape(b * coalitions, n)
        ce = F.cross_entropy(compact.reshape(b * coalitions, classes, n).float(), target, weight=class_weights, reduction='none')
        if class_weights is None:
            utility = -ce.mean(-1).reshape(b, coalitions)
        else:
            denom = class_weights[target].sum(-1).clamp_min(1e-08)
            utility = -(ce.sum(-1) / denom).reshape(b, coalitions)
        if values is None:
            values = utility
        else:
            lam = out['lam'][indices].float()
            values[indices] = (1 - lam) * utility + lam * values[indices]
    if values is None:
        raise RuntimeError('No coalition values available for attribution')
    phi = exact_shapley(values)
    return {'logits': out['logits'], 'coalition_utilities': values, 'shapley': phi, 'efficiency_gap': (phi.sum(-1) - (values[:, 7] - values[:, 0])).abs().max(), 'coalition_points': int(compact.shape[-1]) if compact is not None else 0, 'exact_fraction': out['exact_fraction']}
