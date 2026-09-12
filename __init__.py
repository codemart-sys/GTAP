"""GTAP — point / image / text segmentation with exact Shapley attribution and RL view selection.

Released model
--------------
`PointNextMultimodalRLShapley` with the configuration in `configs/released_v14encbn.yaml`
(`main_multitask: true`, `norm_mode: bn_encoder`). With that configuration the segmentation output is

    logits = SegHead( PointNextDecoder( PointNextEncoder(point cloud) ) )

so the text embedding and the depth view act on the auxiliary training losses only — inference needs
the point cloud alone (`model.inference_modalities == 'point'`, use `forward_point_only`).

The three contributions
-----------------------
1. **Three-modal fusion** (`model.py`): point features are fused with one Agent-selected depth view
   (`DepthEncoder` + `grid_sample` sampling at the projected point locations) and with a frozen
   `BAAI/bge-m3` text embedding through FiLM; the modalities are also aligned by a multi-positive
   contrastive objective (`losses.py`).
2. **Exact Shapley attribution** (`model.py` + `shapley.py`): the players `point / text / image` form
   a characteristic game whose eight coalitions all get a value from a shared fusion block and a
   scalar utility head; those values are calibrated to each coalition's real segmentation loss, and
   contributions are the exact Shapley values
   `phi_i = sum_S |S|!(n-|S|-1)!/n! * (v(S + {i}) - v(S))` — no softmax is applied.
3. **RL view selection** (`model.py` `ActorCritic`): all Fibonacci candidate views are scored from the
   point context plus cached geometry descriptors, one is chosen (sampled during RL, argmax for
   evaluation), and only that view's depth map is encoded.

Files
-----
- `model.py`   — the model: `PointNextMultimodalRLShapley`, `ActorCritic`, `DepthEncoder`,
                 `ExactCoalitionGame`, `replace_bn_with_gn`, and `forward_point_only`.
- `shapley.py` — `MASKS` / `MASK_ID` for the eight coalitions and `exact_shapley`.
- `losses.py`  — `MultimodalLoss`: segmentation CE + contrastive + coalition-utility calibration + A2C.
- `configs/released_v14encbn.yaml` — the released structure/hyper-parameters.

Dependency
----------
The backbone is **not** re-implemented here: `model.py` builds `PointNextEncoder`,
`PointNextDecoder` and `SegHead` through OpenPoints/PointNeXt
(`openpoints.models.build.MODELS` / `build_model_from_cfg`), so an importable `openpoints` package is
required to instantiate the model. Everything else (Shapley, fusion, game, Actor-Critic, losses) is
contained in these files.
"""
from .shapley import exact_shapley, MASKS, MASK_ID
from .model import PointNextMultimodalRLShapley
from .losses import MultimodalLoss

__all__ = ["PointNextMultimodalRLShapley", "MultimodalLoss", "exact_shapley", "MASKS", "MASK_ID"]
