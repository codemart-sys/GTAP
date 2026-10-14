# Slimmed GTAP — network code


| File | What it is |
| --- | --- |
| `model.py` | The model itself. Defines `PointNextMultimodalRLShapley`, the multimodal segmentation model that performs view selection, coalition utilities and Shapley attribution, together with its building blocks: `replace_bn_with_gn`, `ActorCritic` (view-selection policy and value head), `TinyDepthEncoder` (depth-view encoder), `FusionLogitAdapter` (logit-level fusion adapter) and `ExactCoalitionGame` (exact coalition utility game). |
| `amortizer.py` | The amortized utility network. Defines `ShapleyAmortizer`, which predicts coalition utilities in a single forward pass instead of evaluating them exactly. |
| `controller.py` | The routing controller. Defines `MixController`, which decides, per sample, how much of the attribution is computed exactly and how much is delegated to the amortizer. |
| `shapley.py` | The Shapley primitives. Defines the player-coalition masks `MASKS` and `MASK_ID` and the function `exact_shapley`, which evaluates the complete three-player Shapley formula from coalition utilities. |
| `full_exact_shapley.py` | The full-exact variant. Defines `FullExactShapley`, a model that always evaluates all eight coalitions over all query points, and `complete_attribution`, the helper that turns a forward pass into attribution values. |
| `attribution.py` | A four-line re-export module. It only forwards `MASKS`, `MASK_ID`, `exact_shapley`, `ShapleyAmortizer` and `MixController` to the model code. |
| `backbone_stage_factory.py` | The lightweight backbone of the final model. Defines and registers `FourStageFixedInterfaceEncoder`, the four-stage encoder (channels `[24, 32, 64, 384]`) that replaces the original five-stage PointNeXt backbone. |

## Notes

* Dependencies: PyTorch and the upstream PointNeXt `openpoints` package (`MODELS`, `build_model_from_cfg`,
  `PointNextEncoder`, `PointNextDecoder`, `SegHead`). The `openpoints` sources themselves are not included here.
* `model.py` imports its sibling modules with package-relative imports, so these files are meant to live together
  in one package.
* Every resource number quoted for this model was produced without modifying any of these files.
