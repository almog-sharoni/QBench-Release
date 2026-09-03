# Streamlit Dashboard File Map

This is the default dashboard launched by `./dashboard.sh` (or explicitly with
`./dashboard.sh streamlit`). The separate Svelte dashboard remains available
with `./dashboard.sh new`.

`runspace/src/database/dashboard.py` is now a small loader that executes these copied sections in order. The sections share one global namespace, matching the old single-file Streamlit behavior.

| File | Original lines | Purpose |
| --- | ---: | --- |
| `setup.py` | 1-139 | Imports, project path setup, constants, database file helpers, preset helpers. |
| `styles.py` | 140-363 | Global Streamlit/CSS styling. |
| `data_helpers.py` | 364-685 | Datatype parsing, run loading/mutation helpers, dataframe preprocessing, quantization win-rate calculations. |
| `graph_helpers.py` | 686-879 | Architecture graph generation/cache helpers, win-rate renderers, dashboard intro helper. |
| `experiments_tab.py` | 880-3002 | Main Streamlit setup, sidebar controls, feature-matching section, experiments tab and dialogs. |
| `cache_helpers.py` | 3003-3385 | Cache simulation formatting and bank visualization helper functions. |
| `cache_tab.py` | 3386-3583 | Cache Simulation tab UI. |
| `run_models_tab.py` | new | Dashboard launcher for non-interactive `run_interactive.py` model runs. |
| `model_workbench_tab.py` | new | Interactive model support analysis, many-to-many conversion mapping, guided replacement planning, inference, and export. |
| `graph_tab.py` | 3584-3650 | Architecture Graph tab UI and sidebar footer. |

## Model Quantization Workbench

Open **🧪 Model Workbench** after starting the dashboard. The tab enumerates
torchvision and timm catalogs or loads a trusted local
`package.module:factory`, resolves the provider's preferred input size, and
displays reference and proposed quantized operation graphs side by side.
Quantization-aware FX is validated on the actual sample. If a dynamic model
selects an invalid symbolic branch, the workbench falls back to an
input-specialized `torch.export` graph while conversion remains on the safe
eager model. The default layer-type overview keeps large graphs such as
MobileViT readable; per-module and full-operation views remain selectable.
Selecting a node highlights its one-to-one, one-to-many, or many-to-one
conversion mapping. Keep provider sizing enabled for fixed-shape models
(`vit_b_16` uses 224 and `mobilevit_s` uses 256); an invalid sample now fails
with an input-size error instead of a misleading hierarchy-only support view.

Conversion choices are exportable as a JSON recipe. A converted state bundle
is enabled only after reference-versus-converted sample validation succeeds;
it contains that recipe plus a CPU `state_dict`. The validation report includes
realized QBench modules and runtime hook hits. Custom architectures still need
their original Python package when reconstructed. Activation-boundary
quantization is intentionally excluded from direct inference in this tab
because it must execute through QBench's activation transport runtime.

An uncertain custom layer is never silently treated as an equivalent built-in
layer. Choose **replace** to select a target from the registered QBench layer
catalog, enter its constructor arguments, and explicitly map every target
parameter or buffer to source state (or choose a deliberate initializer). The
dashboard previews shape and dtype compatibility, requires confirmation of the
exact recipe, and can apply a compatible recipe across repeated instances of
the same source type. Conversion revalidates every concrete path before it
touches the cloned model; sample and dataset validation remain the final
semantic checks.

The validation section can also compare classification accuracy on a
deterministic subset of a local ImageNet/ImageFolder dataset. Torchvision models
use their DEFAULT weights' evaluation transform, timm models use their resolved
data configuration, and ImageNet WordNet folders are remapped to canonical
labels. A trusted `package.module:factory` may return another labeled Dataset or
DataLoader. The report contains reference/converted top-1 and top-k accuracy,
percentage-point deltas, prediction agreement, timing, throughput, and complete
dataset provenance; it is downloadable as JSON and included in a prepared state
bundle when current for the selected subset settings.
