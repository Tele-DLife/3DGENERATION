# 3DGENERATION

[简体中文](./README_zh.md)

This repository extends
[HorizonRobotics/EmbodiedGen](https://github.com/HorizonRobotics/EmbodiedGen)
with a WebUI for image-to-3D asset generation, interactive segmentation, and
downstream workflow orchestration. It works with
[RoboCousin](https://github.com/Tele-DLife/RoboCousin) for scene execution,
simulation preview, and data collection.

In the current architecture:
- `3DGENERATION`: image-to-3D generation, segmentation interaction, WebUI orchestration
- `RoboCousin`: scene/runtime execution, simulation preview, and data collection

---

## 1. What This Repository Adds

Compared with upstream EmbodiedGen, this repository adds:

- **SAM3 text-prompt segmentation integration**
  - Added `apps/sam3_segment_cli.py`, executed in an isolated `sam3` conda env.
  - Added text-prompt segmentation mode in `apps/app_demo_Eng.py` (prompt + threshold).
  - Preserved point-click segmentation for dual interaction styles.

- **RoboCousin integration in the WebUI**
  - Added Step3 asset categorization (`Actor / Static / Room / General`) with RoboCousin sync.
  - Step4 calls RoboCousin for data collection, Desk Cousin preview, and scene rendering.
  - Added live subprocess logs and controls for stopping the full process group.

- **Unified user-facing workflows**
  - UI (`apps/app_demo_Eng.py`, Chinese version: `apps/app_demo_CHN.py`) with three lanes:
    - `3D Asset Generation`
    - `Desktop Digital Cousin`
    - `Home Scene Builder`

---

## 2. Workspace Layout (Cross-Repo)

`app_demo_Eng.py` assumes sibling repositories under the same root:

```text
<projects_root>/
  3DGENERATION/
  RoboCousin/
  sam3/
```

Notes:
- `RoboCousin` is required for downstream simulation/collection/scene workflows.
- `sam3` is a separate sibling repository used for text-prompt segmentation.
- `thirdparty/TRELLIS` and `thirdparty/sam3d` are managed as git submodules in this repository. The latter provides image-to-3D inference and is different from the sibling `sam3` segmentation runtime.

---

## 3. Environment Setup

### 3.1 Recommended one-shot bootstrap

```bash
git clone --recurse-submodules https://github.com/Tele-DLife/3DGENERATION.git
cd 3DGENERATION

bash scripts/bootstrap_env.sh
```

### 3.2 Equivalent manual setup

```bash
git submodule update --init --recursive --progress
conda env create -f environment.yml
conda activate 3dgeneration
bash install.sh basic
```

### Third-party access and licensing

Some setup commands initialize submodules or retrieve dependencies and model
artifacts directly from third-party repositories and services. Users obtain
these components from the respective third parties and should review and comply
with the applicable upstream terms before installation or use. If a component
requires account registration, license acceptance, or access approval, complete
that process directly with the relevant third party.

Key restrictions include:

- **NVIDIA nvdiffrast:** the setup script retrieves nvdiffrast from its upstream
  repository. The NVIDIA Source Code License (1-Way Commercial) limits the work
  and its derivatives to noncommercial research or evaluation. Redistribution
  must remain under that license, include a complete copy of the license, and
  retain the applicable notices.
- **SAM 3D Objects:** the code and checkpoints are governed by the Meta SAM
  License. Checkpoint access is gated and must be requested by the user directly
  from the upstream model provider. Any permitted redistribution remains subject
  to the SAM License and must include that agreement. Use and distribution must
  also comply with its trade-control provisions and applicable laws.

See [Third-Party Notices](THIRD_PARTY_NOTICES.md) and the applicable upstream
license texts for complete terms.

### 3.3 Prefetch common model assets

Activate the environment and download the public, non-gated assets used by the
WebUI:

```bash
conda activate 3dgeneration
python scripts/prefetch_models.py
```

This prepares TRELLIS, DINOv2, rembg U2Net, MoGe ViT-L, the SAM ViT-H
checkpoint used for point-click segmentation, and the aesthetic/CLIP assets.
The SAM and aesthetic checkpoints are downloaded directly from their official
upstream projects and stored under `models/checkpoints/`.
Set `EMBODIEDGEN_MODEL_ASSETS_DIR` before prefetching and launching the WebUI
to use a different location.
It does **not** download the gated SAM 3D Objects checkpoints. To verify local
artifacts without downloading anything, run:

```bash
python scripts/prefetch_models.py --check
```

### 3.4 Download SAM 3D Objects checkpoints

SAM 3D Objects checkpoints are gated and governed by the Meta SAM License.
Before downloading them:

1. Request access and accept the terms on the
   [SAM 3D Objects model page](https://huggingface.co/facebook/sam-3d-objects).
2. Authenticate the Hugging Face CLI with the account that was granted access.
3. Download the checkpoints to the location expected by the WebUI:

```bash
hf auth login
hf download --repo-type model \
  --local-dir weights/sam-3d-objects \
  --max-workers 1 \
  facebook/sam-3d-objects
```

MoGe is downloaded by the common prefetch step above. See
[Third-Party Notices](THIRD_PARTY_NOTICES.md) for the applicable model and
dependency terms.

### 3.5 Offline and online modes

The WebUI defaults to local-only model loading:

```text
EMBODIEDGEN_OFFLINE=1
LOCAL_MODELS_ONLY=1
```

Complete Sections 3.3 and 3.4 before the first default-mode launch. If the
SAM 3D Objects checkpoints are missing, the WebUI reports an error and does not
download them automatically. To let other model loaders download missing
assets, start the UI in online mode:

```bash
EMBODIEDGEN_OFFLINE=0 LOCAL_MODELS_ONLY=0 \
python apps/app_demo_Eng.py
```

Online mode does not bypass gated-model access or license acceptance.
These flags only control model and asset loading. They do not disable the
configured GPT/VLM API calls used by later pipeline stages.

---

## 4. API Key Setup

Before using GPT-related features, copy the configuration template:

```bash
cp embodied_gen/utils/gpt_api_keys.yaml \
   embodied_gen/utils/gpt_api_keys.local.yaml
```

In `embodied_gen/utils/gpt_api_keys.local.yaml`, replace `xxx` with the API key
for the model you use, such as `gpt-4o` or `qwen2.5-vl`.

---

## 5. Run WebUI

```bash
python apps/app_demo_Eng.py
```

Default port: `8081`.

Related entry scripts:
- `apps/app_demo_Eng.py` (English UI)
- `apps/app_demo_CHN.py` (Chinese UI)

---

## 6. URDF Attribute Estimation Modes

The WebUI converts each generated mesh into a URDF and estimates semantic and
physical attributes such as category, pose, height, mass, friction, and color.
Two interchangeable backends are available:

- **Default** (`embodied_gen/validators/urdf_convertor.py`): asks the configured
  vision-language model to estimate all attributes directly.
- **Enhanced ranked-friction mode**
  (`embodied_gen/validators/urdf_converter_CoT.py`, selected with `cot`): uses
  `URDFGeneratorRankedFriction` and adds two checks:
  1. The model first matches the surface to one of eight ordered material and
     friction reference bands, then selects concrete static and dynamic values
     relative to rubber. Static friction must be no lower than dynamic friction.
  2. After the first height estimate, the object is rendered beside an exact
     0.10 m reference cube. A second vision-language-model pass changes the
     height only when the visual comparison clearly contradicts the estimate;
     uncertain or approximately consistent results are retained.

Both language UIs use the same backend through `apps/common.py`; the mode is
selected at launch rather than through an in-page toggle. If the enhanced
converter cannot be imported, the application logs a warning and falls back to
the default converter.

Select the mode with:

- `EMBODIEDGEN_URDF_CONVERTER=default` (or unset): default estimation
- `EMBODIEDGEN_URDF_CONVERTER=cot`: ranked friction and height verification

### 6.1 Usage examples

```bash
# English UI, default mode
python apps/app_demo_Eng.py

# Chinese UI, default mode
python apps/app_demo_CHN.py

# English UI, enhanced mode
EMBODIEDGEN_URDF_CONVERTER=cot python apps/app_demo_Eng.py

# Chinese UI, enhanced mode
EMBODIEDGEN_URDF_CONVERTER=cot python apps/app_demo_CHN.py
```

---

## 7. SAM3 Runtime Requirements

The SAM3 path is invoked by:

```bash
conda run -n sam3 python apps/sam3_segment_cli.py ...
```

Please ensure:
- a valid `sam3` conda environment exists,
- the SAM3 repository is available at `../sam3` (or update path in code),
- model cache paths are accessible.

The sibling `sam3` runtime is separate from the `thirdparty/sam3d` image-to-3D
submodule. Follow the [official SAM3 setup and checkpoint-access
instructions](https://github.com/facebookresearch/sam3) for that optional
text-prompt segmentation runtime.

---

## 8. Using RoboCousin Workflows

This section describes how 3DGENERATION calls
[RoboCousin](https://github.com/Tele-DLife/RoboCousin) for simulation preview and
data collection.

### 8.1 Setup

Before running these workflows:

- Keep repositories as siblings:

```text
<projects_root>/
  3DGENERATION/
  RoboCousin/
  sam3/   # optional, only required for text-prompt segmentation
```

- Provide a valid Python runtime for downstream subprocesses:
  - recommended: `ROBOCOUSIN_PYTHON=<path_to_robocousin_env_python>`, or
  - make sure a discoverable conda environment named `robocousin` is available.
- In the RoboCousin repository, run `bash script/_install.sh` and install the
  required system packages, including `ffmpeg` and `xdotool`.

### 8.2 Mode A: 3D Asset Generation (`asset_extraction`)

1. In `3D Asset Generation`, upload an image and perform segmentation (point-click or SAM3 text prompt).
2. **Optional:** configure generation parameters in the parameter panel (reconstruction options, segmentation thresholds, URDF converter mode, etc.), then start the pipeline.
3. In Step3, select the storage path based on object functionality (`Actor / Static / Room / General`) to sync assets into `RoboCousin`.
4. In Step4, select the room scene for simulation preview or data collection, then start the task.
5. Stop long-running jobs with the UI stop button when needed.

### 8.3 Mode B: Desktop Digital Cousin (`desktop_twin`)

1. Provide a desktop image and choose the target room scene.
2. The WebUI calls RoboCousin's Desk Cousin workflow to generate and match the relative desk layout.
3. In simulation, preview the desk arrangement reconstructed from the input image, or continue directly to room-level data collection.

### 8.4 Mode C: Scene Builder (`scene_builder`)

1. Select the scene template/configuration to build.
2. The WebUI calls RoboCousin to build and render the layout.
3. Preview the generated scene in simulation and iterate/export as needed.

---

## 9. Repository Responsibilities and Interfaces

The repositories have the following responsibilities:

- `3DGENERATION` owns:
  - WebUI and workflow orchestration
  - segmentation and asset-generation interactions
  - cross-repository calls and parameter passing

- `RoboCousin` owns:
  - Desk Cousin extraction and matching
  - room layout execution/rendering
  - simulation preview and data collection

The projects exchange data through JSON files and configuration fields,
especially `relative_layout_ontop_desktable.json`. Update both repositories when
changing these interfaces.

---

## 10. Upstream References and Citation

This repository builds on:
- EmbodiedGen: [repository](https://github.com/HorizonRobotics/EmbodiedGen)
- RoboCousin: [repository](https://github.com/Tele-DLife/RoboCousin)
- RoboTwin downstream runtime: [repository](https://github.com/RoboTwin-Platform/RoboTwin)

If you use the original EmbodiedGen work, please cite:

```bibtex
@misc{wang2025embodiedgengenerative3dworld,
      title={EmbodiedGen: Towards a Generative 3D World Engine for Embodied Intelligence},
      author={Xinjie Wang and Liu Liu and Yu Cao and Ruiqi Wu and Wenkang Qin and Dehui Wang and Wei Sui and Zhizhong Su},
      year={2025},
      eprint={2506.10600},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.10600},
}
```

---

## 11. License

Unless otherwise stated, original 3DGENERATION contributions and modifications
to EmbodiedGen are available under the [Apache License 2.0](LICENSE).
Applicable EmbodiedGen copyright and attribution notices are retained.

Third-party submodules, model weights, and dependencies remain under their own
terms and are not relicensed by this project. In particular, SAM 3D Objects is
governed by Meta's SAM License rather than Apache-2.0, and NVIDIA nvdiffrast is
subject to its own noncommercial-use and redistribution conditions. Review
[NOTICE](NOTICE) and [Third-Party Notices](THIRD_PARTY_NOTICES.md) before use or
redistribution.
