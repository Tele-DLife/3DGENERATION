# Third-Party Notices

[简体中文](THIRD_PARTY_NOTICES_zh.md)

3DGENERATION is based on EmbodiedGen and uses or interoperates with third-party
source code, Git submodules, Python packages, and model artifacts. The project-level
[Apache License 2.0](LICENSE) applies only to material distributed under that
license by 3DGENERATION, EmbodiedGen, or other relevant rightsholders. It does
not replace the licenses or terms applicable to the components listed below.

This document is an attribution and compliance aid, not legal advice or a
substitute for reading the applicable licenses.

## Source base

| Component | Use in this repository | License |
| --- | --- | --- |
| [HorizonRobotics/EmbodiedGen](https://github.com/HorizonRobotics/EmbodiedGen) | Upstream source base; this repository modifies its structure, WebUI, segmentation, image-to-3D, URDF, and cross-repository orchestration | Apache-2.0; applicable upstream notices and source headers are retained |

Files carrying an EmbodiedGen or Horizon Robotics copyright notice remain
copyrighted by their respective owners. 3DGENERATION claims copyright only in
its original contributions and modifications.

## Git submodules referenced by this repository

The main repository records the upstream locations and pinned revisions of the
following Git submodules; it does not copy their upstream source code into the
main repository. When a user initializes the submodules, Git retrieves the
source code directly from the respective third-party repositories. That source
code remains subject to the applicable upstream licenses.

| Component | Pinned revision | License and required handling |
| --- | --- | --- |
| [Microsoft TRELLIS](https://github.com/microsoft/TRELLIS) | `442aa1e1afb9014e80681d3bf604e8d728a86ee7` | MIT; retain its copyright and license text. After initialization, the submodule's license is available at [`thirdparty/TRELLIS/LICENSE`](thirdparty/TRELLIS/LICENSE). |
| [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects), obtained through [HochCC/sam-3d-objects](https://github.com/HochCC/sam-3d-objects) | `01417d16fb5cc762a60f370c1bf7f59d603ddfaf` | Meta **SAM License**, not Apache-2.0 or MIT. Code and checkpoints remain subject to that agreement. After initialization, the license is available at [`thirdparty/sam3d/LICENSE`](thirdparty/sam3d/LICENSE). |

The SAM License contains redistribution, publication acknowledgement, trade
control, prohibited-use, termination, and other conditions. Anyone enabling,
using, or redistributing the SAM 3D Objects path must review and comply with
the complete license. The corresponding model repository is gated and may
require the user to accept its terms before downloading checkpoints.

## Model artifacts downloaded at runtime

Model weights are not stored in this repository. Installation or runtime
scripts may download them on the user's behalf.

| Model or artifact | Upstream | Declared terms |
| --- | --- | --- |
| TRELLIS-image-large | [Microsoft on Hugging Face](https://huggingface.co/microsoft/TRELLIS-image-large) | MIT |
| SAM 3D Objects checkpoints | [Meta on Hugging Face](https://huggingface.co/facebook/sam-3d-objects) | SAM License; gated access may apply |
| MoGe ViT-L weights | [Ruicheng/moge-vitl](https://huggingface.co/Ruicheng/moge-vitl) | Apache-2.0 |
| Segment Anything code/checkpoint | [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) | Apache-2.0 |
| DINOv2 code/checkpoint | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | Apache-2.0 |
| Improved Aesthetic Predictor weights | [christophschuhmann/improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) | Apache-2.0 |
| CLIP model/code | [openai/CLIP](https://github.com/openai/CLIP) | MIT |
| rembg U²-Net integration | [danielgatis/rembg](https://github.com/danielgatis/rembg) | MIT |

## Direct source dependencies installed by the setup script

These repositories are not distributed with the source code. The setup script
retrieves them separately from the identified third-party upstream locations.
They are listed here only to identify their installation source and applicable
license, and they are not relicensed by 3DGENERATION:

| Component | License |
| --- | --- |
| [EasternJournalist/utils3d](https://github.com/EasternJournalist/utils3d) | MIT |
| [OpenAI CLIP](https://github.com/openai/CLIP) | MIT |
| [Segment Anything](https://github.com/facebookresearch/segment-anything) | Apache-2.0 |
| [NVIDIA nvdiffrast](https://github.com/NVlabs/nvdiffrast) | NVIDIA Source Code License (1-Way Commercial); review the complete upstream license |
| [NVIDIA Kaolin](https://github.com/NVIDIAGameWorks/kaolin) | Apache-2.0 |
| [gsplat](https://github.com/nerfstudio-project/gsplat) | Apache-2.0 |
| [PyTorch3D](https://github.com/facebookresearch/pytorch3d) | BSD-3-Clause |
| [Microsoft MoGe](https://github.com/microsoft/MoGe) | MIT |

## Installation environments and materials not distributed

This open-source release consists only of the project source code and the
documentation provided with it. It does not include a preinstalled Python or
Conda environment, a container or virtual-machine image, Python wheel packages,
or copies of third-party package source code or binaries.

Python packages, system components, and their dependencies downloaded from
third-party sources when a user follows the installation instructions form part
of the user's locally obtained runtime environment. They are not distributed
as part of this release and are therefore not listed in this notice as
third-party materials distributed with the project.

If a future release includes a preinstalled environment, container image,
binary package, or copy of a third-party package, the project will review that
additional deliverable before release and provide the applicable license and
notice information for the materials actually included. Such materials are not
within the scope of this release unless separately reviewed.

## External projects and services

`RoboCousin` and the sibling `sam3` segmentation runtime are separate
projects and are not distributed as part of this repository. API providers,
hosted model services, and downloaded assets are governed by their respective
terms.

## Inputs, outputs, names, and marks

Users are responsible for ensuring that they have the rights required for
their input images, prompts, datasets, generated assets, and downstream uses.
Model licenses or service terms may impose conditions relevant to generated
outputs. References to third-party projects, companies, models, and trademarks
are for identification and attribution only and must not be construed as
endorsement, sponsorship, or approval of 3DGENERATION by any third party.
