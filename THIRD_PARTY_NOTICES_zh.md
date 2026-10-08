# 第三方声明

[English](THIRD_PARTY_NOTICES.md)

本文件为 [Third-Party Notices](THIRD_PARTY_NOTICES.md) 的中文版本。如中英文
表述存在差异，应以相关第三方发布的原始许可证及条款为准。

3DGENERATION 基于 EmbodiedGen 开发，并使用或调用第三方源代码、Git 子模块、
Python 包和模型权重。根目录的 [Apache License 2.0](LICENSE) 仅适用于
3DGENERATION、EmbodiedGen 或其他相关权利人依据该许可证发布的材料，不会替代
下列组件各自适用的许可证或其他条款。

本文件仅用于协助署名与合规审阅，不构成法律意见，也不能替代对适用许可证全文的
审阅。

## 源码基础

- [HorizonRobotics/EmbodiedGen](https://github.com/HorizonRobotics/EmbodiedGen)：
  本仓库的上游源码基础，采用 Apache‑2.0。本仓库保留适用的 Horizon Robotics
  与 EmbodiedGen 源码头和署名；3DGENERATION 仅对自身原创贡献和修改部分主张
  相应权利。

## 仓库内 Git 子模块

本仓库记录以下 Git 子模块的上游地址和固定版本，不在主仓库中复制相应上游源代码。
用户初始化子模块时，Git 将直接从相应第三方上游仓库获取源代码；用户取得的该等
源代码分别适用各上游项目自身的许可证。

- [Microsoft TRELLIS](https://github.com/microsoft/TRELLIS)，固定提交
  `442aa1e1afb9014e80681d3bf604e8d728a86ee7`，采用 MIT License；初始化后的
  子模块中，许可证文本见
  [`thirdparty/TRELLIS/LICENSE`](thirdparty/TRELLIS/LICENSE)。
- [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects)，通过
  [HochCC/sam-3d-objects](https://github.com/HochCC/sam-3d-objects) 引入，
  固定提交 `01417d16fb5cc762a60f370c1bf7f59d603ddfaf`。其源码和权重采用 Meta
  的 **SAM License**，不是 Apache‑2.0 或 MIT；初始化后的子模块中，全文见
  [`thirdparty/sam3d/LICENSE`](thirdparty/sam3d/LICENSE)。

SAM License 包含再分发、论文署名、贸易管制、禁止用途、终止及其他条件；模型
仓库还可能要求用户登录并接受相关条款。启用、使用或再分发该功能前，应完整审阅
并遵守相关许可证和条款。

## 运行时下载的模型

模型权重不随本仓库分发。安装脚本或程序运行时可能按用户操作下载以下资源：

| 模型或资源 | 上游 | 上游声明的许可证或条款 |
| --- | --- | --- |
| TRELLIS-image-large | [Microsoft](https://huggingface.co/microsoft/TRELLIS-image-large) | MIT |
| SAM 3D Objects 权重 | [Meta](https://huggingface.co/facebook/sam-3d-objects) | SAM License，可能需要门控授权 |
| MoGe ViT-L 权重 | [Ruicheng/moge-vitl](https://huggingface.co/Ruicheng/moge-vitl) | Apache‑2.0 |
| Segment Anything | [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) | Apache‑2.0 |
| DINOv2 | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | Apache‑2.0 |
| Improved Aesthetic Predictor | [上游仓库](https://github.com/christophschuhmann/improved-aesthetic-predictor) | Apache‑2.0 |
| OpenAI CLIP | [openai/CLIP](https://github.com/openai/CLIP) | MIT |
| rembg | [danielgatis/rembg](https://github.com/danielgatis/rembg) | MIT |

## 安装脚本获取的源代码依赖

下列项目不随本仓库源码一并分发。用户执行安装脚本时，安装工具将从对应的第三方
上游地址获取；以下列示仅用于说明安装来源及适用许可。

| 组件 | 许可证 |
| --- | --- |
| [EasternJournalist/utils3d](https://github.com/EasternJournalist/utils3d) | MIT |
| [OpenAI CLIP](https://github.com/openai/CLIP) | MIT |
| [Segment Anything](https://github.com/facebookresearch/segment-anything) | Apache‑2.0 |
| [NVIDIA nvdiffrast](https://github.com/NVlabs/nvdiffrast) | NVIDIA Source Code License (1-Way Commercial)，需单独审阅许可证全文 |
| [NVIDIA Kaolin](https://github.com/NVIDIAGameWorks/kaolin) | Apache‑2.0 |
| [gsplat](https://github.com/nerfstudio-project/gsplat) | Apache‑2.0 |
| [PyTorch3D](https://github.com/facebookresearch/pytorch3d) | BSD‑3‑Clause |
| [Microsoft MoGe](https://github.com/microsoft/MoGe) | MIT |

## 安装环境与非分发内容

本次开源交付范围仅包括项目源代码和随源码提供的文档，不包含已经安装完成的
Python 或 Conda 运行环境，也不包含容器镜像、虚拟机镜像、Python wheel 安装包、
第三方软件包的源代码或二进制副本。

用户按照项目说明自行安装环境时，由包管理工具从第三方软件源下载的 Python 包、
系统组件及其依赖，属于用户自行取得的本地运行环境，不属于本项目本次对外分发的
内容。因此，本声明不将该等环境依赖列为随项目分发的第三方材料。

如项目后续拟将已经安装完成的运行环境、容器镜像、二进制安装包或第三方软件包
副本作为项目内容一并对外提供，项目组将在发布前对该等新增交付物另行开展开源
合规审查，并根据实际包含的组件补充相应的许可证和声明。未经另行审查的上述内容
不属于本项目的公开发布范围。

## 外部项目与服务

`RoboCousin` 和同级 `sam3` 分割运行时均为独立项目，不随本仓库分发。API
提供商、托管模型服务及下载的模型或其他资源分别适用其各自的条款。

## 输入、输出、名称与商标

用户应自行确保其对输入图片、提示词、数据集、生成资产及下游用途享有必要权利。
模型许可证或服务条款可能对生成结果规定额外条件。文中引用的第三方项目、公司、
模型及商标仅用于识别和署名，不应被解释为任何第三方对 3DGENERATION 的认可、
赞助、批准或背书。
