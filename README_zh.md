# 3DGENERATION

[English](./README.md)

本仓库基于 [HorizonRobotics/EmbodiedGen](https://github.com/HorizonRobotics/EmbodiedGen)
扩展，提供图像到 3D 资产生成、交互式分割和下游流程编排 WebUI。项目可与
[RoboCousin](https://github.com/Tele-DLife/RoboCousin) 配合使用，由 RoboCousin
负责场景运行、仿真预览和数据采集。

在当前架构中：
- `3DGENERATION`：图像到 3D 生成、分割交互、WebUI 编排
- `RoboCousin`：下游场景运行、仿真预览、数据采集

---

## 1. 本仓库新增能力

相较上游 EmbodiedGen，本仓库主要增加以下功能：

- **SAM3 文本提示分割接入**
  - 新增 `apps/sam3_segment_cli.py`，通过独立 `sam3` conda 环境调用。
  - `apps/app_demo_Eng.py`（中文版本：`apps/app_demo_CHN.py`）增加文本提示分割模式（prompt + threshold）。
  - 保留点选分割，可在两种分割方式间选择。

- **WebUI 集成 RoboCousin**
  - Step3 增加资产分类（`Actor / Static / Room / General`）并同步至 RoboCousin。
  - Step4 可调用 RoboCousin 完成数据采集、Desk Cousin 预览和场景渲染。
  - 提供实时子进程日志，并支持停止整个进程组。

- **三条流程统一入口**
  - 在 `apps/app_demo_Eng.py`（中文版本：`apps/app_demo_CHN.py`）中整合：
    - `3D资产生成`
    - `桌面数字表亲`
    - `家庭场景搭建`

---

## 2. 跨仓目录约定

WebUI 默认从同一项目根目录下查找 3DGENERATION、RoboCousin 和 sam3：

```text
<projects_root>/
  3DGENERATION/
  RoboCousin/
  sam3/
```

说明：
- `RoboCousin`：下游仿真、渲染、布局运行和数据采集。
- `sam3`：用于文本提示分割的独立同级仓库，由 `apps/sam3_segment_cli.py` 调用。
- `thirdparty/TRELLIS` 与 `thirdparty/sam3d`：本仓库管理的 Git 子模块；其中 `thirdparty/sam3d` 提供图像到 3D 推理，与上述同级 `sam3` 分割运行时并非同一组件。

---

## 3. 环境准备

### 3.1 推荐一键安装

```bash
git clone --recurse-submodules https://github.com/Tele-DLife/3DGENERATION.git
cd 3DGENERATION

bash scripts/bootstrap_env.sh
```

### 3.2 手动安装

```bash
git submodule update --init --recursive --progress
conda env create -f environment.yml
conda activate 3dgeneration
bash install.sh basic
```

### 第三方组件的取得与许可

部分环境准备命令会初始化 Git 子模块，或从第三方仓库和服务直接获取依赖及模型
文件。相关组件由用户从对应第三方取得；安装或使用前，请审阅并遵守上游许可条款。
如组件要求注册账户、接受许可协议或申请访问权限，请由用户直接向相应第三方完成。

需要特别注意：

- **NVIDIA nvdiffrast：**安装脚本会从其上游仓库获取 nvdiffrast。该组件适用
  NVIDIA Source Code License (1-Way Commercial)，仅允许将其作品及衍生作品
  用于非商业研究或评估。再分发时须继续适用该许可证，附上完整许可证文本，并保留
  相关声明。
- **SAM 3D Objects：**代码和权重适用 Meta SAM License。权重属于门控资源，
  用户须直接向上游模型提供方申请访问。任何许可范围内的再分发仍须遵守 SAM
  License 并附上该协议；使用和分发还须遵守其中的贸易管制条款及适用法律。

完整条款请参阅[第三方声明](THIRD_PARTY_NOTICES_zh.md)及相应上游许可证文本。

### 3.3 预下载通用模型资产

激活环境并下载 WebUI 使用的公开、非门控模型资产：

```bash
conda activate 3dgeneration
python scripts/prefetch_models.py
```

该脚本会准备 TRELLIS、DINOv2、rembg U2Net、MoGe ViT-L、点选分割使用的
SAM ViT-H 权重，以及美学评估与 CLIP 资产。其中 SAM 与美学评估权重会直接
从各自的官方上游项目下载，并保存到 `models/checkpoints/`。如需更改位置，请在
预下载和启动 WebUI 前设置 `EMBODIEDGEN_MODEL_ASSETS_DIR`。该脚本**不会**下载
需要单独授权的 SAM 3D Objects 权重。如需只检查本地资产是否齐全而不执行下载，可运行：

```bash
python scripts/prefetch_models.py --check
```

### 3.4 下载 SAM 3D Objects 权重

SAM 3D Objects 权重属于门控模型，并受 Meta SAM License 约束。下载前请：

1. 在 [SAM 3D Objects 模型页面](https://huggingface.co/facebook/sam-3d-objects)
   申请访问并接受相关条款；
2. 使用获批账号登录 Hugging Face CLI；
3. 将权重下载到 WebUI 默认读取的位置：

```bash
hf auth login
hf download --repo-type model \
  --local-dir weights/sam-3d-objects \
  --max-workers 1 \
  facebook/sam-3d-objects
```

SAM 3D Objects 使用的 MoGe 权重已包含在上一节的通用预下载流程中。相关模型与
依赖协议请参阅[第三方声明](THIRD_PARTY_NOTICES_zh.md)。

### 3.5 离线与在线模式

WebUI 默认以仅本地方式加载模型：

```text
EMBODIEDGEN_OFFLINE=1
LOCAL_MODELS_ONLY=1
```

首次按默认模式启动前，请完成 3.3 和 3.4 节。若缺少 SAM 3D Objects 权重，
程序会报错，不会自动联网下载。如需允许其他模型加载器联网获取缺失资产，可启动
在线模式：

```bash
EMBODIEDGEN_OFFLINE=0 LOCAL_MODELS_ONLY=0 \
python apps/app_demo_Eng.py
```

在线模式不会绕过门控模型的访问授权或协议接受要求。
以上变量仅控制模型和资源的加载方式，不会禁用后续管线中已配置的 GPT/VLM
API 调用。

---

## 4. API Key 配置

使用 GPT 相关功能前，复制配置模板：

```bash
cp embodied_gen/utils/gpt_api_keys.yaml \
   embodied_gen/utils/gpt_api_keys.local.yaml
```

在 `embodied_gen/utils/gpt_api_keys.local.yaml` 中，将 `xxx` 替换为所用模型的
API Key，例如 `gpt-4o` 或 `qwen2.5-vl`。

---

## 5. 启动 WebUI

```bash
python apps/app_demo_Eng.py
```

默认端口：`8081`。

相关入口脚本：
- `apps/app_demo_Eng.py`（英文 UI）
- `apps/app_demo_CHN.py`（中文 UI）

---

## 6. URDF 属性估计模式

WebUI 会将生成的网格转换为 URDF，并估计类别、姿态、高度、质量、摩擦系数和
颜色等语义与物理属性。当前提供两个可互换的后端：

- **默认模式**（`embodied_gen/validators/urdf_convertor.py`）：由配置的视觉语言
  模型直接估计全部属性。
- **增强的分档摩擦模式**
  （`embodied_gen/validators/urdf_converter_CoT.py`，通过 `cot` 选择）：使用
  `URDFGeneratorRankedFriction`，并增加两项检查：
  1. 模型先根据物体表面材质匹配八个由低到高排列的参考摩擦档位，再在对应范围内
     选择相对于橡胶表面的静摩擦和动摩擦系数；静摩擦系数不得低于动摩擦系数。
  2. 完成第一次高度估计后，将物体与边长严格为 0.10 m 的参考方块并排渲染。
     第二次视觉语言模型检查只会在画面明显否定原估计时调整高度；结果不确定或
     大致一致时保留原值。

中英文 UI 都通过 `apps/common.py` 使用同一后端；该模式需要在启动时选择，目前
没有页面内切换按钮。如果增强转换器导入失败，日志会提示警告并回退到默认模式。

通过以下环境变量选择模式：

- `EMBODIEDGEN_URDF_CONVERTER=default`（或不设置）：默认估计
- `EMBODIEDGEN_URDF_CONVERTER=cot`：启用分档摩擦和高度复核

### 6.1 调用示例

```bash
# 英文 UI，默认模式
python apps/app_demo_Eng.py

# 中文 UI，默认模式
python apps/app_demo_CHN.py

# 英文 UI，增强模式
EMBODIEDGEN_URDF_CONVERTER=cot python apps/app_demo_Eng.py

# 中文 UI，增强模式
EMBODIEDGEN_URDF_CONVERTER=cot python apps/app_demo_CHN.py
```

---

## 7. SAM3 运行要求

UI 会通过如下方式调用 SAM3：

```bash
conda run -n sam3 python apps/sam3_segment_cli.py ...
```

请确保：
- 已有可用的 `sam3` conda 环境；
- SAM3 仓库位于 `../sam3`（或在代码中修改路径）；
- 模型缓存路径可访问。

同级 `sam3` 运行时与 `thirdparty/sam3d` 图像到 3D 子模块是两个不同组件。
可选的文本提示分割功能请按照 [SAM3 官方说明](https://github.com/facebookresearch/sam3)
完成环境安装与权重授权。

---

## 8. 与 RoboCousin 配合使用

本节介绍 3DGENERATION 调用
[RoboCousin](https://github.com/Tele-DLife/RoboCousin) 完成仿真预览和数据采集的方式。

### 8.1 使用前配置

请先完成以下配置：

- 按同级目录组织工程：

```text
<projects_root>/
  3DGENERATION/
  RoboCousin/
  sam3/   # 可选，仅文本提示分割需要
```

- 为下游子进程配置可用 Python 入口：
  - 推荐设置 `ROBOCOUSIN_PYTHON=<robocousin_env_python_path>`，或
  - 保证存在名为 `robocousin`、可被自动发现的 conda 环境。
- 在 RoboCousin 仓库中运行 `bash script/_install.sh`，并安装 `ffmpeg`、
  `xdotool` 等系统依赖。

### 8.2 模式 A：3D 资产生成（asset_extraction）

1. 在 `3D资产生成` 中上传图片并完成分割（点选或 SAM3 文本提示）。
2. **可选：**在参数配置区设置重建参数、分割阈值和 URDF 转换模式，再点击
   “启动重构/同步”。
3. 在 Step3 按物体功能选择存储路径（`Actor / Static / Room / General`），将资产同步到 `RoboCousin` 对应目录。
4. 在 Step4 选择用于数据采集或预览的房间场景，启动下游任务。
5. 如需中断长任务，使用 UI 停止按钮终止子进程。

### 8.3 模式 B：桌面数字表亲（desktop_twin）

1. 输入桌面图片并选择目标房间场景。
2. WebUI 调用 RoboCousin 的 Desk Cousin 流程，生成并匹配桌面相对布局。
3. 在仿真中预览根据输入图片重建的桌面物体布局，或继续执行房间内的数据采集。

### 8.4 模式 C：场景搭建（scene_builder）

1. 选择要搭建的场景模板/配置。
2. WebUI 调用 `RoboCousin` 下游布局与渲染流程。
3. 在仿真中预览生成的场景，并按需调整或导出结果。

---

## 9. 仓库职责与接口

两个仓库的职责如下：

- `3DGENERATION` 负责：
  - WebUI 和流程编排
  - 分割与资产生成交互
  - 跨仓调用和参数传递

- `RoboCousin` 负责：
  - Desk Cousin 提取和匹配
  - 房间布局执行与渲染
  - 仿真预览与数据采集

两个项目通过 JSON 文件和配置字段交互，尤其是
`relative_layout_ontop_desktable.json`。修改这些接口时，应同步更新两个仓库。

---

## 10. 上游与引用

本仓库基于以下项目开发：
- EmbodiedGen：[仓库](https://github.com/HorizonRobotics/EmbodiedGen)
- RoboCousin：[仓库](https://github.com/Tele-DLife/RoboCousin)
- RoboTwin 下游运行时：[仓库](https://github.com/RoboTwin-Platform/RoboTwin)

如使用原始 EmbodiedGen 工作，请引用其论文（BibTeX 见 `README.md`）：

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

## 11. 开源协议

除非文件或组件另有说明，3DGENERATION 的原创贡献及对 EmbodiedGen 的修改以
[Apache License 2.0](LICENSE) 发布；EmbodiedGen 原有版权与署名声明继续保留。

第三方子模块、模型权重和依赖仍受各自协议约束，本项目不会将其重新许可为
Apache-2.0。尤其需要注意，SAM 3D Objects 适用 Meta 的 SAM License，而不是
Apache-2.0；NVIDIA nvdiffrast 另有非商业用途及再分发条件。使用或再分发前请
阅读 [NOTICE](NOTICE) 与[第三方声明](THIRD_PARTY_NOTICES_zh.md)（[英文版](THIRD_PARTY_NOTICES.md)）。
