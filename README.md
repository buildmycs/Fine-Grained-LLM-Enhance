 # Fine-Grained Multimodal Sentiment Analysis via LLM-Enhanced Text and Monotonic Ordinal Learning

This project focuses on fine-grained multimodal sentiment analysis by incorporating LLM-enhanced text, dual-text fusion, and monotonic ordinal learning into the ALMT framework, with particular emphasis on sentiment intensity classification metrics such as Acc-5 and Acc-7.

The method follows a progressive pipeline from representation learning to decision-making: context-assisted text enhancement first enriches semantic information; the original and enhanced texts are then fused to guide multimodal representation learning; finally, continuous regression and ordinal prediction are combined to model sentiment intensity and its ordinal relationships.

The project supports the CMU-MOSI and CMU-MOSEI datasets.

## 1. Method Overview

### LLM-Based Text Enhancement

A locally deployed Qwen2.5 model performs semantic explicitation using the target utterance and contextual utterances from the same video. This process includes resolving references, disambiguating expressions, and making implicit meanings explicit.

Any added information must be supported by evidence from the source text, and constraints are imposed to preserve the original sentiment polarity and intensity. The current C4-Explicit implementation uses up to four preceding utterances and two following utterances.

The LLM is used exclusively for offline data preprocessing and does not participate in the subsequent joint training of the multimodal model.

### Dual-Text and Multimodal Representation Learning

The original and enhanced texts share a BERT encoder and are fused through gated cross-attention anchored in the original text. ALMT's language-guided mechanism then integrates audio and visual information.

### Monotonic Ordinal Prediction and Regression Fusion

The model contains a continuous regression head and a monotonic ordinal head. The ordinal head uses six cumulative binary decisions to model seven sentiment levels ranging from −3 to +3. An ordered-threshold parameterization ensures the monotonicity of the cumulative probabilities.

The final prediction is:

$$
\hat{y}=(1-\rho)\hat{y}_{reg}+\rho\hat{y}_{ord}
$$

where:

- `ordinal_prediction_weight` corresponds to the prediction fusion coefficient $\rho$.
- `ordinal_weight` controls the weight of the ordinal loss during training.

These parameters serve different purposes: even when `ordinal_prediction_weight=0`, ordinal supervision can still influence shared representation learning as long as `ordinal_weight>0`.

## 2. Project Structure

```text
ALMT/
├── configs/                         # Dataset, model, and training configurations
│   ├── mosi_dual_c4_intensity.yaml   # Dual-text fusion + ordinal learning
│   └── mosei_dual_c4_intensity.yaml
├── core/
│   ├── dataset.py                   # Original data loader
│   ├── dataset_dual.py              # Dual-text data loader
│   ├── intensity_objective.py       # Regression, ordinal, and optional contrastive losses
│   ├── fusion_selection.py          # Fusion coefficient selection and resolution
│   ├── model_selection.py           # Validation-based checkpoint selection
│   ├── metric.py                    # Evaluation metrics
│   ├── scheduler.py                 # Learning-rate warmup and cosine scheduling
│   └── utils.py
├── models/
│   ├── almt_dual.py                 # Dual-text model with ordinal prediction
│   ├── almt_layer.py                # ALMT network components
│   ├── bert.py                      # BERT text encoder
│   ├── dual_text_fusion.py          # Dual-text fusion module
│   └── intensity_heads.py           # Monotonic ordinal head and related components
├── scripts/
│   ├── generate_c4_explicit.py      # Offline text enhancement
│   ├── validate_llm_text.py         # Enhanced-text validation
│   ├── build_dual_text_pkl.py       # Dual-text dataset construction
│   ├── evaluate_selected_test.py    # Standalone test evaluation and prediction export
│   ├── bestweight.py                # Validation-based fusion coefficient search
│   └── analyze_acc7.py              # Acc-7 and confusion matrix analysis
├── tests/                           # Unit tests
├── datasets/                        # Datasets; prepare separately
├── pretrained/                      # Pretrained models; prepare separately
├── ckpt/                            # Checkpoints and prediction results
├── log/                             # TensorBoard logs
├── figures/                         # Visualization assets
├── train.py                         # Original ALMT training entry point
├── train_dual.py                    # Training entry point for the proposed method
├── requirements.txt
└── LICENSE
```

## 3. Environment Setup

Run all commands from the project root directory.

```bash
git clone https://github.com/buildmycs/Fine-Grained-LLM-Enhance
cd ALMT

conda create -n llm-almt python=3.11 -y
conda activate llm-almt
```

First, install a PyTorch version compatible with your GPU and CUDA environment. Then install the project dependencies:

```bash
pip install -r requirements.txt
```

Note: PyTorch is not included in `requirements.txt` and must be installed separately.

To generate LLM-enhanced text or plot confusion matrices, install the corresponding additional dependencies:

```bash
# Required for offline LLM generation
pip install accelerate

# Required for confusion matrix visualization
pip install matplotlib
```

## 4. Datasets and Pretrained Models

### 4.1 Data Preparation

Use CMU-MOSI / CMU-MOSEI feature files in the MMSA format. For data preparation, refer to:

- [MMSA](https://github.com/thuiar/MMSA)

Example directory layout:

```text
datasets/
├── mosi/
│   ├── unaligned_50.pkl
│   └── unaligned_50_dual_qwen25_c4.pkl
└── mosei/
    ├── unaligned_50.pkl
    └── unaligned_50_dual_qwen25_c4.pkl

pretrained/
└── bert-base-uncased/
    └── BERT model weights, configuration, and tokenizer files
```

The original PKL file should contain the `train`, `valid`, and `test` splits, including text, audio, visual features, labels, and sample IDs.

The dual-text PKL extends the original dataset with:

- `text_bert_llm`: BERT inputs for the enhanced text.
- `raw_text_llm`: Enhanced text strings.

If the dual-text PKL files are already available, skip the next subsection.

### 4.2 Offline Text Enhancement

Prepare a local copy of Qwen2.5-7B-Instruct beforehand. The following example uses MOSI. Set `--model-path` to the actual location of your model.

Generate enhanced text for all splits:

```bash
python scripts/generate_c4_explicit.py --data-path datasets/mosi/unaligned_50.pkl --model-path ../Qwen2.5-7B-Instruct --output-path datasets/mosi/qwen25_c4_explicit.jsonl --splits train valid test --batch-size 8
```

Validate the generated text:

```bash
python scripts/validate_llm_text.py --data-path datasets/mosi/unaligned_50.pkl --enhanced-path datasets/mosi/qwen25_c4_explicit.jsonl --splits train valid test --bert-path pretrained/bert-base-uncased --report-path datasets/mosi/qwen25_c4_validation.json
```

Build the dual-text PKL:

```bash
python scripts/build_dual_text_pkl.py --data-path datasets/mosi/unaligned_50.pkl --enhanced-path datasets/mosi/qwen25_c4_explicit.jsonl --output-path datasets/mosi/unaligned_50_dual_qwen25_c4.pkl --bert-path pretrained/bert-base-uncased
```

Follow the same procedure for MOSEI, replacing `mosi` with `mosei` in the dataset paths above.

Notes:

- The generation script requires a CUDA-capable GPU.
- Ground-truth sentiment labels or scores must not be included in the enhancement prompts.
- The current method uses both preceding and following context and therefore operates in an offline context-enhancement setting.
- By default, the dataset construction script requires complete enhancement records for all three splits.
- The original PKL file will not be overwritten.

## 5. Training

### 5.1 Configuration Checklist

Adjust the following YAML fields to match your experimental setup:

| Configuration Field | Description |
| --- | --- |
| `base.project_name` | Experiment name; determines the log and checkpoint directories |
| `base.seed` | Random seed |
| `base.lr` | Learning rate |
| `base.batch_size` | Training batch size |
| `base.n_epochs` | Total number of training epochs |
| `base.lr_warmup_epochs` | Fixed number of learning-rate warmup epochs |
| `base.num_workers` | Number of data-loading worker processes |
| `dataset.dataPath` | Path to the dual-text PKL file |
| `model.bert_pretrained` | Path to the local BERT model |
| `model.ordinal_prediction_weight` | Prediction fusion coefficient for the regression and ordinal heads |
| `objective.ordinal_weight` | Ordinal loss weight |
| `objective.auxiliary_warmup_epochs` | Number of auxiliary-loss warmup epochs |

Learning-rate warmup and auxiliary-loss warmup are configured independently.

The repository includes configurations used for hyperparameter exploration. The actual parameter values are determined by the YAML contents and should not be inferred solely from filenames or `project_name`. The contrastive loss weight is set to 0 in the current main configurations; contrastive learning is an optional experimental feature.

To avoid overwriting existing results, use a unique `project_name` for each independent experiment and retain its corresponding configuration.

### 5.2 Train the Full Model

CMU-MOSI:

```bash
python train_dual.py --config_file configs/mosi_dual_c4_intensity.yaml --gpu_id 0
```

CMU-MOSEI:

```bash
python train_dual.py --config_file configs/mosei_dual_c4_intensity.yaml --gpu_id 0
```

The random seed can be overridden from the command line:

```bash
python train_dual.py --config_file configs/mosei_dual_c4_intensity.yaml --gpu_id 0 --seed 42
```

Note: Changing `--seed` does not automatically change the output directory. Assign a separate experiment name to each run when evaluating multiple seeds.

## 6. Checkpoint Selection and Testing

### 6.1 Validation-Based Selection Protocol

Verify the following settings in the full-model configuration:

```yaml
base:
  evaluation_protocol: validation_selected
  selection_metric: Mult_acc_7
  selection_mode: max
  selection_secondary_metric: MAE
  selection_secondary_mode: min
  validation_rho_candidates: null
```

This configuration selects the checkpoint using validation Acc-7, with validation MAE as the secondary selection criterion. Setting `validation_rho_candidates: null` disables per-epoch fusion coefficient search and uses the fixed coefficient specified in the configuration for checkpoint selection.

After training, the program automatically loads the selected checkpoint and evaluates it on the test set.

By default, the following files are saved:

```text
ckpt/<project_name>/
├── best_validation_model.pth
├── best_validation_predictions.npz
├── best_validation_predictions.csv
├── best_validation_selection.json
├── best_test_predictions.npz
└── best_test_predictions.csv
```

Logs are saved to:

```text
log/<project_name>/
```

`best_test_predictions` contains the test predictions of the validation-selected model. It does not indicate that the model was selected based on test-set performance.

## 7. Acc-7 and Prediction Head Analysis

Replace `EXPERIMENT_NAME` with the actual experiment name.

Analyze the fused predictions:

```bash
python scripts/analyze_acc7.py --predictions ckpt/EXPERIMENT_NAME/selected_test_predictions.npz --output-dir ckpt/EXPERIMENT_NAME/analysis_fused
```

Analyze the regression head:

```bash
python scripts/analyze_acc7.py --predictions ckpt/EXPERIMENT_NAME/selected_test_predictions.npz --prediction-key regression_predictions --output-dir ckpt/EXPERIMENT_NAME/analysis_regression
```

Analyze the ordinal head:

```bash
python scripts/analyze_acc7.py --predictions ckpt/EXPERIMENT_NAME/selected_test_predictions.npz --prediction-key ordinal_predictions --output-dir ckpt/EXPERIMENT_NAME/analysis_ordinal
```

The analysis includes Acc-7, macro recall, macro F1, per-class metrics, and confusion matrices.

For validation-set diagnostics, replace the input file with `best_validation_predictions.npz`.
 
 
Fine-Grained Multimodal Sentiment Analysis via LLM-Enhanced Text and Monotonic Ordinal Learning

本项目面向细粒度多模态情感分析，在 ALMT 框架基础上结合 LLM 文本增强、双文本融合与单调有序学习，重点关注 Acc-5 和 Acc-7 等情感强度分类指标。

方法构建从表征到决策的递进过程：首先通过上下文辅助的文本增强补充语义信息，再融合原始文本与增强文本以引导多模态表征学习，最后结合连续回归和有序预测，建模情感强度及其等级关系。

支持 CMU-MOSI 和 CMU-MOSEI 数据集。

## 1. 方法概述

### LLM 文本增强

使用离线部署的 Qwen2.5，根据目标语句及同视频上下文进行语义显式化，包括指代补全、歧义消解和隐含语义补充。

生成过程要求新增信息具有原文证据，并施加保持原始情感极性和强度的约束。当前 C4-Explicit 实现使用目标语句之前最多 4 个片段和之后最多 2 个片段。

LLM 仅用于离线数据预处理，不参与后续多模态模型的联合训练。

### 双文本与多模态表征学习

原始文本和增强文本共享 BERT 编码器，通过以原始文本为锚点的门控交叉注意力进行融合，再利用 ALMT 的语言引导机制整合音频和视觉信息。

### 单调有序预测与回归融合

模型包含连续回归头与单调有序头。有序头使用 6 个累积判断建模从 −3 到 +3 的 7 个情感等级，并通过有序阈值参数化保证累积概率的单调性。

最终预测为：

$$
\hat{y}=(1-\rho)\hat{y}_{reg}+\rho\hat{y}_{ord}
$$

其中：

- `ordinal_prediction_weight` 对应预测融合系数 $\rho$。
- `ordinal_weight` 控制训练时有序损失的权重。

二者作用不同：即使 `ordinal_prediction_weight=0`，只要 `ordinal_weight>0`，有序监督仍然可以影响共享表征的学习。

## 2. 项目结构

```text
ALMT/
├── configs/                         # 数据集、模型与训练配置         
│   ├── mosi_dual_c4_intensity.yaml   # 双文本 + 有序学习
│   └── mosei_dual_c4_intensity.yaml
├── core/
│   ├── dataset.py                   # 原始数据加载
│   ├── dataset_dual.py              # 双文本数据加载
│   ├── intensity_objective.py       # 回归、有序及可选对比损失
│   ├── fusion_selection.py          # 融合系数选择与解析
│   ├── model_selection.py           # 验证集 checkpoint 选择
│   ├── metric.py                    # 评估指标
│   ├── scheduler.py                 # 学习率 warmup 与余弦调度
│   └── utils.py
├── models/
│   ├── almt_dual.py                 # 双文本与有序预测模型
│   ├── almt_layer.py                # ALMT 网络组件
│   ├── bert.py                      # BERT 文本编码器
│   ├── dual_text_fusion.py          # 双文本融合模块
│   └── intensity_heads.py           # 单调有序头等预测组件
├── scripts/
│   ├── generate_c4_explicit.py      # 离线生成增强文本
│   ├── validate_llm_text.py         # 校验增强文本
│   ├── build_dual_text_pkl.py       # 构建双文本数据
│   ├── evaluate_selected_test.py    # 独立测试与预测导出
│   ├── bestweight.py                # 基于验证集搜索融合系数
│   └── analyze_acc7.py              # Acc-7 与混淆矩阵分析
├── tests/                           # 单元测试
├── datasets/                        # 数据集，需自行准备
├── pretrained/                      # 预训练模型，需自行准备
├── ckpt/                            # checkpoint 与预测结果
├── log/                             # TensorBoard 日志
├── figures/                         # 可视化资源
├── train.py                         # 原始 ALMT 训练入口
├── train_dual.py                    # 本方法训练入口
├── requirements.txt
└── LICENSE
```
## 3. 环境配置

以下命令均在项目根目录执行。

```bash
git clone https://github.com/buildmycs/Fine-Grained-LLM-Enhance
cd Fine-Grained-LLM-Enhance

conda create -n llm-almt python=3.11 -y
conda activate llm-almt
```

先安装与本机 GPU、CUDA 环境匹配的 PyTorch，再安装项目依赖：

```bash
pip install -r requirements.txt
```
注意：`requirements.txt` 不包含 PyTorch，需要单独安装。

如需生成 LLM 增强文本或绘制混淆矩阵，补充安装对应依赖：

```bash
# LLM 离线生成所需
pip install accelerate

# 混淆矩阵绘图所需
pip install matplotlib
```

## 4. 数据与预训练模型

### 4.1 数据准备

使用 MMSA 格式的 CMU-MOSI / CMU-MOSEI 特征文件。数据准备可参考：

- [MMSA](https://github.com/thuiar/MMSA)

目录示例：

```text
datasets/
├── mosi/
│   ├── unaligned_50.pkl
│   └── unaligned_50_dual_qwen25_c4.pkl
└── mosei/
    ├── unaligned_50.pkl
    └── unaligned_50_dual_qwen25_c4.pkl

pretrained/
└── bert-base-uncased/
    └── BERT 模型权重、配置与 tokenizer 文件
```

原始 PKL 应包含 `train`、`valid`、`test` 三个划分及文本、音频、视觉、标签和样本 ID。

双文本 PKL 在原始数据基础上增加：

- `text_bert_llm`：增强文本的 BERT 输入。
- `raw_text_llm`：增强后的原始文本。

如果已经准备好双文本 PKL，可以跳过下一节。

### 4.2 离线生成增强文本

提前准备本地 Qwen2.5-7B-Instruct 模型。以下以 MOSI 为例，`--model-path` 按实际模型位置修改。

生成全部划分的增强文本：

```bash
python scripts/generate_c4_explicit.py --data-path datasets/mosi/unaligned_50.pkl --model-path ../Qwen2.5-7B-Instruct --output-path datasets/mosi/qwen25_c4_explicit.jsonl --splits train valid test --batch-size 8
```

校验生成结果：

```bash
python scripts/validate_llm_text.py --data-path datasets/mosi/unaligned_50.pkl --enhanced-path datasets/mosi/qwen25_c4_explicit.jsonl --splits train valid test --bert-path pretrained/bert-base-uncased --report-path datasets/mosi/qwen25_c4_validation.json
```

构建双文本 PKL：

```bash
python scripts/build_dual_text_pkl.py --data-path datasets/mosi/unaligned_50.pkl --enhanced-path datasets/mosi/qwen25_c4_explicit.jsonl --output-path datasets/mosi/unaligned_50_dual_qwen25_c4.pkl --bert-path pretrained/bert-base-uncased
```

MOSEI 使用相同流程，将上述数据路径中的 `mosi` 替换为 `mosei`。

注意：

- 生成脚本需要 CUDA GPU。
- 增强过程不得将真实情感标签或评分作为提示输入。
- 当前方法使用前后文，属于离线上下文增强设置。
- 构建脚本默认要求三个划分的增强记录完整。
- 原始 PKL 不会被覆盖。

## 5. 启动训练

### 5.1 训练前检查

根据运行环境修改 YAML 中的以下字段：

| 配置字段 | 作用 |
| --- | --- |
| `base.project_name` | 实验名称，决定日志和 checkpoint 保存目录 |
| `base.seed` | 随机种子 |
| `base.lr` | 学习率 |
| `base.batch_size` | 训练 batch size |
| `base.n_epochs` | 总训练轮数 |
| `base.lr_warmup_epochs` | 固定的学习率 warmup 轮数 |
| `base.num_workers` | 数据加载进程数 |
| `dataset.dataPath` | 双文本 PKL 路径 |
| `model.bert_pretrained` | 本地 BERT 路径 |
| `model.ordinal_prediction_weight` | 回归头与有序头的预测融合系数 |
| `objective.ordinal_weight` | 有序损失权重 |
| `objective.auxiliary_warmup_epochs` | 辅助损失 warmup 轮数 |

学习率 warmup 与辅助损失 warmup 是两个独立设置。

仓库中包含参数探索配置，实际参数以 YAML 内容为准，不应仅根据文件名或 `project_name` 推断。当前主配置的对比损失权重为 0，对比学习属于可选实验功能。

为避免覆盖已有结果，每次独立实验应使用不同的 `project_name`，并保存对应配置。

### 5.2 训练完整模型

CMU-MOSI：

```bash
python train_dual.py --config_file configs/mosi_dual_c4_intensity.yaml --gpu_id 0
```

CMU-MOSEI：

```bash
python train_dual.py --config_file configs/mosei_dual_c4_intensity.yaml --gpu_id 0
```

可以通过命令行覆盖随机种子：

```bash
python train_dual.py --config_file configs/mosei_dual_c4_intensity.yaml --gpu_id 0 --seed 42
```

注意：修改 `--seed` 不会自动修改输出目录，运行多个种子前需要分别设置实验名称。

## 6. Checkpoint 选择与测试

### 6.1 验证集选择协议

在完整模型的配置中确认：

```yaml
base:
  evaluation_protocol: validation_selected
  selection_metric: Mult_acc_7
  selection_mode: max
  selection_secondary_metric: MAE
  selection_secondary_mode: min
  validation_rho_candidates: null
```

该设置使用验证集 Acc-7 选择 checkpoint，并以验证集 MAE 作为次级选择依据。`validation_rho_candidates: null` 表示不在每个 epoch 搜索融合系数，而是使用配置中的固定系数进行选择。

训练结束后，程序会自动加载选中的 checkpoint 并评估测试集。

默认保存位置为：

```text
ckpt/<project_name>/
├── best_validation_model.pth
├── best_validation_predictions.npz
├── best_validation_predictions.csv
├── best_validation_selection.json
├── best_test_predictions.npz
└── best_test_predictions.csv
```

日志保存在：

```text
log/<project_name>/
```

`best_test_predictions` 表示验证集选中模型的测试预测，并不表示通过测试集挑选出的最佳模型。

## 7. Acc-7 与预测头分析

将 `EXPERIMENT_NAME` 替换为实际实验名称。

分析融合头：

```bash
python scripts/analyze_acc7.py --predictions ckpt/EXPERIMENT_NAME/selected_test_predictions.npz --output-dir ckpt/EXPERIMENT_NAME/analysis_fused
```

分析回归头：

```bash
python scripts/analyze_acc7.py --predictions ckpt/EXPERIMENT_NAME/selected_test_predictions.npz --prediction-key regression_predictions --output-dir ckpt/EXPERIMENT_NAME/analysis_regression
```

分析有序头：

```bash
python scripts/analyze_acc7.py --predictions ckpt/EXPERIMENT_NAME/selected_test_predictions.npz --prediction-key ordinal_predictions --output-dir ckpt/EXPERIMENT_NAME/analysis_ordinal
```

分析内容包括 Acc-7、Macro Recall、Macro F1、各类别指标及混淆矩阵。

也可以将输入替换为 `best_validation_predictions.npz`，用于验证集诊断。


