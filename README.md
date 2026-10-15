# P-CBM-2S

Implementation of **Concept-Gated Multimodal Structural Fusion for Drug–Target Interaction Prediction**.

P-CBM-2S combines ligand and protein graphs with frozen MolT5 and ProtT5 representations. Five physicochemical descriptors condition the pair representation through descriptor concatenation and residual gating. Auxiliary reconstruction supervises concept information, and exact Shapley analysis quantifies concept contributions through the gate.

## Repository contents

- Ligand GraphMamba and protein GNNPlus-style encoders with RWSE.
- Eight-head within-entity graph–sequence fusion.
- Descriptor-conditioned prediction and auxiliary supervision.
- Training, evaluation, and exact five-concept gate attribution.
- Graph construction, sequence embeddings, interface descriptors, and four evaluation split protocols.

This source-code package uses user-provided datasets, pretrained weights, and prepared docking poses.

## Installation

Use a dedicated Python 3.10 or 3.11 environment. Install a suitable [PyTorch build](https://pytorch.org/get-started/previous-versions/), then run:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The GraphMamba encoder uses `mamba_ssm.Mamba` and its selective-scan backend. Use a compatible Linux/CUDA environment and follow the [Mamba installation instructions](https://github.com/state-spaces/mamba). The reference dependency combination is PyTorch 2.1.2/CUDA 11.8, Mamba 2.2.2, and causal-conv1d 1.4.0. Install PyTorch before Mamba; consult [PyG installation guidance](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for optional extensions.

For a CPU workflow check, select the reduced GNNPlus diagnostic configuration explicitly:

```bash
python -m pcbm2s.smoke --device cpu --encoder gnnplus
```

## Prepare inputs

Prepare `train.pt`, `val.pt`, and `test.pt` as lists of graph samples containing both entity graphs, pooled sequence embeddings, five measured descriptors, and a binary label. Use trusted PyTorch files: loading pickled inputs can execute code.

See [data format](docs/data_format.md), [feature preparation](docs/feature_preparation.md), and [evaluation splits](docs/splits.md). Optional T5 extraction dependencies are in `requirements-lm.txt`.

Validate inputs and compute missing RWSE tensors:

```bash
python -m pcbm2s.preprocess --input data/raw/train.pt --output data/prepared/train.pt --protein-rwse
python -m pcbm2s.preprocess --input data/raw/val.pt --output data/prepared/val.pt --protein-rwse
python -m pcbm2s.preprocess --input data/raw/test.pt --output data/prepared/test.pt --protein-rwse
```

## Train and evaluate

Training fits descriptor statistics on the training split and selects the checkpoint using validation AUROC. Evaluation uses the decision threshold saved from validation.

```bash
python -m pcbm2s.train --data-dir data/prepared --output-dir runs/seed42 --seed 42 --config configs/default.json
python -m pcbm2s.evaluate --checkpoint runs/seed42/best.pt --data data/prepared/test.pt --output predictions.csv
```

Use a separate output directory for each seed. Keep feature preparation, split manifests, configuration, and checkpoint metadata together for reproducibility.

## Explain the concept gate

```bash
python -m pcbm2s.attribute --checkpoint runs/seed42/best.pt --data data/prepared/test.pt --output attributions.json --index 0
```

Exact Shapley attribution evaluates all 32 concept coalitions. The pair representation and descriptor-concatenation input remain fixed; absent gate concepts take their training-set mean. Contributions sum to the probability difference between the measured-concept gate and this reference. Positive and negative values indicate increases and decreases relative to the reference.

## Configuration and tests

`configs/default.json` specifies five ligand GraphMamba layers, six protein GINE layers, RWSE injection, eight-head fusion, concept gating, and auxiliary reconstruction. Feature dimensions and positional encodings should match the selected configuration and checkpoint.

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

See [model details](docs/model_and_provenance.md) and [verification](docs/verification.md). Third-party sources and license terms are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSE](LICENSE).

## 中文说明

本代码包提供模型、特征准备、四类划分、训练、评估及概念门控归因流程。数据集、预训练权重和对接构象由使用者准备。安装依赖后，可先运行上述 CPU 合成样例检查，再按数据格式说明准备真实输入。完整 GraphMamba 配置使用兼容的 Linux/CUDA 与 Mamba 环境。
