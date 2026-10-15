# P-CBM-2S

Code for **Concept-Gated Multimodal Structural Fusion for Drug–Target Interaction Prediction**.

P-CBM-2S combines ligand graph, protein residue graph, and precomputed molecular/protein language-model representations. Five measured physicochemical descriptors enter the predictor through concatenation and a residual gate. An auxiliary head reconstructs the descriptors from the drug–target pair representation. Exact Shapley analysis isolates their contributions through the gate.

This release contains model, training, evaluation, graph/embedding preparation, identity-grouped splitting, descriptor extraction, and attribution code. Datasets, pretrained embeddings, docking poses, and trained checkpoints are not included. The CPU smoke example generates synthetic tensors and tests the software workflow; it does not represent an experimental result.

## Installation

Use Python 3.10 or 3.11 in a dedicated environment. Install a PyTorch build appropriate for your machine first, following the [official instructions](https://pytorch.org/get-started/previous-versions/). Then, from this repository:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pcbm2s.smoke --device cpu
```

The CPU smoke test explicitly selects a small GNNPlus diagnostic configuration. Full GraphMamba runs require a working `mamba_ssm.Mamba` installation and its selective-scan backend. Linux with an NVIDIA CUDA environment is the recommended route for that configuration. Follow the [Mamba installation instructions](https://github.com/state-spaces/mamba), choosing a version compatible with your installed PyTorch and CUDA. The retained research environment used `mamba-ssm==2.2.2`, `causal-conv1d==1.4.0`, and PyTorch 2.1.2/CUDA 11.8. This is a historical environment reference, not a claim that every newer dependency combination has been tested.

For example, the research-era CUDA PyTorch build can be installed using:

```bash
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

Install PyTorch before Mamba, and consult the instructions for the selected Mamba version. Current Mamba releases have different CUDA-extension installation options from version 2.2.2. No private wheel paths or Windows Mamba compatibility are assumed. [PyG installation guidance](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) covers optional extensions if your environment requires them.

## Data interface

Provide trusted PyTorch files containing lists of graph samples. Each sample includes ligand/protein graphs, pooled MolT5/ProtT5 embeddings, the five measured descriptors, and a binary label. See [the input schema](docs/data_format.md) and [model details](docs/model_and_provenance.md). Utilities for graph construction from SMILES/PDB and frozen T5 embeddings are described in [feature preparation](docs/feature_preparation.md); optional language-model dependencies are listed in `requirements-lm.txt`.

Validate a split and compute missing ligand RWSE values:

```bash
python -m pcbm2s.preprocess --input data/raw/train.pt --output data/prepared/train.pt --protein-rwse
python -m pcbm2s.preprocess --input data/raw/val.pt --output data/prepared/val.pt --protein-rwse
python -m pcbm2s.preprocess --input data/raw/test.pt --output data/prepared/test.pt --protein-rwse
```

PyTorch `.pt` files may contain pickled objects. Use files from sources you trust. Preprocessing validates the declared feature contract; it cannot establish biological correctness or recover missing sequence/structure provenance.

## Train and evaluate

The prepared directory should contain `train.pt`, `val.pt`, and `test.pt`. Training computes descriptor statistics from the training split, selects the checkpoint by validation AUROC, and saves configuration and normalization statistics with the model.

```bash
python -m pcbm2s.train --data-dir data/prepared --output-dir runs/full --seed 42 --config configs/default.json
python -m pcbm2s.evaluate --checkpoint runs/full/best.pt --data data/prepared/test.pt --output predictions.csv
```

Run additional seeds explicitly, using a different output directory for each run. The default configuration is an audited reference configuration; reproducing any manuscript table also requires its actual data, splits, run-specific settings, and checkpoints.

## Attribute the concept gate

```bash
python -m pcbm2s.attribute --checkpoint runs/full/best.pt --data data/prepared/test.pt --output attributions.json --index 0
```

Attribution enumerates all 32 coalitions of the five concepts. The train-set mean is the reference, the descriptor-concatenation input remains fixed, and coalition substitutions affect the gate. The pair representation is encoded once and reused, so stochastic GraphMamba permutations do not vary between coalitions. Contributions explain the gate-mediated probability difference relative to this reference, rather than the entire prediction or a causal binding mechanism. Negative contributions are valid.

## Configuration and implementation status

- `configs/default.json`: five ligand GraphMamba layers, six protein GNNPlus-style GINE layers, eight-head within-entity fusion, descriptor concatenation, residual gating, and auxiliary reconstruction.
- RWSE tensors and graph feature dimensions must match the selected configuration and checkpoint.
- Frozen language-model features are inputs; the downloaded language-model weights are not bundled.
- Release tooling fails on malformed/nonfinite data rather than silently skipping batches. Standard F1 and threshold-based metrics are explicitly defined by the evaluation code.

The [identity split utility](docs/splits.md) supports all four evaluation protocols and records double-cold exclusions. [Feature preparation](docs/feature_preparation.md) covers graph construction, frozen sequence embeddings and measured interface descriptors. No docking executable or model weights are redistributed.

Run offline checks using `python -m unittest discover -s tests -v`. See [model and provenance notes](docs/model_and_provenance.md) for the equations and differences from the retained research scripts. Third-party attributions appear in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The license for original project code is pending author designation; see [LICENSE](LICENSE).

## 中文快速开始

本仓库仅提供代码，不包含数据集、真实样本、预训练嵌入和训练权重。先安装 PyTorch 与 `requirements.txt` 中的依赖，再执行 `python -m pip install -e . --no-deps`。可用 `python -m pcbm2s.smoke --device cpu` 检查合成样例的前向、训练及归因流程；完整 GraphMamba 配置建议在 Linux/CUDA 环境安装兼容版本的 Mamba。

真实数据需预先准备成 `train.pt`、`val.pt`、`test.pt`，字段见 `docs/data_format.md`。使用上方的 `preprocess`、`train`、`evaluate` 和 `attribute` 命令即可依次校验、训练、评价和归因。RWSE 与节点特征维度应与具体配置及 checkpoint 保持一致。公开 GitHub 链接由仓库上传后生成；本代码包不预设作者账号或仓库地址。
