# TreeHCA: Tree-Based Hindsight Credit Assignment for Multi-Turn LLM Agents

## Overview

Outcome-supervised group-relative policy optimization (GRPO) gives every token in an LLM agent's trajectory the same advantage, leaving useful and unhelpful decisions undifferentiated. Credit assignment can refine this feedback, but its estimates depend on trajectories containing informative alternatives at consequential histories. We introduce **TreeHCA**, which couples *sampling via Hindsight ratios* to generate these alternatives with *credit assignment to samples* to evaluate their contribution. We use changes in reference-answer likelihood to revisit intermediate histories and select alternative continuations, forming a tree of reasoning and tool interactions. This exploration changes the distribution of sampled continuations. We derive the corresponding inverse-hindsight importance correction, including an exact finite-pool identity, and use its structure in a normalized tree backup that propagates terminal outcomes into turn-level advantages. This common hindsight structure is TreeHCA's central design principle, linking informative exploration to sampling-aware credit. Across seven question-answering benchmarks and three Qwen backbones, TreeHCA outperforms every baseline in aggregate exact match, validating our paired design for multi-turn LLM training.

## Setup

### Local Retriever Tool Initialization

#### Environment
```
conda create -y -n retriever_env python=3.10
conda activate retriever_env
conda install -y pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.44.2 datasets==4.8.5 pyserini
conda install -y -c pytorch -c nvidia faiss-gpu=1.8.0
pip install uvicorn fastapi
```

#### Download Retriever Data
```
mkdir rag_data
python rag_server/download.py
cat rag_data/part_* > rag_data/e5_Flat.index
gzip -d rag_data/wiki-18.jsonl.gz
```

### Training Environment Installation
CUDA and GCC need to be preinstalled
```
conda create -n treehca_env python==3.10 -y
conda activate treehca_env

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -r TreeHCA/requirements.txt
```

### Dataset
```
mkdir rl_datasets
python data_process/hotpotqa_multihop_train.py
python data_process/multihop_test.py
python data_process/nq_singlehop_train.py
python data_process/singlehop_test.py
```

### Models
```
mkdir models
huggingface-cli download Qwen/Qwen3-4B --local-dir models/Qwen3-4B
huggingface-cli download Qwen/Qwen3-8B --local-dir models/Qwen3-8B
```

### Logs
```
mkdir logs
mkdir search_cache
```

## Usage

### Train
```
bash train.sh <atpo|treehca> <hotpotqa|nq> <qwen4b|qwen8b>
```

### Eval
```
bash eval.sh <atpo|treehca> <hotpotqa|nq> <checkpoint_path (global_step_...)>
```
