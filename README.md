# ESP-Simulator

## 1. Installation

```bash
uv sync
uv pip install .
```

## 2. Prepare base agents

### Train agents

```bash
# Partially observable agent
python scripts/train_agent.py

# Fully observable agent
python scripts/train_agent.py --agent_view_size -1
```

### Test agents

```bash
# Partially observable agent
python scripts/test_agent.py --checkpoint CHECKPOINT_PATH

# Fully observable agent
python scripts/test_agent.py --agent_view_size -1 --checkpoint CHECKPOINT_PATH

# Optimal agent
python scripts/test_agent.py --agent_view_size -1 --optimal
```

### Evaluate agents

```bash
# Partially observable agent
python scripts/evaluate_agent.py --checkpoint CHECKPOINT_PATH

# Fully observable agent
python scripts/evaluate_agent.py --agent_view_size -1 --checkpoint CHECKPOINT_PATH

# Optimal agent
python scripts/evaluate_agent.py --agent_view_size -1 --optimal
```

## 3. Prepare base classifier

### Generate data for classifier

```bash
# Partially observable agent
## Train
python scripts/gendata_classifier.py \
        --checkpoint CHECKPOINT_PATH \
        --seed_envs 0-9999 \
        --output_path ./output/train_po.npy
## Val
python scripts/gendata_classifier.py \
        --checkpoint CHECKPOINT_PATH \
        --seed_envs 10000-11999 \
        --output_path ./output/val_po.npy
## Test
python scripts/gendata_classifier.py \
        --checkpoint CHECKPOINT_PATH \
        --seed_envs 12000-13999 \
        --output_path ./output/test_po.npy

# Fully observable agent
## Train
python scripts/gendata_classifier.py --agent_view_size -1 \
        --checkpoint CHECKPOINT_PATH \
        --seed_envs 0-9999 \
        --output_path ./output/train_fo.npy
## Val
python scripts/gendata_classifier.py --agent_view_size -1 \
        --checkpoint CHECKPOINT_PATH \
        --seed_envs 10000-11999 \
        --output_path ./output/val_fo.npy
## Test
python scripts/gendata_classifier.py --agent_view_size -1 \
        --checkpoint CHECKPOINT_PATH \
        --seed_envs 12000-13999 \
        --output_path ./output/test_fo.npy
```

### Train classifier

```bash
python scripts/train_classifier.py \
        --dataset_normal_train ./output/train_po.npy \
        --dataset_normal_val ./output/val_po.npy \
        --dataset_esp_train ./output/train_fo.npy \
        --dataset_esp_val ./output/val_fo.npy
```

### Evaluate classifier

```bash
python scripts/evaluate_classifier.py \
        --checkpoint CHECKPOINT_PATH \
        --dataset_normal ./output/test_po.npy \
        --dataset_esp ./output/test_fo.npy
```

## 4. Train adversarial agent

```bash
# Train agent and classifier.
# Remove `--train_classifier` if you don't want to train the classifier.
python scripts/train_adversarial_agent.py --agent_view_size -1 \
        --checkpoint CHECKPOINT_CHEATER_PATH \
        --checkpoint_normal CHECKPOINT_NORMAL_PATH \
        --checkpoint_classifier CHECKPOINT_CLASSIFIER_PATH \
        --adv_coef 0.1 \
        --train_classifier
```
