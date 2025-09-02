# ESP-Simulator

## 1. Installation

```bash
uv sync
uv pip install .
```

## 2. Prepare base agents

### Train agents

```bash
# Specify `--cheater` to use pure cheater
python scripts/grid_world/train_agent.py [--cheater] --seed 0
```

### Test agents

```bash
# Specify `--mixed` to use (mixed) cheater
python scripts/grid_world/test_agent.py [--cheater | --mixed] --checkpoint CHECKPOINT_PATH
```

### Evaluate agents

```bash
python scripts/grid_world/evaluate_agent.py [--cheater | --mixed] \
        --checkpoint CHECKPOINT_PATH \
        --seed_envs 12000-13999
```

## 3. Prepare mixed agents

```bash
python scripts/grid_world/construct_mixed_agent.py \
        --checkpoint CHECKPOINT_CHEATER_PATH \
        --checkpoint_normal CHECKPOINT_NORMAL_PATH
```

## 4. Prepare base classifier

### Generate data for classifier

```bash
## Train
python scripts/grid_world/gendata_classifier.py [--cheater] \
        --checkpoint CHECKPOINT_PATH \
        --seed_envs 0-9999 \
        --output_path ./output/train_data.npy
## Val
python scripts/grid_world/gendata_classifier.py [--cheater] \
        --checkpoint CHECKPOINT_PATH \
        --seed_envs 10000-11999 \
        --output_path ./output/val_data.npy
## Test
python scripts/grid_world/gendata_classifier.py [--cheater | --mixed] \
        --checkpoint CHECKPOINT_PATH \
        --seed_envs 12000-13999 \
        --output_path ./output/test_data.npy
```

### Train classifier

```bash
python scripts/grid_world/train_classifier.py \
        --dataset_normal_train ./output/train_noncheater.npy \
        --dataset_normal_val ./output/val_noncheater.npy \
        --dataset_esp_train ./output/train_cheater.npy \
        --dataset_esp_val ./output/val_cheater.npy
```

### Evaluate classifier

```bash
python scripts/grid_world/evaluate_classifier.py \
        --checkpoint CHECKPOINT_PATH \
        --dataset_normal ./output/test_noncheater.npy \
        --dataset_esp ./output/test_cheater.npy
```

## 5. Train adversarial agent

```bash
# Remove `--train_classifier` if you don't want to train the detector.
python scripts/grid_world/train_adversarial_agent.py --mixed \
        --checkpoint CHECKPOINT_CHEATER_PATH \
        --checkpoint_normal CHECKPOINT_NORMAL_PATH \
        --checkpoint_classifier CHECKPOINT_DETECTOR_PATH \
        --adv_coef 1.0 \
        --train_classifier
```
