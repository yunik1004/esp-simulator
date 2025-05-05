# ESP-Simulator

## 1. Installation

```bash
uv sync
uv pip install .
```

## 2. Prepare base agents

### Train agents

```bash
python scripts/train_agent.py [--cheater] --seed 0
```

### Test agents

```bash
python scripts/test_agent.py [--cheater | --mixed] --checkpoint CHECKPOINT_PATH
```

### Evaluate agents

```bash
python scripts/evaluate_agent.py [--cheater | --mixed] \
        --checkpoint CHECKPOINT_PATH \
        --seed_envs 12000-13999
```

## 3. Prepare mixed agents

```bash
python scripts/construct_mixed_agent.py \
        --checkpoint CHECKPOINT_CHEATER_PATH \
        --checkpoint_normal CHECKPOINT_NORMAL_PATH
```

## 4. Prepare base classifier

### Generate data for classifier

```bash
## Train
python scripts/gendata_classifier.py [--cheater | --mixed] \
        --checkpoint CHECKPOINT_PATH \
        --seed_envs 0-9999 \
        --output_path ./output/train_data.npy
## Val
python scripts/gendata_classifier.py [--cheater | --mixed] \
        --checkpoint CHECKPOINT_PATH \
        --seed_envs 10000-11999 \
        --output_path ./output/val_data.npy
## Test
python scripts/gendata_classifier.py [--cheater | --mixed] \
        --checkpoint CHECKPOINT_PATH \
        --seed_envs 12000-13999 \
        --output_path ./output/test_data.npy
```

### Train classifier

```bash
python scripts/train_classifier.py \
        --features_dim 64 \
        --dataset_normal_train ./output/train_noncheater.npy \
        --dataset_normal_val ./output/val_noncheater.npy \
        --dataset_esp_train ./output/train_cheater.npy \
        --dataset_esp_val ./output/val_cheater.npy
```

### Evaluate classifier

```bash
python scripts/evaluate_classifier.py \
        --features_dim 64 \
        --checkpoint CHECKPOINT_PATH \
        --dataset_normal ./output/test_noncheater.npy \
        --dataset_esp ./output/test_cheater.npy
```

## 5. Train adversarial agent

```bash
# Remove `--train_classifier` if you don't want to train the detector.
python scripts/train_adversarial_agent.py --mixed \
        --checkpoint CHECKPOINT_CHEATER_PATH \
        --checkpoint_normal CHECKPOINT_NORMAL_PATH \
        --features_dim 64 \
        --checkpoint_classifier CHECKPOINT_DETECTOR_PATH \
        --adv_coef 0.1 \
        --train_classifier
```
