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
python scripts/gendata_classifier.py
```

### Train classifier

```bash
python scripts/train_classifier.py
```

### Evaluate classifier

```bash
python scripts/evaluate_classifier.py
```

## 4. Train adversarial agent

```bash
# Train agent only
python scripts/train_adversarial_agent.py --agent_view_size -1 --checkpoint CHECKPOINT_PATH --checkpoint_classifier CHECKPOINT_CLASSIFIER_PATH

# Train both agent and classifier
python scripts/train_adversarial_agent.py --agent_view_size -1 --checkpoint CHECKPOINT_PATH --checkpoint_normal CHECKPOINT_NORMAL_PATH --checkpoint_classifier CHECKPOINT_CLASSIFIER_PATH
```
