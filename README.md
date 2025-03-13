# ESP-Simulator

## Installation

```bash
uv sync
uv pip install .
```

## Train agents

```bash
# Partially observable agent
python scripts/train_agent.py

# Fully observable agent
python scripts/train_agent.py --agent_view_size -1
```

## Test agents

```bash
# Partially observable agent
python scripts/test_agent.py --checkpoint CHECKPOINT_PATH

# Fully observable agent
python scripts/test_agent.py --agent_view_size -1 --checkpoint CHECKPOINT_PATH

# Optimal agent
python scripts/test_agent.py --agent_view_size -1 --optimal
```

## Evaluate agents

```bash
# Partially observable agent
python scripts/evaluate_agent.py --checkpoint CHECKPOINT_PATH

# Fully observable agent
python scripts/evaluate_agent.py --agent_view_size -1 --checkpoint CHECKPOINT_PATH

# Optimal agent
python scripts/evaluate_agent.py --agent_view_size -1 --optimal
```
