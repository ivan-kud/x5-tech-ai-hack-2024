# Гиперпараметры дообучения моделей

## `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`

```python
OPTIMIZER = "adamw_torch"
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 3e-5

SCHEDULER = "cosine"
WARMUP_RATIO = 0.00
WARMUP_STEPS = 0

TRAIN_EPOCHS = 2
LOGGING_STEPS = 10

USE_CPU = False  # Used 2 GPUs
AUTO_FIND_BATCH_SIZE = False
PER_DEVICE_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1
```

## `sileod/deberta-v3-large-tasksource-nli`

```python
OPTIMIZER = "adamw_torch"
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 3e-5

SCHEDULER = "cosine"
WARMUP_RATIO = 0.00
WARMUP_STEPS = 0

TRAIN_STEPS = 80
LOGGING_STEPS = 10

USE_CPU = False  # Used 2 GPUs
AUTO_FIND_BATCH_SIZE = False
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
```

## `microsoft/deberta-v2-xlarge-mnli`

```python
OPTIMIZER = "adamw_torch"
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 3e-5

SCHEDULER = "cosine"
WARMUP_RATIO = 0.00
WARMUP_STEPS = 0

TRAIN_STEPS = 80
LOGGING_STEPS = 10

USE_CPU = False  # Used 2 GPUs
AUTO_FIND_BATCH_SIZE = False
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
```
