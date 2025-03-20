# LLMs Can't Please Them All

## Overview

This notebook demonstrates how to generate essays using pre-trained transformer models, specifically focusing on GPT-Neo, GPT-2, and T5. It covers GPU utilization, model loading, essay generation, and saving the generated essays to files.

## Environment Setup

### GPU Verification

The notebook begins by verifying the availability and status of a GPU using `nvidia-smi` and PyTorch's CUDA utilities.

```python
gpu_info = !nvidia-smi
# ... (GPU information display) ...

import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
```

This ensures that the notebook can leverage GPU acceleration for model inference.

## Model Loading

The `load_model` function loads a specified pre-trained model and its tokenizer. It supports both causal language models (like GPT-Neo and GPT-2) and sequence-to-sequence models (like T5).

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_name, device=device, low_cpu_mem_usage=False):
    # ... (model loading logic) ...
```

## Essay Generation

The `generate_and_save_essays` function generates essays for a list of topics using a specified model. It handles model-specific parameters and saves the generated essays to individual text files.

```python
def generate_and_save_essays(topics, model_name, pipeline_function, tokenizer):
    # ... (essay generation logic) ...
```

## Logging

Logging is configured to provide detailed information about the notebook's execution.

```python
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug("Logging is initialized")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

if device.type == 'cuda':
    torch.cuda.set_device(0)

def flush_logs():
    for handler in logging.getLogger().handlers:
        handler.flush()
```

## CUDA Memory Check

The notebook includes a check for CUDA memory usage to monitor GPU utilization.

```python
if torch.cuda.is_available():
    current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(current_device)} bytes")
    logging.info(f"CUDA memory cached: {torch.cuda.memory_cached(current_device)} bytes")
```

