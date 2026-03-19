# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

FedLLM_CE contains **GhostSuite**, a PyTorch framework for efficient per-sample gradient information computation for data-centric ML research (data selection, reweighting, synthetic data generation). Based on the ICLR'25 Outstanding Paper Runner-up ["Data Shapley in One Training Run"](https://openreview.net/pdf?id=HD6bWcj87Y).

The core insight: per-sample gradient dot-products between validation and training samples can be computed in a **single backpropagation pass** by reusing activations and output gradients already computed during standard backprop — avoiding full gradient materialization.

## Installation

```bash
cd GhostSuite
pip install -r requirements.txt
```

## Running Examples

```bash
cd GhostSuite/examples

# Minimal MLP demo (GradDotProd)
python ghost_mlp.py

# Gradient projection for MLP
python ghost_gradproj_mlp.py --mode project --proj_rank_total 64

# Gradient projection for language models
python ghost_gradproj_lm.py --proj_layers "attn.c_attn,mlp.c_fc"

# TorchTitan LLM pretraining with GradDotProd (Llama 3 130M)
CONFIG_FILE="./torchtitan/torchtitan/models/llama3/train_configs/llama3_130m_ghost.toml" ./torchtitan/run_train_with_ghost.sh
```

> **Note**: `GradDotProd_LM/` and `GradProj_LM/` examples are deprecated in v0.33. Use v0.2 if needed.

## Architecture

### Two Core Engines

**`GradDotProdEngine`** (`ghostEngines/graddotprod_engine.py`)
- **Use when**: online, per-step gradient similarity is needed (data selection, reweighting, curriculum learning, auditing training dynamics)
- Concatenates validation + training batches in a single forward/backward pass
- Recovers aggregated training gradients into `.grad` for the optimizer step
- Key implementation: `autograd_grad_sample_dotprod.py` uses `_NamedSavedTensorManager` (a scope-stack-based thread-safe tensor capture) to intercept saved tensors during backprop

**`GradProjLoRAEngine`** (`ghostEngines/gradProjection/gradproj_engine.py`)
- **Use when**: offline, corpus-scale gradient analysis for a fixed checkpoint
- Applies Kronecker-structured random projection $P = P_i \otimes P_o$ via a zero-impact LoRA-style side branch
- Stores projected per-sample gradients to disk for later similarity computation
- Preserves inner products up to Johnson-Lindenstrauss distortion

### `GhostEngineManager` — The Integration Interface

`ghostEngines/engine_manager.py` is the entry point for training loop integration. It auto-selects the engine based on config and provides a method-agnostic API:

```python
from ghostEngines import GhostEngineManager

ghost_engine = GhostEngineManager(config, model, optimizer, ddp_info, val_data)

for iteration in range(max_steps):
    optimizer.zero_grad(set_to_none=True)
    ghost_engine.attach_train_batch(X_train, Y_train, iteration, batch_idx)
    X_fwd, Y_fwd = ghost_engine.prepare_forward_input(X_train, Y_train)

    with ghost_engine.saved_tensors_context():   # no-op for non-GradDotProd
        loss = model(input_ids=X_fwd, labels=Y_fwd).loss
        loss.backward()

    ghost_engine.prepare_gradients()  # moves accumulated grads to .grad
    optimizer.step()
    ghost_engine.aggregate_and_log()
    ghost_engine.clear_gradients()
```

With gradient accumulation: call `aggregate_and_log()` after each microbatch; move `prepare_gradients()` / `optimizer.step()` to the end of the accumulation window.

For evaluation: use `ghost_engine.detach_for_evaluation()` / `ghost_engine.reattach_after_evaluation()`.

### Layer Support

`supported_layers_grad_samplers_dotprod.py` implements per-sample gradient computation for: Linear, Conv2D, Embedding, LayerNorm, and attention layers.

`transformers_support.py` patches Hugging Face models for compatibility (dummy bias handling, embedding fixes).

`ghostEngines/gradProjection/lora_modules.py` implements the LoRA side branches used by GradProjLoRAEngine without affecting model forward pass outputs.

### TorchTitan Integration

`examples/torchtitan/` integrates GhostSuite with PyTorch's TorchTitan training stack for large-scale LLM pretraining. Config via `.toml` files (see `llama3_130m_ghost.toml` for the ghost-specific `[ghost]` section).
