# ♻️ Tu(r)ning AI Green

*Quantifying how energy-efficient techniques applied at different stages of an AI pipeline interact, stack, and occasionally collide.*

---

## Table of Contents
1. [Project Goals](#project-goals)
2. [Supported Tasks & Models](#supported-tasks--models)
3. [Repository Layout](#repository-layout)
4. [Running an Experiment](#running-an-experiment)
5. [Energy & Carbon Instrumentation](#energy--carbon-instrumentation)
6. [Result Files](#result-files)

---

## Project Goals

Large-scale AI systems burn substantial energy across **five canonical stages**:

1. **Data preparation**
2. **Model Architecture design / selection**
3. **Training or fine-tuning**
4. **System-level deployment**
5. **Inference**

This study builds *multiple variants* of the **same end-to-end pipeline**, each enabling energy-saving techniques in **one or more** of those stages. By comparing energy usage, carbon footprint, latency and task accuracy across variants, we ask:

*Do savings add up linearly, cancel one another out, or compound super-linearly?*  
*Does "optimise everywhere" always beat "optimise the bottleneck"?*  

---

## Supported Tasks & Models

| Task | Dataset | Model(s) | Notes |
|------|---------|----------|-------|
| **Vulnerability Detection** | **BigVul** | **ModernBERT** | Binary vulnerability classification |

> Each pipeline variant re-uses the same datasets & model checkpoints so that only the *energy-efficiency knobs* differ.

---

## Repository Layout

```
green-pipeline-study/
├── variants/                # One folder per experimental pipeline
│   ├── v0_baseline/        # Baseline implementation
│   ├── v1_gradient_checkpointing/
│   ├── v2_lora_peft/       # Parameter efficient fine-tuning
│   ├── v3_quantization/    # Model quantization
│   ├── v4_tokenizer/       # Tokenizer optimizations
│   ├── v5_power_limit_100W/ # Power limiting
│   ├── v6_optimizer/       # Optimizer configurations
│   ├── v7_f16/            # FP16 precision
│   ├── v8_sequence_length_trimming/
│   ├── v9_inference_engine/
│   ├── v10_dataloader_pin_memory/
│   ├── v11_torch_compile/
│   ├── v12_attention/
│   ├── v13_layer_pruning_4_top/
│   └── ...                 # Additional variants
│
├── common/                 # Shared components
│   ├── layer_drop.py      # Layer pruning utilities
│   └── generate_configs.py # Configuration generation
│
├── analysis_results/      # Analysis outputs
├── energy_modelling.py    # Energy modeling utilities
├── analysis.py           # Analysis scripts
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

Each variant in **`variants/`** is **self-contained** with:
* `config.yaml` – structured hyper-parameters
* `train_all_variants.sh` – pipeline execution script

---

## Running an Experiment

```bash
# 1. Set up environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run a pipeline variant
cd variants/v0_baseline  # or any other variant
python3 train.py
```

All energy, carbon and performance numbers are saved under `analysis_results/`.

---

## Energy & Carbon Instrumentation

| Layer | Tool | What it Measures |
|-------|------|------------------|
| CPU/GPU/RAM system | **CodeCarbon** | Process-level energy + g CO₂/kWh |

Each stage starts an **energy session**; deltas are aggregated into the final analysis.

---

## Result Files

A run produces:

* `<VARIANT>/results/*`

You can also find the analyzed and combined result along with plots in `analysis_results/`.

  ```jsonc
  {
    "variant": "v2_lora_peft",
    "energy_kwh": {
      "data": 0.14,
      "architecture": 0.00,
      "training": 1.92,
      "system": 0.08,
      "inference": 0.37,
      "total": 2.51
    },
    "co2_kg": 0.98,
    "accuracy": 0.842,
    "latency_ms": 23.5
  }
  ```
* `analysis_results/aggregated_metrics.json` – comprehensive analysis across all variants
* Training progress logs under `training_progress.log`


