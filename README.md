# ♻️ Green-Pipeline Study

*Quantifying how energy-efficient techniques applied at different stages of an AI pipeline interact, stack, and occasionally collide.*

---

## Table of Contents
1. [Project Goals](#project-goals)
2. [Research Questions](#research-questions)
3. [Supported Tasks & Models](#supported-tasks--models)
4. [Repository Layout](#repository-layout)
5. [Running an Experiment](#running-an-experiment)
6. [Energy & Carbon Instrumentation](#energy--carbon-instrumentation)
7. [Result Files](#result-files)

---

## Project Goals

Large-scale AI systems burn substantial energy across **five canonical stages**:

1. **Data preparation**
2. **Architecture design / selection**
3. **Training or fine-tuning**
4. **System-level deployment**
5. **Inference**

This study builds *multiple variants* of the **same end-to-end pipeline**, each enabling energy-saving techniques in **one or more** of those stages. By comparing energy usage, carbon footprint, latency and task accuracy across variants, we ask:

*Do savings add up linearly, cancel one another out, or compound super-linearly?*  
*Does “optimise everywhere” always beat “optimise the bottleneck”?*  

---

## Research Questions

| ID | Question | Metric Highlights |
|----|----------|-------------------|
| **RQ1** | *Stage-wise contribution* – How much energy does each individual stage save? | Δ kWh, Δ CO₂, Δ accuracy |
| **RQ2** | *Additive vs. synergistic effects* – Do multi-stage optimisations add (linear) or compound (sub / super-linear)? | *Synergy factor* |
| **RQ3** | *Performance / energy trade-off* – What price (accuracy / latency) do we pay for greener pipelines? | Accuracy, latency |
| **RQ4** | *Return on investment* – Which stage yields the biggest bang-for-buck? | kWh-saved ÷ Δ accuracy |

---

## Supported Tasks & Models

| Task | Dataset | Model(s) | Notes |
|------|---------|----------|-------|
| **Software Generation (Translation)** | CodexGLUE `icse24-lost-in-translation` | **SmalLm (Qwen-2.5 B)** | Treat as code-to-code translation |
| **Vulnerability Classification** | **BigVul** | **ModernBERT** | Binary vulnerability label |

> Each pipeline variant re-uses the same datasets & model checkpoints so that only the *energy-efficiency knobs* differ.

---

## Repository Layout

```
green-pipeline-study/
├── variants/                # One folder per experimental pipeline
│   ├── V0_baseline/
│   │   ├── run.sh
│   │   └── README.md
│   ├── V1_data/
│   ├── V2_training/
│   ├── ...
│   └── V7_all_optimised/
│
├── common/                  # Shared dataloaders, utils, energy loggers
│   ├── data.py
│   ├── models.py
│   └── energy.py
│
├── results/                 # Auto-generated CSV + JSON logs
│
├── requirements.txt
└── README.md                # ← you are here
```

*Every* folder inside **`variants/`** is **self-contained**:

* `run.sh` – launches the full pipeline with the variant’s flags  
* `README.md` – documents exactly **which stages are optimised, and how**  
* `config.yaml` *(optional)* – structured hyper-parameters  

---

## Running an Experiment

```bash
# 1. Set up environment (choose one)
conda env create -f environment.yml      # with conda
# or
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download / cache data once
python common/data.py --prepare-all

# 3. Run a pipeline variant, e.g. baseline
cd variants/V0_baseline
bash run.sh        # logs go to ../../results/V0_baseline_YYYY-MM-DD.json
```

All energy, carbon and performance numbers are printed to stdout **and** saved under `results/`.

---

## Energy & Carbon Instrumentation

| Layer | Tool | What it Measures |
|-------|------|------------------|
| GPU power | **NVIDIA-SMI + pynvml** | Instantaneous W, integrated kWh |
| CPU / system | **CodeCarbon** | Process-level energy + g CO₂/kWh |
| Carbon intensity | **ElectricityMap API** *(opt)* | Regional real-time grid mix |
| Job metadata | **MLflow** *(opt)* | Params, metrics, artefacts |

Each stage starts an **energy session**; deltas are aggregated into the final CSV.


https://huggingface.co/docs/transformers/en/perf_train_gpu_one
---

## Result Files

A run produces:

* `results/<VARIANT>/<timestamp>.json`
  ```jsonc
  {
    "variant": "V2_training",
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
* `results/summary.csv` – convenience table across all variants  
* (optional) per-epoch power traces under `results/traces/`


