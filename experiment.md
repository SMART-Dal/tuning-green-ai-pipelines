# 📊 Experiment Matrix

This file is the single source‑of‑truth that links **pipeline variants ↔ tasks ↔ datasets ↔ models ↔ directories**.

> **Legend**  
> ✅ = optimisation enabled | ❌ = baseline / disabled

---

## 1  Tasks & Datasets

| Task ID | Task Description | Dataset | Split Notes |
|---------|-----------------|---------|-------------|
| **T1**  | Software generation – *code‑to‑code translation* | CodexGLUE `icse24-lost-in-translation` | Use authors’ default train / valid / test splits |
| **T2**  | Vulnerability classification | BigVul | Random 80 / 10 / 10 split; dedup before split |


## 2  Models

| Model ID | Base Model | Task(s) | Notes |
|----------|------------|---------|-------|
| **M1**   | `SmalLm (Qwen‑2.5 B)` | T1 | decoder‑only; fine‑tuned for translation |
| **M2**   | `ModernBERT` | T2 | encoder‑only; fine‑tuned for vulnerability label |


## 3  Variant Directory Map

Each variant directory under `variants/` runs *both* tasks in sequence (T1 → T2) with the configuration shown below. Stages not listed inherit the baseline behaviour.

| Variant | Dir | Data Stage | Arch. Stage | Training Stage | System Stage | Inference Stage |
|---------|-----|------------|-------------|----------------|--------------|-----------------|
| **Baseline** | `V0_baseline` | Regular tokenizer • batch 32 | FP32 weights | SGD / AdamW FP32 | Default power | Full layers |
| **Data‑Only** | `V1_data` | **Rust tokenizer** ✅ • batch {16,32,64} sweep | — | — | — | — |
| **Training‑Only** | `V2_training` | — | — | **GradAccum + AMP** ✅ | — | — |
| **Inference‑Only** | `V3_inference` | — | — | — | — | **Layer skipping** ✅ |
| **Architecture‑Only** | `V4_architecture` | — | **INT8 / FP8 quant** ✅ | — | — | — |
| **Data + Training** | `V5_data_training` | Rust tokenizer ✅ | — | GradAccum + AMP ✅ | — | — |
| **Data + Training + Inference** | `V6_data_inference_training` | Rust tokenizer ✅ | — | GradAccum + AMP ✅ | — | Layer skipping ✅ |
| **All‑Optimised** | `V7_all_optimised` | Rust tokenizer ✅ | INT8/FP8 ✅ | GradAccum + AMP ✅ | **Power‑cap + DVFS** ✅ | Layer skipping ✅ |

> **Note:** If you add a new variant, create a sibling folder under `variants/` (**e.g. `V8_arch_system`**) and document its row here.


## 4  Directory Mini‑Checklist

```
variants/
  Vx_<name>/
    run.sh          # entrypoint – MUST emit JSON to ../../results/
    README.md       # per‑variant explanation (keep ≤ 100 lines)
    config.yaml     # (optional) structured hyper‑params
results/
  <variant>_<timestamp>.json
experiment.md       # ← you are here
```

---

### 5  Changelog

| Date | Change |
|------|--------|
| 2025‑04‑29 | Initial matrix & mapping |

---

*Keep this document in sync with the code – it is used by the analysis scripts to auto‑discover experiment metadata.*

