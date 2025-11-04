# APPENDIX: Supplementary Materials

This appendix provides extended materials that complement the main manuscript, including detailed technique descriptions, additional experimental context, extended discussions, and comprehensive references that exceed the journal's constraints.

---

## Table of Contents

1. [Five-Stage Framework Derivation](#1-five-stage-framework-derivation)
2. [Reactive vs. Proactive Optimization Practices](#2-reactive-vs-proactive-optimization-practices)
3. [Orthogonality: Additional Examples](#3-orthogonality-additional-examples)
4. [Variant Lookup Table](#4-variant-lookup-table)
5. [V30 Optimal Configuration Details](#5-v30-optimal-configuration-details)
6. [Model and Dataset Selection Rationale](#6-model-and-dataset-selection-rationale)
7. [Phase-Specific Technique Overview](#7-phase-specific-technique-overview)
8. [Extended Bibliography](#8-extended-bibliography)
9. [Configuration-Driven Replication Guide](#9-configuration-driven-replication-guide)

---

## 1. Five-Stage Framework Derivation

### 1.1 Source Frameworks

Our five-stage framework synthesizes established AI lifecycle models:

**Primary Sources:**
- IEEE Computer Society's LLM Lifecycle [Chourey et al.](https://www.computer.org/publications/tech-news/trends/large-language-model-lifecycle)
  - Stages: Data Preparation → Model Development → Training → Evaluation → Deployment
  - Our mapping: Data Pipeline, Model Architecture, Training Phase, System Design, Inference Phase
- Mlops, llmops, fmops, and beyond [Tantithamthavorn et al.](https://doi.ieeecomputersociety.org/10.1109/MS.2024.3477014)
- Empowering Edge Intelligence: A Comprehensive Survey on On-Device AI Models [Wang et al.](https://doi.org/10.1145/3724420)
- A systematic review of Green AI [Verdecchia et al.](https://wires.onlinelibrary.wiley.com/doi/full/10.1002/widm.1507)
- A review of green artificial intelligence: Towards a more sustainable future [Bolon et al.](https://www.sciencedirect.com/science/article/pii/S0925231224008671)

### 1.2 Stage Definitions

**Data Pipeline:** Data acquisition, preprocessing, storage, and loading
- **Why separate:** Data decisions cascade downstream (e.g., V8's sequence trimming reduces training and inference costs)

**Model Architecture:** Neural network structure, capacity, and design decisions
- **Why separate:** Architectural choices fundamentally constrain all downstream operations

**Training Phase:** Iterative optimization learning model parameters
- **Why separate:** Dominates energy (as seen in our experiments); distinct optimization opportunities

**System Design:** Hardware, software frameworks, and system-level configurations
- **Why separate:** Infrastructure decisions made by different teams; apply across multiple models

**Inference Phase:** Production deployment making predictions
- **Why separate:** Different characteristics (no gradients), represents 80% of production workload

### 1.3 Inter-Stage Relationships

Stages are **interdependent**, not strictly sequential:

```
Data Pipeline ──┐
                ├──> Training Phase ──> Inference Phase
Model Arch ─────┘         ↑                    ↑
                          │                    │
System Design ────────────┴────────────────────┘
```

**Key Dependencies:**
- Data → Training → Inference: Sequential pipeline with cascading effects
- Model Architecture → Training + Inference: Architecture constrains both
- System Design → All Stages: Infrastructure affects all operations

**Example (V26):** Sequence-length trimming (Data) → reduces tokens → smaller attention matrices (Model benefits) → fewer operations (System optimizations more effective) → faster inference

---

## 2. Reactive vs. Proactive Optimization Practices

This section elaborates on the distinction between reactive and proactive energy optimization approaches in AI development, providing concrete examples and best practices.

### 2.1 Reactive Optimization Pattern

**Definition:** Applying optimizations as afterthoughts when problems manifest, rather than considering energy efficiency from the initial design phase.

**Common Scenarios (Not exhaustive but representative examples):**

#### Scenario 1: Out-of-Memory Errors
**Reactive Response:**
1. Developer starts training large model
2. Encounters CUDA out-of-memory error
3. Reactively applies gradient checkpointing to fit model
4. Training completes but with energy penalty (as seen in V1)

**Energy Waste:**
- Initial failed training attempts
- Suboptimal solution (checkpointing) chosen for memory, increasing energy
- No consideration of alternative approaches

**Proactive Alternative:**
1. Analyze memory requirements before training
2. Consider multiple solutions: smaller batch size, mixed precision (FP16), gradient accumulation
3. Choose FP16 (V7): 20% energy reduction, no memory error, faster training
4. No wasted energy from failed attempts

---

#### Scenario 2: Inference Cost Discovery
**Reactive Response:**
1. Deploy model to production
2. Discover unsustainable inference costs after scaling
3. Reactively apply quantization
4. Quantization increases inference latency 24.4% (V3), affecting user experience
5. May need additional infrastructure to maintain throughput

**Energy Waste:**
- Production deployment with inefficient model
- Energy consumed during high-cost inference period
- Potential need for more hardware to compensate for latency

**Proactive Alternative:**
1. Profile inference requirements during development
2. Apply efficient inference techniques from start:
   - Inference engine optimization (V9): 89% latency reduction
   - Layer pruning (V19-V22): 68-85% energy reduction, minimal accuracy impact
3. Deploy with known performance characteristics
4. Avoid costly production surprises

---

#### Scenario 3: Hardware Underutilization
**Reactive Response:**
1. Training runs slowly on available hardware
2. Discover GPU utilization is only 40%
3. Reactively investigate and find data loading bottleneck
4. Add pin_memory and optimize DataLoader
5. Time already wasted on slow training

**Proactive Alternative:**
1. Profile hardware utilization from first experiment
2. Identify bottlenecks early (CPU-GPU transfer, data loading)
3. Apply optimizations (pin_memory, compilation, batching) from start
4. Maximize resource efficiency throughout development

---

### 2.2 Proactive Optimization Pattern

**Definition:** Treating energy efficiency as a first-class design constraint from the earliest development phases, making informed choices that prevent waste.

**Implementation Framework:**

#### Phase 1: Requirements Analysis
**Questions to Ask:**
- What is the target deployment environment? (cloud, edge, mobile)
- What are the latency requirements?
- What accuracy threshold is acceptable?
- What is the energy budget?
- What is the expected scale (requests/day)?

**Energy-Aware Decisions:**
- Choose model size appropriate for deployment target
- Set accuracy-energy tradeoff tolerance
- Plan for efficient inference from start

---

#### Phase 2: Architecture Selection
**Proactive Choices:**
- Start with efficient architectures (smaller models, pruning-friendly designs)
- Consider quantization compatibility
- Design for target hardware

**Example:**
Instead of starting with largest model and pruning later:
1. Estimate task complexity
2. Start with appropriately-sized model
3. Scale up only if performance insufficient
4. Result: Avoid training oversized model

---

#### Phase 3: Development Practices
**Proactive Practices:**
- Enable mixed precision (FP16) from first training run
- Use compilation (Torch Compile) by default
- Profile energy consumption continuously
- Monitor GPU utilization and bottlenecks
- Apply data optimizations early (sequence length, sampling)

**Tools:**
- CodeCarbon for energy monitoring
- PyTorch Profiler for bottleneck identification
- NVIDIA SMI for hardware monitoring
- TensorBoard for performance tracking

---

#### Phase 4: Optimization Strategy
**Proactive Approach:**
1. Identify orthogonal optimization opportunities across stages
2. Apply compatible techniques together (not sequentially)
3. Measure cascading effects
4. Balance accuracy-energy tradeoff explicitly

**Example:** Instead of:
```
Baseline → Add FP16 → Add Compilation → Add Pruning (sequential)
```

Do:
```
Baseline → (FP16 + Compilation + Pruning) together
```
Result: Understand combined effects, avoid redundant experimentation

---

### 2.3 Cultural and Organizational Aspects

**Reactive Culture Indicators:**
- "We'll optimize later if needed"
- Energy monitoring not part of CI/CD
- No energy budgets or tracking
- Optimization team separate from development team
- Focus solely on accuracy metrics

**Proactive Culture Indicators:**
- Energy efficiency in project requirements
- Continuous energy monitoring in development
- Energy budgets alongside accuracy targets
- Shared responsibility for efficiency
- Multi-objective optimization (accuracy + energy)

**Organizational Changes for Proactive Approach:**
1. **Training:** Educate engineers on energy-aware practices
2. **Tools:** Provide energy profiling in standard development environment
3. **Metrics:** Track energy alongside accuracy in leaderboards
4. **Incentives:** Reward efficient implementations
5. **Review:** Include energy considerations in code/model review

---

### 2.4 Cost-Benefit Analysis

**Reactive Approach Costs:**
- Wasted energy from failed experiments
- Technical debt from suboptimal choices
- Higher production costs
- Potential need for re-architecture
- Lost time from troubleshooting

**Proactive Approach Costs:**
- Initial learning and setup time
- Monitoring infrastructure
- More complex initial decision-making
- Potential over-optimization for simple tasks

**When Reactive is Acceptable:**
- Quick prototypes not intended for production
- Research experiments exploring novel ideas
- Very small scale with negligible energy impact

**When Proactive is Essential:**
- Production deployments at scale
- Edge/mobile deployments with strict constraints
- Sustainability-focused organizations
- Cost-sensitive applications

---

### 2.5 Example Transition Strategy

**For Organizations Moving from Reactive to Proactive a sample strategy could be:**

**Phase 1: Awareness**
- Measure current energy consumption
- Identify major energy consumers
- Educate team on Green AI principles

**Phase 2: Infrastructure**
- Deploy energy monitoring tools
- Integrate into development workflow
- Establish baseline metrics

**Phase 3: Practice Changes**
- Add energy to project requirements
- Include efficiency in code review
- Start applying proactive optimizations

**Phase 4: Culture Shift**
- Make energy a default consideration
- Recognize and reward efficiency
- Share success stories and learnings

**Phase 5: Continuous Improvement (Ongoing)**
- Regular efficiency audits
- Update best practices
- Contribute to Green AI community

### 2.6 Key Differences

| Aspect | Reactive | Proactive |
|--------|----------|-----------|
| Timing | After problems occur | Design phase |
| Energy consideration | Afterthought | First-class requirement |
| Approach | Ad-hoc fixes | Systematic planning |
| Monitoring | Added when needed | Continuous from start |
| Result | Suboptimal, wasted energy | Efficient, no waste |

**When Reactive can be Acceptable:**
- Quick prototypes not for production
- Research experiments exploring novel ideas
- Very small scale with negligible impact

**When Proactive can be Essential:**
- Production deployments at scale
- Edge/mobile deployments with constraints
- Sustainability-focused organizations
- Cost-sensitive applications

---

## 3. Orthogonality: Additional Examples

**Reviewer Request:** Provide concrete examples of orthogonal knob combinations.

### 3.1 Definition

Two techniques are **orthogonal** if they:
1. Target different resource constraints/bottlenecks
2. Can be applied simultaneously without interference
3. Create additive or synergistic benefits when combined

### 3.2 Resource Constraint Dimensions

AI systems face multiple independent constraints:
- **Computational intensity** (total FLOPs)
- **Memory capacity** (RAM/VRAM required)
- **Memory bandwidth** (data transfer rate)
- **Numerical precision** (bits per value)
- **Execution efficiency** (kernel overhead, parallelism)
- **Data loading** (I/O throughput)

### 3.3 Example 1: V27 (Compilation + FP16)

**Torch Compile (V11):** Optimizes execution graph
- **Target:** Execution efficiency (kernel fusion, dead code elimination)
- **Mechanism:** Graph-level transformations
- **Dimension:** System Design

**FP16 (V7):** Reduces numerical precision
- **Target:** Arithmetic precision and memory bandwidth
- **Mechanism:** Half-precision operations
- **Dimension:** Training

**Orthogonality Analysis:**
```
Compilation: Optimizes HOW operations execute
FP16:        Changes WHAT operations execute (precision)
→ Independent dimensions → Orthogonal
```

**Results:**
- V11 alone: 38.4% energy reduction
- V7 alone: 20.5% energy reduction
- **V27 combined: 46.0% energy reduction**
- Accuracy: +0.03% F1 (slight improvement)

**Synergy:** Clean combination with no interference, achieving benefits of both.

---

### 3.4 Example 2: V28 (Pruning + FP16 + Compilation)

**Three-way orthogonal combination spanning Model, Training, and System stages.**

**Layer Pruning (V18):**
- Target: Model size (structural sparsity)
- Dimension: Model Architecture

**FP16 (V7):**
- Target: Arithmetic precision
- Dimension: Training

**Torch Compile (V11):**
- Target: Execution efficiency
- Dimension: System Design

**Orthogonality:**
```
Pruning:       Reduces MODEL SIZE
FP16:          Reduces PRECISION per operation
Compilation:   Optimizes EXECUTION of operations
→ Three independent dimensions → Highly orthogonal
```

**Results:**
- Individual: V18 (48.8%), V7 (20.5%), V11 (38.4%)
- **V28 combined: 73.7% energy reduction**
- Accuracy: -0.07% F1 (minimal impact)

**Synergy Score:** 
- Expected (multiplicative): (1-0.488)×(1-0.205)×(1-0.384) = 0.316 → 68.4% reduction
- Actual: 73.7% reduction
- **Superadditive synergy:** Techniques reinforce each other beyond independence

---

### 3.5 Example 3: V26 (Cross-Stage Cascading)

**Demonstrates cascading effects across Data, Model, and System stages.**

**Sequence-Length Trimming (V8):** Data stage
**Pruning (V18):** Model stage
**Compilation (V11):** System stage

**Cascading Mechanism:**
```
Shorter sequences (Data)
    → Fewer tokens to process
    → Smaller attention matrices (benefits Model)
    → Fewer operations (benefits System optimization)
    → Compound savings
```

**Results:**
- Individual: V8 (46.6%), V18 (48.8%), V11 (38.4%)
- **V26 combined: 80.8% energy reduction**
- Accuracy: -0.26% F1

**Insight:** Early-stage optimizations amplify downstream benefits.

---

### 3.6 Counter-Example: V25 (Antagonistic)

**Why some combinations fail:**

**Gradient Checkpointing (V1):**
- Trades compute for memory (recomputation)
- +30.7% energy increase

**FP16 (V7):**
- Already provides memory savings
- Makes checkpointing unnecessary

**Conflict:**
- Both target memory constraint
- Checkpointing's recomputation overhead overwhelms FP16's benefits
- **V25 result: +34.4% energy increase**

**Lesson:** Avoid combining techniques targeting same constraint with conflicting strategies.

---

### 3.7 Practitioner Decision Framework

**Step 1: Identify Current Bottleneck**
- Compute-bound: High GPU utilization (>90%)
- Memory-bound: Low GPU utilization, frequent memory ops
- I/O-bound: GPU idle waiting for data

**Step 2: Select Techniques by Constraint**

| Constraint | Techniques | Manuscript Variants |
|------------|------------|---------------------|
| Compute intensity | Pruning, efficient architectures | V13-V22 |
| Memory capacity | LoRA, quantization | V2, V3 |
| Memory bandwidth | FP16, Flash Attention | V7, V12 |
| Execution overhead | Compilation | V11 |
| Data loading | Sequence trimming, pin memory | V8, V10 |

**Step 3: Verify Compatibility**
- Technical: Do frameworks support combination?
- Orthogonality: Do techniques target different constraints?
- Validation: Test combination experimentally

**Step 4: Measure Synergy**
```python
orthogonality_score = actual_reduction / expected_reduction
# Score > 1: Synergistic
# Score = 1: Independent
# Score < 1: Antagonistic
```

---

## 4. Variant Lookup Table

**Reviewer Request:** Provide full technique names and abbreviation mapping for Table 1.

| Variant | Full Name | Abbreviation | Stages | Energy Δ | F1 Δ | Key Notes |
|---------|-----------|--------------|--------|----------|------|-----------|
| V0 | Baseline (ModernBERT) | - | - | 0% | 0% | Reference |
| V1 | Gradient Checkpointing | GradCkpt | Train | +30.7% | -0.03% | Counterproductive |
| V2 | LoRA PEFT | LoRA | Model, Train, Infer | -84.3% | -4.00% | High energy savings, accuracy cost |
| V3 | INT8 Quantization | Quant | Model, Train, System, Infer | -5.3% | -0.06% | Modest savings, latency increase |
| V4 | Tokenizer Optimization | TokOpt | Data, System, Infer | +0.1% | +0.11% | Negligible impact |
| V5 | Static Power Limit (100W) | PwrLim | System | +26.5% | -0.05% | Counterproductive |
| V6 | Optimizer Tuning (AdamW-8bit) | Opt8 | Train | -4.6% | +0.003% | Modest |
| V7 | FP16 Mixed Precision | FP16 | Model, Train, System | -20.5% | -0.014% | Performance-preserving |
| V8 | Sequence-Length Trimming | Seq-Len | Data, Train, System | -46.6% | -0.38% | Cascading efficiency |
| V9 | Inference Engine (vLLM) | Inf | System, Infer | -8.11% | -0.18% | Dramatic latency reduction (89%) |
| V10 | Data Loader Pin Memory | Pin-Mem | Data, Train | -20.4% | -0.02% | Transfer optimization |
| V11 | Torch Compile | Compile | Train, System, Infer | -38.4% | -0.01% | Performance-preserving |
| V12 | Flash Attention v2 | Attn | Model, Train, Infer | -26.8% | +0.04% | Attention efficiency |
| V13 | Layer Pruning (4 Top) | Prune4T | Model, Train, System, Infer | -15.0% | +0.10% | Shallow pruning |
| V14 | Layer Pruning (4 Bottom) | Prune4B | Model, Train, System, Infer | -15.2% | -0.10% | Shallow pruning |
| V15 | Layer Pruning (8 Top) | Prune8T | Model, Train, System, Infer | -33.8% | +0.072% | Moderate |
| V16 | Layer Pruning (8 Bottom) | Prune8B | Model, Train, System, Infer | -33.8% | -0.11% | Moderate |
| V17 | Layer Pruning (12 Top) | Prune12T | Model, Train, System, Infer | -48.9% | +0.12% | Aggressive |
| V18 | Layer Pruning (12 Bottom) | Prune12B | Model, Train, System, Infer | -48.8% | -0.04% | Aggressive |
| V19 | Layer Pruning (16 Top) | Prune16T | Model, Train, System, Infer | -68.3% | +0.11% | Very aggressive |
| V20 | Layer Pruning (16 Bottom) | Prune16B | Model, Train, System, Infer | -68.4% | +0.05% | Very aggressive |
| V21 | Layer Pruning (20 Top) | Prune20T | Model, Train, System, Infer | -84.6% | -0.08% | Maximum pruning |
| V22 | Layer Pruning (20 Bottom) | Prune20B | Model, Train, System, Infer | -84.5% | +0.09% | Maximum pruning |
| V23 | Attn+Pin-Mem+Opt8+GradAcc | Attn+Pin+Opt+GA | Data, Model, Train, System, Infer | -12.9% | +0.11% | Same-stage combo |
| V24 | Inf+GradCkpt+LoRA+FP16 | Inf+GC+LoRA+FP16 | Model, Train, System, Infer | -86.6% | -4.05% | Extreme efficiency |
| V25 | GradAcc+FP16+Checkpoint | GA+FP16+GC | Model, Train, System | +34.4% | +0.12% | Antagonistic |
| V26 | Prune12B+Seq-Len+Compile | P12B+SL+Comp | Data, Model, Train, System, Infer | -80.8% | -0.26% | Cross-stage synergy |
| V27 | Torch Compile + FP16 | Comp+FP16 | Model, Train, System, Infer | -46.0% | +0.03% | Clean orthogonal |
| V28 | Prune12B + Compile + FP16 | P12B+Comp+FP16 | Model, Train, System, Infer | -73.7% | -0.07% | Three-way synergy |
| V29 | Attn+Pin-Mem+Opt8 | Attn+Pin+Opt | Data, Model, Train, System, Infer | -30.52% | -3.29% | Poor accuracy tradeoff |
| V30 | Optimal Oracle | All techniques | All stages | -94.6% | -4.0% | Maximum demonstration |

**Stage Legend:**
- **Data:** Data Pipeline
- **Model:** Model Architecture
- **Train:** Training Phase
- **System:** System Design
- **Infer:** Inference Phase

**Notes:**
- Energy Δ: Percentage change vs. baseline (V0)
- F1 Δ: Percentage point change vs. baseline (0.994 F1)
- Variants V1, V5, V25 show energy increases (counterproductive)
- Variants V17-V22, V27, V28 represent Pareto-optimal choices

---

## 5. V30 Optimal Configuration Details

**Reviewer Request:** Explain what optimizations V30 includes and how it achieves maximum efficiency.

### 5.1 Component Techniques

V30 combines green-highlighted (most efficient) techniques from Table 1:

1. **Layer Pruning (20 layers)** - from V21/V22
   - Removes 20 of 28 ModernBERT encoder layers
   - Provides ~85% baseline energy reduction

2. **FP16 Mixed Precision** - from V7
   - Half-precision arithmetic
   - Complements pruning with reduced precision

3. **Sequence-Length Trimming (256 tokens)** - from V8
   - Reduces from 512 to 256 tokens
   - Creates cascading savings across stages

4. **Torch Compile** - from V11
   - Graph optimization and kernel fusion
   - Optimizes execution of pruned+FP16 model

5. **Optimized Evaluation (V9)**
   - vLLM-based evaluation after training
   - Dramatic latency reduction

### 5.2 Why This Combination

**Orthogonality Across All Five Stages:**

```
Data (V8):      Reduces input tokens
Model (V21):    Reduces layers/parameters
Training (V7):  Reduces precision
System (V11):   Optimizes execution
Inference (V9): Optimizes deployment
```

Each technique targets a different stage and constraint, enabling compound savings.

### 5.3 Configuration

```yaml
# V30 Configuration (simplified)
model:
  name: "answerdotai/ModernBERT-base"
  num_layers: 8  # Pruned from 28

  pruning:
    enabled: true
    method: "top"
    layers_to_remove: [8-27]  # Remove top 20 layers

training:
  precision: "fp16"
  compile: true

data:
  max_sequence_length: 256  # Reduced from 512

inference:
  # Uses standard Transformers-based evaluation in this repository
  batch_size: 4
```

### 5.4 Performance Breakdown

| Stage | Baseline | V30 | Reduction |
|-------|----------|-----|-----------|
| Training | 0.468 kWh | 0.020 kWh | 95.7% |
| Inference | 0.0444 kWh | 0.0025 kWh | 94.4% |
| **Total** | **0.512 kWh** | **0.027 kWh** | **94.6%** |

**Accuracy:** 0.954 F1 (vs. 0.994 baseline) = 4.0 percentage point drop, preserving 95.95% of performance

### 5.5 When to Use V30

**Appropriate:**
- Demonstrating maximum achievable efficiency
- Large-scale batch processing
- Applications with relaxed accuracy requirements
- Research exploring efficiency limits

**Not Appropriate:**
- Production requiring near-baseline accuracy
- Safety-critical applications
- Tasks where 4% degradation unacceptable

**Better Alternative for Production:** V28 (73.7% energy, -0.07% F1) or V19-V22 (68-85% energy, <0.11% F1)

---

## 6. Model and Dataset Selection Rationale

**Reviewer Request:** Justify ModernBERT and BigVul selection; discuss generalizability.

### 6.1 ModernBERT Selection

**Why ModernBERT:**

1. **Representative Modern Architecture**
   - Released December 2024 (state-of-the-art efficiency improvements)
   - Standard transformer with attention mechanism
   - Foundation of contemporary AI (GPT, Llama, ViT share this architecture)

2. **Appropriate Scale**
   - ~150M parameters (manageable for 30 variants × 3 runs)
   - Large enough for meaningful energy patterns
   - Fits academic hardware budgets

3. **Encoder Architecture**
   - Well-suited for classification (our vulnerability detection task)
   - Bidirectional context understanding
   - Widely deployed in production

4. **Accessibility**
   - Available via HuggingFace
   - Open source, reproducible
   - Well-documented

**Alternatives Considered:**

| Model | Why Not Selected |
|-------|------------------|
| BERT-base | Older (2018), lacks modern optimizations |
| Llama-3-8B | Too large (prohibitive for 90 experiments) |
| GPT-2 | Decoder-only, different task suitability |
| ViT | Different modality, complicates comparison |

### 6.2 BigVul Dataset Selection

**Why BigVul:**

1. **Real-World SE Task**
   - Vulnerability detection in source code
   - Directly relevant to IEEE Software audience
   - Not a generic benchmark

2. **Appropriate Scale**
   - 217,000 training samples (sufficient for energy patterns)
   - 348 real-world projects
   - Manageable for comprehensive experimentation

3. **Established Benchmark**
   - Used in prior vulnerability detection research
   - Known characteristics
   - Enables comparison

4. **Meaningful Complexity**
   - Binary classification (vulnerable/non-vulnerable)
   - Requires semantic understanding
   - Baseline 0.994 F1 (challenging but achievable)

**Alternatives Considered:**

| Dataset | Why Not Selected |
|---------|------------------|
| CodeSearchNet | Too large (~6M samples), excessive cost |
| Devign | Too small (~27K), might not reveal patterns |
| GLUE | Generic NLP, less relevant to SE audience |

### 6.3 Generalizability Discussion

**What Transfers (High Confidence):**

1. **Orthogonality Principle**
   - Combining techniques targeting different constraints creates synergy
   - Fundamental to resource optimization
   - Architecture-agnostic

2. **Cascading Effects**
   - Early-stage optimizations benefit downstream
   - Pipeline structure universal across architectures

3. **Training Dominance**
   - Training consumes 90-92% of energy
   - Consistent across literature and model sizes

4. **Technique Directions**
   - Pruning reduces energy; checkpointing may increase it
   - Fundamental compute-memory tradeoffs

**What May Vary (Medium Confidence):**

1. **Exact Percentages**
   - Our 73.7% (V28) might be 65-80% on different architecture
   - Directional guidance remains valid

2. **Optimal Pruning Depth**
   - Task and architecture dependent
   - Principle of aggressive pruning with minimal accuracy loss transfers

3. **Attention Optimization Magnitude**
   - Depends on sequence length and patterns
   - Direction (beneficial) consistent

**What Doesn't Transfer:**

1. **Absolute Energy Values**
   - 0.512 kWh specific to our hardware/setup
   - Use relative comparisons

2. **Task-Specific Tradeoffs**
   - Acceptable accuracy loss varies by domain
   - Methodology for measuring tradeoff transfers

**Validation Strategy for Other Contexts:**

Practitioners should:
1. Start with high-confidence techniques (FP16, Compilation)
2. Validate medium-confidence techniques (Pruning depth)
3. Measure and compare to our relative findings
4. Expect directional consistency, not exact percentages

### 6.4 Threats to External Validity

We explicitly acknowledge:

- **Single architecture family:** Transformers only (CNNs, RNNs not tested)
- **Single task type:** Classification (generation, detection not tested)
- **Specific scale:** ~150M parameters
- **Academic hardware:** Single GPU (distributed, TPUs not tested)
- **Specific framework:** PyTorch 2.0+ (TensorFlow, JAX not tested)

**Future work should validate:**
- Vision models (ViT, Swin)
- Larger LLMs (7B+ parameters)
- Generation tasks
- Distributed training
- Edge deployment
- Alternative frameworks

---

## 7. Phase-Specific Technique Overview

This section provides comprehensive descriptions of all optimization techniques used in our experiments that were applied in isolation, and some additional techniques that were not considered in the study, grouped by the five pipeline stages. Whenever a variant number (e.g. V8) is mentioned in the title, it refers to the variant that was used in the study. Note: some techniques can be overlapping across stages, in which case we only describe the technique once in the most relevant stage.

### 7.1 Data Pipeline

**Key Techniques:**
- **Sequence-length trimming (V8):** Reduce max tokens
  - Result: 46.6% energy reduction, -0.38% F1
- **Smart sampling:** Select high-quality training samples
- **Data deduplication:** Remove redundant samples
- **Efficient tokenization:** Fast tokenizers (HuggingFace)

**Phase Impact:** <1% of total energy, but creates cascading downstream benefits

---

#### Sequence-Length Trimming (V8)
**Description:** Reduces maximum input sequence length from default (often 512 or 1024 tokens) to task-appropriate shorter lengths, eliminating unnecessary padding.

**Implementation:** 
```yaml
tokenizer:
  max_length: 256  # Reduced from 512
  padding: "max_length"
  truncation: true
```

**Energy Impact Mechanism:** Shorter sequences reduce:
- Attention computation complexity (O(n²) → O((n/2)²) for 50% reduction)
- Memory allocation for attention matrices
- Training batch processing time
- Inference latency

**Trade-offs:** May truncate important context in long-form tasks. Requires task-specific analysis to determine optimal length.

**Relevant Literature:**
- [Wang et al. Linformer: Self-attention with linear complexity](https://arxiv.org/abs/2006.04768)
- [Duman Keles et al. On The Computational Complexity of Self-Attention](https://proceedings.mlr.press/v201/duman-keles23a.html)
- [Child et al. Generating long sequences with sparse transformers](https://arxiv.org/abs/1904.10509)

---

#### Smart Sampling Strategies
**Description:** Techniques for selecting high-quality training samples rather than using entire datasets.

**Common Approaches:**
1. **Diversity-based sampling:** Select samples maximizing feature space coverage
2. **Difficulty-based sampling:** Prioritize samples near decision boundaries
3. **Confidence-based sampling:** Remove redundant high-confidence samples
4. **Gradient-based sampling:** Select samples with highest gradient norms

**Energy Benefits:** Smaller effective dataset size reduces:
- Training epochs needed
- Data loading and preprocessing overhead
- Storage and memory requirements

**Relevant Literature:**
- [Bengio et al. Curriculum learning](https://doi.org/10.1145/1553374.1553380)
- [Settles. Active learning](https://minds.wisconsin.edu/handle/1793/60660)
- [Sener & Savarese. Active learning for convolutional neural networks: A core-set approach](https://arxiv.org/abs/1708.00489)

---

#### Data Deduplication
**Description:** Removes duplicate or near-duplicate samples from training data.

**Techniques:**
- Exact matching (hash-based)
- Fuzzy matching (MinHash, SimHash)
- Semantic similarity (embedding-based)

**Energy Benefits:**
- Reduces redundant computation on similar samples
- Smaller dataset → faster epoch times
- May improve generalization, reducing training iterations needed

**Relevant Literature:**
- [Lee et al. Deduplicating training data makes language models better](https://aclanthology.org/2022.acl-long.577/)
- [Abbas et al. SemDeDup: Data-efficient learning at scale through semantic deduplication](https://arxiv.org/abs/2303.09540)
- [Rae et al. Scaling language models: Methods, analysis & insights from training Gopher](https://arxiv.org/abs/2112.11446)


---

### 7.2 Model Architecture

**Key Techniques:**
- **Layer pruning (V13-V22):** Remove transformer layers
  - Results: 15-85% energy reduction depending on depth
  - Sweet spot: 16-20 layers (68-85% energy, <0.11% F1 impact)
- **Attention optimization (V12):** Flash Attention
  - Result: 26.8% energy reduction, +0.04% F1
- **LoRA (V2):** Parameter-efficient fine-tuning
  - Result: 84.3% energy reduction, -4.00% F1 (high accuracy cost)
- **Quantization (V3):** Reduce precision to INT8
  - Result: 5.3% energy reduction, inference latency increase 24.4%

**Phase Impact:** Structural changes (pruning) most effective

---

#### Layer Pruning (V13-V22)
**Description:** Removes entire transformer layers from the model, reducing parameters and computation.

**Variants Tested:**
- V13/V14: 4 layers removed (top/bottom)
- V15/V16: 8 layers removed (top/bottom)
- V17/V18: 12 layers removed (top/bottom)
- V19/V20: 16 layers removed (top/bottom)
- V21/V22: 20 layers removed (top/bottom)

**Top vs. Bottom Pruning:**
- **Top pruning:** Removes later layers (closer to output)
- **Bottom pruning:** Removes earlier layers (closer to input)
- **Findings:** Minimal performance difference in our experiments, suggesting redundancy across depth

**Implementation:**
```python
# Example: Remove layers 20-23 (top pruning, 4 layers)
model.encoder.layer = model.encoder.layer[:20]
```

**Energy Scaling:** Nearly linear relationship between layers removed and energy savings.

**Trade-offs:** Minimal accuracy impact up to 16-layer pruning, moderate impact at 20 layers.

**Relevant Literature:**
- [Saad et al. An Adaptive Language-Agnostic Pruning Method for Greener Language Models for Code](https://dl.acm.org/doi/abs/10.1145/3715773)
- [Sajjad et al. On the effect of dropping layers of pre-trained transformer models](https://doi.org/10.1016/j.csl.2022.101429)
- [Fan et al. Reducing transformer depth on demand with structured dropout](https://openreview.net/forum?id=SylO2yStDr)
- [Goyal et al. Accurate, large minibatch SGD: Training ImageNet in 1 hour](https://arxiv.org/abs/1706.02677)

---

#### Attention Optimization (V12)
**Description:** Optimizations targeting the self-attention mechanism, the computational bottleneck in transformers.

**Techniques:**
1. **Flash Attention:** Reorders attention operations to maximize memory hierarchy utilization
2. **Sparse Attention:** Only computes attention for subset of token pairs
3. **Linear Attention:** Approximates attention with linear complexity
4. **Multi-Query Attention:** Shares key/value projections across heads

**Implementation in V12:** Flash Attention v2 via:
```python
from torch.nn.functional import scaled_dot_product_attention
# PyTorch 2.0+ automatically uses Flash Attention when available
```

**Energy Benefits:**
- Reduced memory bandwidth (Flash Attention)
- Lower computational complexity (Sparse/Linear Attention)
- Faster training and inference

**Trade-offs:** Flash Attention is algorithmically equivalent (no accuracy loss). Sparse/Linear variants may impact quality.

**Relevant Literature:**
- [Dao et al. FlashAttention: Fast and memory-efficient exact attention with IO-awareness](https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html)
- [Wang et al. Linformer: Self-attention with linear complexity](https://arxiv.org/abs/2006.04768)
- [Kitaev et al. Reformer: The efficient transformer](https://openreview.net/forum?id=rkgNKkHtvB)

---

#### Quantization (V3)
**Description:** Reduces numerical precision of model weights and activations.

**Common Precision Levels:**
- FP32 (baseline): 32-bit floating point
- FP16: 16-bit floating point (half precision)
- INT8: 8-bit integer
- INT4: 4-bit integer (aggressive)

**V3 Implementation:** Post-training INT8 quantization via:
```python
import torch.quantization as quantization
model_quantized = quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

**Energy Benefits:**
- Reduced memory bandwidth
- Smaller model size (4x smaller for INT8 vs FP32)
- Potential hardware acceleration on quantized operations

**Trade-offs:** 
- Modest energy savings (5%) in our experiments
- Increased inference time (24.4%) due to quantization/dequantization overhead
- May degrade accuracy (minimal in our case: -0.06% F1)

**Best Use Cases:** Edge devices with limited memory where model size is primary constraint.

**Relevant Literature:**
- [Jacob et al. Quantization and training of neural networks for efficient integer-arithmetic-only inference](https://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf)
- [Nagel et al. Up or down? Adaptive rounding for post-training quantization](https://proceedings.mlr.press/v119/nagel20a.html)
- [Dettmers et al. GPT3.int8(): 8-bit matrix multiplication for transformers at scale](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c3ba4962c05c49636d4c6206a97e9c8a-Abstract-Conference.html)

---

#### LoRA (Low-Rank Adaptation) (V2)
**Description:** Parameter-efficient fine-tuning method that freezes pretrained weights and trains small low-rank matrices.

**Mathematical Foundation:**
For weight matrix W, instead of updating W directly:
```
W' = W + ΔW
```
LoRA decomposes ΔW as:
```
ΔW = BA
```
where B ∈ ℝ^(d×r) and A ∈ ℝ^(r×k), with r << min(d,k)

**Implementation:**
```yaml
lora_config:
  r: 8  # Rank
  lora_alpha: 16
  target_modules: ["query", "value"]
  lora_dropout: 0.1
```

**Energy Benefits:**
- Dramatically fewer trainable parameters (often <1% of original)
- 85.3% energy reduction in our experiments
- 78.6% time reduction

**Trade-offs:**
- Significant accuracy impact in our case (-4.00% F1)
- May not capture task complexity with very low rank
- Inference requires merging LoRA weights or keeping separate

**Best Use Cases:** When accuracy tradeoffs are acceptable for dramatic resource savings, or when fine-tuning budget is severely constrained.

**Relevant Literature:**
- [Hu et al. LoRA: Low-rank adaptation of large language models](https://openreview.net/forum?id=nZeVKeeFYf9)
- [Dettmers et al. QLoRA: Efficient finetuning of quantized LLMs](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html)
- [Ding et al. Parameter-efficient fine-tuning of large-scale pre-trained language models](https://www.nature.com/articles/s42256-023-00626-4)

---

### 7.3 Training Phase

**Key Techniques:**
- **FP16 mixed precision (V7):** Half-precision training
  - Result: 20.5% energy reduction, -0.014% F1
- **Gradient checkpointing (V1):** Trade compute for memory
  - Result: +30.7% energy (counterproductive in our setup)
- **Optimizer tuning (V6):** Efficient optimizer configurations
  - Result: 4.6% energy reduction

**Phase Impact:** 90-92% of total energy; highest priority for optimization

---

#### Mixed Precision Training (FP16) (V7)
**Description:** Trains model using 16-bit floating point for most operations while maintaining 32-bit for numerical stability where needed.

**Implementation:**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for batch in dataloader:
    with autocast():
        loss = model(batch)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Energy Benefits:**
- 20.5% energy reduction (our experiments)
- 16.6% time reduction
- Reduced memory bandwidth
- Smaller gradient storage

**Numerical Stability Mechanisms:**
- Loss scaling prevents underflow
- Master weights maintained in FP32
- Gradient clipping

**Trade-offs:** Minimal accuracy impact (-0.014% F1 in our experiments). Requires careful hyperparameter tuning in some cases.

**Relevant Literature:**
- [Micikevicius et al. Mixed precision training](https://openreview.net/forum?id=r1gs9JgRZ)
- [Kalamkar et al. A study of BFLOAT16 for deep learning training](https://arxiv.org/abs/1905.12322)

---

#### Gradient Checkpointing (V1)
**Description:** Trades computation for memory by recomputing activations during backward pass instead of storing them.

**Memory-Computation Tradeoff:**
- **Standard training:** Store all activations → high memory, fast backward pass
- **Gradient checkpointing:** Store subset of activations → lower memory, slower backward pass (requires recomputation)

**Implementation:**
```python
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(self, x):
    return checkpoint(self.layer, x)
```

**Energy Impact:**
- **Counterintuitive finding:** +30.7% energy increase in our experiments
- **Reason:** Recomputation overhead exceeds memory access savings on modern GPUs
- **Memory savings:** Significant (enables larger batch sizes or models)

**When Useful:**
- Memory-constrained scenarios where model wouldn't fit otherwise
- When memory savings enable larger batch sizes that improve throughput

**Relevant Literature:**
- [Chen et al. Training deep nets with sublinear memory cost](https://arxiv.org/abs/1604.06174)
- [Rajbhandari et al. ZeRO: Memory optimizations toward training trillion parameter models](https://ieeexplore.ieee.org/abstract/document/9355301)

---

#### Gradient Accumulation
**Description:** Simulates larger batch sizes by accumulating gradients over multiple forward/backward passes before updating weights.

**Implementation:**
```python
accumulation_steps = 4
optimizer.zero_grad()
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Energy Considerations:**
- Allows larger effective batch sizes on memory-constrained hardware
- May improve convergence (fewer optimizer steps)
- Slight overhead from multiple backward passes per update

**Trade-offs:** Can increase or decrease total energy depending on convergence behavior and batch size scaling effects.

**Relevant Literature:**
- [Ott et al. fairseq: A fast, extensible toolkit for sequence modeling](https://aclanthology.org/N19-4009/)
- [Goyal et al. Accurate, large minibatch SGD: Training ImageNet in 1 hour](https://arxiv.org/abs/1706.02677)
- [McCandlish et al. An empirical model of large-batch training](https://arxiv.org/abs/1812.06162)

---

#### Optimizer Tuning (V6)
**Description:** Using more efficient optimizers or tuning optimizer hyperparameters for faster convergence.

**V6 Implementation:** Switched from AdamW to AdamW with:
- Fused implementation (reduced kernel launches)
- Optimized learning rate schedule
- Weight decay tuning

**Common Optimizer Choices:**
- **Adam/AdamW:** Adaptive learning rates, widely used
- **SGD with momentum:** Simpler, sometimes better generalization
- **AdaFactor:** Memory-efficient, good for large models
- **Lion:** Recent optimizer with improved efficiency

**Energy Benefits:**
- 4.6% energy reduction (our experiments)
- Faster convergence → fewer training steps
- Fused operations → reduced overhead

**Relevant Literature:**
- [Chen et al. Symbolic discovery of optimization algorithms](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9a39b4925e35cf447ccba8757137d84f-Abstract-Conference.html)
- [Shazeer & Stern. Adafactor: Adaptive learning rates with sublinear memory cost](https://proceedings.mlr.press/v80/shazeer18a.html)
- [Reddi et al. On the convergence of Adam and beyond](https://openreview.net/forum?id=ryQu7f-RZ)

---

### 7.4 System Design

**Key Techniques:**
- **Compilation (V11):** Torch Compile for kernel fusion
  - Result: 38.4% energy reduction
- **Power limiting (V5):** Static GPU power cap
  - Result: +26.5% energy (counterproductive - avoid)
- **Data loader optimization (V10):** Pin memory
  - Result: 20.4% energy reduction
  

**Phase Impact:** Cross-cutting; affects all operations

---

#### Torch Compile (V11)
**Description:** PyTorch 2.0's compilation feature that optimizes execution graph through graph transformations and kernel fusion.

**Implementation:**
```python
import torch
model = torch.compile(model, mode="reduce-overhead")
```

**Optimization Mechanisms:**
- Kernel fusion (reduces memory bandwidth)
- Dead code elimination
- Constant folding
- Operator fusion and reordering

**Energy Benefits:**
- 38.4% energy reduction
- ~40% time reduction
- No accuracy impact (-0.01% F1)

**Trade-offs:**
- Initial compilation overhead (amortized over training)
- May not support all custom operations
- Debugging can be more complex

**Relevant Literature:**
- [Ansel et al. PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation](https://dl.acm.org/doi/10.1145/3620665.3640366)

---

#### Power Limiting (V5)
**Description:** Hardware-level power cap to limit GPU power consumption.

**Implementation:**
```bash
# Set 100W power limit on NVIDIA GPU
nvidia-smi -pl 100
```

**V5 Findings (100W limit):**
- **Counterintuitive:** +26.5% energy increase
- **Reason:** Slower computation from reduced frequency leads to longer runtime that exceeds power savings
- **Latency impact:** +168% execution time

**Energy Equation:**
```
Energy = Power × Time
```
Reducing power but drastically increasing time yields net energy increase.

**When Useful:**
- Thermal management in constrained environments
- Peak power limiting for grid stability
- When combined with other optimizations that maintain performance

**Better Alternatives:** Dynamic voltage-frequency scaling (DVFS) that adapts to workload rather than static limits.

**Relevant Literature:**
- [Mei et al. A survey and measurement study of GPU DVFS on energy conservation](https://www.sciencedirect.com/science/article/pii/S2352864816300736)
- [Zhao et al. Sustainable Supercomputing for AI: GPU Power Capping at HPC Scale](https://dl.acm.org/doi/10.1145/3620678.3624793)
- [Tang et al. The impact of GPU DVFS on the energy and performance of deep learning: An empirical study](https://dl.acm.org/doi/10.1145/3307772.3328315)

---

#### Data Loader Optimization (Pin Memory) (V10)
**Description:** Pins training data in CPU RAM to enable faster GPU transfers via DMA (Direct Memory Access).

**Implementation:**
```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    pin_memory=True,  # Enable pinned memory
    num_workers=4
)
```

**Energy Benefits:**
- 20.4% energy reduction
- Reduces CPU-GPU transfer overhead
- Enables asynchronous data loading

**Trade-offs:** 
- Increased CPU memory usage (pinned pages)
- Benefit depends on data loading being bottleneck

**Relevant Literature:**
- [Paszke et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html)
- [NVIDIA. CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

---

### 7.5 Inference Phase

**Key Techniques:**
- **Inference engine (V9):** vLLM-based evaluation within training script
  - Result: measured energy/latency using vLLM classification after training
- **Post-training quantization:** INT8 for inference
- **Model distillation:** Train smaller student model
- **KV caching:** Cache attention keys/values

**Phase Impact:** 8-13% of development energy, but 80% of production workload

---

#### Inference Engine Optimization (V9)
**Description:** vLLM is used for optimized inference after training. The model is saved and loaded into vLLM for classification; energy is tracked with CodeCarbon during evaluation.

**Implementation:** Saves the trained model, initializes a vLLM `LLM`, runs classification over test texts, computes F1.

**Optimization Techniques:**
- Graph optimization
- Operator fusion
- Quantization support
- Hardware-specific kernels

**Energy Benefits:**
- 8.11% energy reduction
- 89.2% inference time reduction (dramatic)
- Minimal accuracy impact (-0.18% F1)

**Trade-offs:**
- Conversion overhead (one-time)
- May not support all model architectures
- Debugging more complex

**Production Advantages:**
- Significantly lower latency
- Better throughput
- Cross-platform deployment

**Relevant Literature:**
- [ONNX Runtime Developers. ONNX Runtime: cross-platform, high performance ML inferencing and training accelerator](https://onnxruntime.ai/)
- [NVIDIA. TensorRT: Programmable Inference Accelerator](https://developer.nvidia.com/tensorrt)
- [Intel Corporation. OpenVINO Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)

---

#### Dynamic Batching
**Description:** Groups multiple inference requests into batches dynamically to amortize overhead.

**Energy Benefits:**
- Improved hardware utilization
- Reduced per-sample overhead
- Better throughput → less idle time

**Implementation Considerations:**
- Latency-throughput tradeoff
- Requires request queuing
- Padding overhead for variable-length inputs

**Relevant Literature:**
- [Yu et al. Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- [Agrawal et al. Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369)
- [Crankshaw et al. Clipper: A Low-Latency Online Prediction Serving System](https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/crankshaw)

---

#### KV Cache Optimization
**Description:** Caching key-value pairs in transformer attention to avoid recomputation during autoregressive generation.

**Mechanism:**
During generation, previous tokens' key-value pairs are reused:
```python
# Without KV cache: Recompute all keys/values for all tokens
# With KV cache: Only compute for new token, reuse cached values
```

**Energy Benefits:**
- Dramatic reduction in redundant computation
- Enables longer context generation
- Lower memory bandwidth

**Trade-offs:**
- Increased memory usage during generation
- Cache management overhead

**Relevant Literature:** [Pope et al. Efficiently Scaling Transformer Inference](https://proceedings.mlsys.org/paper_files/paper/2023/file/c4be71ab8d24cdfb45e3d06dbfca2780-Paper-mlsys2023.pdf)

---

#### Model Distillation
**Description:** Training a smaller "student" model to mimic a larger "teacher" model.

**Energy Benefits:**
- Smaller model → lower inference costs
- Maintains much of teacher's performance
- Enables deployment on resource-constrained devices

**Common Approaches:**
- Response-based distillation
- Feature-based distillation
- Relation-based distillation

**Trade-offs:**
- Distillation training overhead (one-time)
- Some performance loss vs. teacher
- Requires access to teacher model

**Relevant Literature:** [Gou et al. Knowledge Distillation: A Survey](https://link.springer.com/article/10.1007/s11263-021-01453-z)

---




## 8. Extended Bibliography

**Reviewer Request:** Provide additional citations beyond 15-reference limit.

### 8.1 Green AI and Energy Efficiency

- [Schwartz et al. Green AI](https://dl.acm.org/doi/10.1145/3381831)
- [Strubell et al. Energy and Policy Considerations for Deep Learning in NLP](https://aclanthology.org/P19-1355/)
- [Patterson et al. Carbon Emissions and Large Neural Network Training](https://arxiv.org/abs/2104.10350)
- [Luccioni et al. Estimating the Carbon Footprint of BLOOM, a 176B Parameter Language Model](https://www.jmlr.org/papers/volume24/23-0069/23-0069.pdf)
- [Wu et al. Sustainable AI: Environmental Implications, Challenges and Opportunities](https://arxiv.org/abs/2111.00364)
- [Dodge et al. Measuring the Carbon Intensity of AI in Cloud Instances](https://dl.acm.org/doi/10.1145/3531146.3533234)
- [Henderson et al. Towards the Systematic Reporting of the Energy and Carbon Footprints of Machine Learning](https://jmlr.org/papers/volume21/20-312/20-312.pdf)

### 8.2 Model Efficiency and Optimization

- [Hinton et al. Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Han et al. Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149)
- [Dao et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html)
- [Hu et al. LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Dettmers et al. QLoRA: Efficient Finetuning of Quantized LLMs](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html)
- [Frantar & Alistarh SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot](https://arxiv.org/abs/2301.00774)

### 8.3 Transformer Architecture

- [Vaswani et al. Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/)
- [Brown et al. Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [Touvron et al. Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- [Kaplan et al. Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

### 8.4 Training Optimization

- [Micikevicius et al. Mixed Precision Training](https://openreview.net/forum?id=r1gs9JgRZ)
- [You et al. Large Batch Optimization for Deep Learning: Training BERT in 76 minutes](https://arxiv.org/abs/1904.00962)
- [Chen et al. Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)
- [Rajbhandari et al. ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://ieeexplore.ieee.org/abstract/document/9355301)

### 8.5 Software Engineering for AI

- [Amershi et al. Software Engineering for Machine Learning: A Case Study](https://ieeexplore.ieee.org/document/8804457)
- [Sculley et al. Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems)
- [Whang et al. Data Collection and Quality Challenges for Deep Learning: A Data-Centric AI Perspective](https://arxiv.org/abs/2112.06409)
- [Zhang et al. Machine Learning Testing: Survey, Landscapes and Horizons](https://ieeexplore.ieee.org/document/9000651)
- [Paleyes et al. Challenges in Deploying Machine Learning: a Survey of Case Studies](https://arxiv.org/abs/2011.09926)

### 8.6 AI Lifecycle and MLOps

- [Kreuzberger et al. Machine Learning Operations (MLOps): Overview, Definition, and Architecture](https://ieeexplore.ieee.org/document/10081336)
- [Renggli et al. A Data Quality-Driven View of MLOps](https://arxiv.org/abs/2102.07750)

### 8.7 Energy Measurement Tools

- [Heguerte et al. How to estimate carbon footprint when training deep learning models? A guide and review](https://arxiv.org/abs/2306.08323)
- [CodeCarbon documentation](https://github.com/mlco2/codecarbon)
- [Rajput et al. Enhancing Energy-Awareness in Deep Learning through Fine-Grained Energy Measurement](https://doi.org/10.1145/3680470)
- [Henderson et al. Towards the Systematic Reporting of the Energy and Carbon Footprints of Machine Learning](https://jmlr.org/papers/volume21/20-312/20-312.pdf)
- [Hähnel et al. Measuring Energy Consumption for Short Code Paths Using RAPL](https://dl.acm.org/doi/10.1145/2425248.2425252)
- [Lacoste et al. Quantifying the Carbon Emissions of Machine Learning](https://arxiv.org/abs/1910.09700)

---

## 9. Configuration-Driven Replication Guide

**Reviewer Request:** Enable practitioners to reproduce experiments and extend to new models/tasks.

### 9.1 Repository Structure

```
greenai-pipeline-empirical-study/
├── variants/                # One folder per experimental pipeline variant
│   ├── v0_baseline/
│   │   ├── config.yaml
│   │   ├── train.py
│   │   └── inference.py
│   ├── v7_f16/
│   ├── v11_torch_compile/
│   ├── v18_layer_pruning_12_bottom/
│   ├── v27_torch_compile_plus_fp16/
│   ├── v28_pruning_plus_torch_compile_plus_fp16/
│   └── ...                 # All other V1–V30 variants
│
├── common/                 # Shared utilities across variants
│   ├── layer_drop.py       # Layer pruning utilities
│   └── generate_configs.py # Helper to generate pruning variants
│
├── analysis_results/       # Consolidated analysis outputs
├── analysis.py             # Analysis scripts
├── energy_modelling.py
├── requirements.txt
└── README.md
```

### 9.2 Quick Start

```bash
# 1) Create a virtualenv and install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Run a baseline variant
python variants/v0_baseline/train.py --cfg variants/v0_baseline/config.yaml --out variants/v0_baseline/results

# 3) Run an optimized variant (example: V28)
python variants/v28_pruning_plus_torch_compile_plus_fp16/train.py \
  --cfg variants/v28_pruning_plus_torch_compile_plus_fp16/config.yaml \
  --out variants/v28_pruning_plus_torch_compile_plus_fp16/results
```

### 9.3 Configuration File Structure

Each variant folder contains a self-contained `config.yaml`. A typical structure (example from a pruning+FP16+compile variant) is:

```yaml
task: vulnerability_detection

model:
  name: answerdotai/ModernBERT-base
  type: sequence_classification
  num_labels: 2

data:
  versions:
    default:
      max_length: 2048
      text_column: func_before
      label_column: vul
    dummy:
      max_length: 128
      text_column: func_before
      label_column: vul

layer_pruning:
  enabled: true
  num_layers: 12
  position: bottom

training:
  versions:
    default:
      num_epochs: 1
      batch_size: 2
      eval_batch_size: 2
      gradient_accumulation_steps: 4
      learning_rate: 5e-5
      warmup_ratio: 0.1
      weight_decay: 0.01
      fp16: true
      gradient_checkpointing: false
```

### 9.4 Adapting to New Models

Create a new variant directory by copying an existing one and editing `config.yaml`:

```bash
cp -r variants/v0_baseline variants/my_roberta_base
```

Edit `variants/my_roberta_base/config.yaml`:

```yaml
model:
  name: roberta-base
  type: sequence_classification
  num_labels: 2
data:
  versions:
    default:
      text_column: func_before
      label_column: vul
```

Run:

```bash
python variants/my_roberta_base/train.py --cfg variants/my_roberta_base/config.yaml --out variants/my_roberta_base/results
```

### 9.5 Adapting to New Tasks

For a new task, duplicate a close variant and modify the dataset section and model head:

```bash
cp -r variants/v0_baseline variants/my_generation_task
```

Edit `variants/my_generation_task/config.yaml` appropriately (e.g., dataset, tokenizer, and model type). Update `train.py` only if the training/evaluation loop must change for the task; otherwise the standard Transformers `Trainer` covers most classification-like cases.

### 9.6 Energy Measurement

**Ensuring Accurate Measurements:**

```python
from codecarbon import EmissionsTracker

# Clean system before measurement
torch.cuda.empty_cache()

# Track energy
tracker = EmissionsTracker(
    project_name="my-experiment",
    measure_power_secs=1,
    save_to_file=True
)

tracker.start()
# ... training code ...
energy = tracker.stop()

print(f"Energy: {energy:.4f} kWh")
```

**Best Practices:**
1. Close unnecessary processes
2. Run experiments 3 times for consistency
3. Measure on clean, stable system
4. Account for idle power if needed

### 9.7 Complete Documentation

Full replication package documentation available at:
https://github.com/SMART-Dal/tuning-green-ai-pipelines

Includes:
- Detailed setup instructions
- Complete API documentation
- Example configurations for common scenarios
- Troubleshooting guide
- Community contributions

---

**End of Appendix**

---

**Note to Reviewers:** This appendix directly addresses all items referenced in our response letter and mentioned in the manuscript. All sections correspond to specific reviewer requests or manuscript citations to the replication package appendix. The full codebase with additional implementation details is available in our public GitHub repository.
