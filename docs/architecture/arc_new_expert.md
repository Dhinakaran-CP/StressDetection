# Expert Submodality Pipeline Plan

## 1. Purpose

This plan defines an expert-based modeling pipeline where each modality is decomposed into meaningful subparts and each subpart is handled by a specialized expert model. The goal is to improve stress detection performance, capture finer-grained patterns, and test whether expert specialization outperforms the current best full-modality models [web:692][web:705][web:704].

## 2. Core idea

Instead of training one model on an entire modality, the agent must:
- split each modality into subparts,
- train one or more expert models for each subpart,
- route samples or segments to the most relevant expert,
- and fuse the expert outputs into one final prediction [web:692][web:694][web:705].

The agent must only keep this approach if it improves subject-independent validation and test performance.

## 3. Supported modalities

The expert system must support:
- face,
- voice,
- physiology.

Each modality must be divided into meaningful subparts based on the available representation.

## 4. Recommended subparts

### 4.1 Face subparts
Possible subparts include:
- upper face,
- lower face,
- eye region,
- mouth region,
- early-window facial dynamics,
- late-window facial dynamics.

### 4.2 Voice subparts
Possible subparts include:
- low-frequency components,
- mid-frequency components,
- high-frequency components,
- voiced regions,
- unvoiced regions,
- early-utterance and late-utterance segments.

### 4.3 Physiology subparts
Possible subparts include:
- rising segments,
- falling segments,
- peak regions,
- baseline segments,
- short-term dynamics,
- long-term dynamics.

The agent may simplify these if the dataset representation does not support all of them.

## 5. Expert architecture

Each expert should be lightweight and task-focused.

Recommended expert types:
- MLP expert for tabular feature subparts.
- GRU or LSTM expert for temporal subparts.
- Small TCN expert for local temporal patterns.
- Small CNN expert for spatial or local feature maps.

Do not use a large expert for every subpart. The goal is specialization, not unnecessary model growth.

## 6. Router design

The agent must implement a router that decides:
- which expert to use,
- how much weight to assign to each expert,
- and when to combine outputs.

Router options:
- rule-based router,
- learned gating network,
- attention-based router,
- confidence-based router.

If the data is limited, start with a simpler router before moving to a learned gating system.

## 7. Training flow

The agent must follow this exact flow:

### Step 1: Load the prepared dataset
Use the extracted 2s, 5s, and 10s feature sets.

### Step 2: Split by subject
Create train, validation, and test splits using subject independence.

### Step 3: Build subpart views
Transform each modality into its expert subparts.

### Step 4: Train experts
Train each expert only on its designated subpart.

### Step 5: Train router
Train the router using only training data.

### Step 6: Fuse outputs
Combine expert outputs into a single prediction.

### Step 7: Validate
Evaluate on the validation split.

### Step 8: Test
Evaluate on the untouched test split.

### Step 9: Save artifacts
Store the models, plots, metrics, and config files.

## 8. Output structure

The agent must create the following folder layout:

```text
outputs/
├── expert_pipeline/
│   ├── 2sec/
│   ├── 5sec/
│   └── 10sec/
```

Inside each window folder:

```text
outputs/expert_pipeline/5sec/
├── face/
│   ├── eye_expert/
│   ├── mouth_expert/
│   └── face_router/
├── voice/
│   ├── low_freq_expert/
│   ├── mid_freq_expert/
│   ├── high_freq_expert/
│   └── voice_router/
├── physio/
│   ├── baseline_expert/
│   ├── peak_expert/
│   ├── trend_expert/
│   └── physio_router/
├── fusion/
├── metrics/
├── plots/
└── reports/
```

## 9. Metrics to compute

For every expert and for the final fused system, compute:
- accuracy,
- precision,
- recall,
- F1-score,
- balanced accuracy,
- ROC-AUC,
- confusion matrix,
- per-class metrics,
- fold mean and standard deviation.

Also record:
- runtime,
- inference time,
- model size,
- parameter count,
- router weight distribution if applicable.

## 10. Required plots

The agent must save:
- confusion matrix,
- ROC curve,
- precision-recall curve,
- fold metric chart,
- router weight distribution,
- expert contribution chart,
- calibration plot if probabilities are available.

## 11. Comparison protocol

The expert pipeline must be compared against:
- the best classical model,
- the best deep temporal model,
- the best multimodal fusion model,
- and the GAN-augmented best model if available.

The expert system is accepted only if it:
- improves held-out subject performance,
- or offers a better recall/F1 balance,
- or improves interpretability without a major performance loss.

## 12. Quality control rules

The agent must reject or revise the expert design if:
- the subparts are too arbitrary,
- experts collapse into similar behavior,
- performance decreases across folds,
- or the router becomes unstable.

The agent must keep only meaningful expert subparts that show measurable benefit.

## 13. Logging and reproducibility

The agent must save:
- data split indices,
- subpart definitions,
- expert configurations,
- router configuration,
- random seeds,
- training logs,
- and version numbers.

## 14. Final decision rule

The agent must not assume expert specialization is automatically better. It must be validated against the best existing models using the same subject-independent protocol. Only retain the expert-based pipeline if it consistently improves the project goals.

## 15. Final instruction

Build and evaluate expert submodels for meaningful subparts of each modality, route them through a controlled fusion mechanism, and compare the full expert system against the best baseline and GAN models on 2s, 5s, and 10s datasets.