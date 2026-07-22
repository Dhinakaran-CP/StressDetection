# 4-Model Architecture Comparison

## SSVB-CASA-AIS (41K params)

```mermaid
%%{init: {"theme": "dark", "flowchart": {"defaultRenderer": "elk"}} }%%
flowchart TD
    %% Stage 1: Raw Data Streams
    subgraph Stage1 ["Stage 1: Raw Data Streams"]
        direction LR
        Eye(["Eye Data"])
        Mouth(["Mouth Data"])
        GFace(["Global Face"])
        Pros(["Prosody"])
        Spec(["Spectral"])
        Qual(["Voice Quality"])
        Card(["Cardiovascular"])
        Mot(["Motion"])
    end

    %% Stage 2: Experts
    subgraph Stage2 ["Stage 2: Sub-Modality Sequence Experts"]
        direction LR
        e_eye["Eye Expert"]
        e_mouth["Mouth Expert"]
        e_gface["Global Face Expert"]
        e_pros["Prosody Expert"]
        e_spec["Spectral Expert"]
        e_qual["Voice Qual Expert"]
        e_card["Cardio Expert"]
        e_mot["Motion Expert"]
    end
    
    Eye --> e_eye
    Mouth --> e_mouth
    GFace --> e_gface
    Pros --> e_pros
    Spec --> e_spec
    Qual --> e_qual
    Card --> e_card
    Mot --> e_mot

    %% Stage 3: Intra-Modality Fusion
    subgraph Stage3 ["Stage 3: Intra-Modality Gated Routing"]
        direction TB
        gate_face{"Face Gate"}
        gate_voice{"Voice Gate"}
        gate_physio{"Physio Gate"}
    end

    e_eye --> gate_face
    e_mouth --> gate_face
    e_gface --> gate_face

    e_pros --> gate_voice
    e_spec --> gate_voice
    e_qual --> gate_voice

    e_card --> gate_physio
    e_mot --> gate_physio

    %% Stage 4: Inter-Modality Cross-Attention
    subgraph Stage4 ["Stage 4: Inter-Modality Cross-Attention"]
        direction TB
        subgraph FaceReinforcement ["Face Reinforcement"]
            attn_fv["Face-Voice Attn"]
            attn_fp["Face-Physio Attn"]
        end
        subgraph VoiceReinforcement ["Voice Reinforcement"]
            attn_vf["Voice-Face Attn"]
            attn_vp["Voice-Physio Attn"]
        end
        subgraph PhysioReinforcement ["Physio Reinforcement"]
            attn_pf["Physio-Face Attn"]
            attn_pv["Physio-Voice Attn"]
        end
    end

    %% Linking Gates to Attn
    gate_face -->|Face Queries| attn_fv
    gate_voice -->|Voice Keys/Values| attn_fv
    
    gate_face -->|Face Queries| attn_fp
    gate_physio -->|Physio Keys/Values| attn_fp
    
    gate_voice -->|Voice Queries| attn_vf
    gate_face -->|Face Keys/Values| attn_vf
    
    gate_voice -->|Voice Queries| attn_vp
    gate_physio -->|Physio Keys/Values| attn_vp
    
    gate_physio -->|Physio Queries| attn_pf
    gate_face -->|Face Keys/Values| attn_pf
    
    gate_physio -->|Physio Queries| attn_pv
    gate_voice -->|Voice Keys/Values| attn_pv

    %% Quality Masking Note
    QualityMasks[("Realtime Quality Masks")] -. "Inhibits bad signals" .-> attn_fv
    QualityMasks -. "Inhibits bad signals" .-> attn_fp
    QualityMasks -. "Inhibits bad signals" .-> attn_vp
    QualityMasks -. "Inhibits bad signals" .-> attn_vf
    QualityMasks -. "Inhibits bad signals" .-> attn_pf
    QualityMasks -. "Inhibits bad signals" .-> attn_pv

    %% Stage 5: Fusion & Global MoE
    subgraph Stage5 ["Stage 5: Global MoE and Pooling"]
        f_re["Projected Face Feats"]
        v_re["Projected Voice Feats"]
        p_re["Projected Physio Feats"]
        global_gate{"Global MoE Router"}
        pooling(("Sequence Mean Pooling"))
    end
    
    attn_fv --> f_re
    attn_fp --> f_re
    attn_vf --> v_re
    attn_vp --> v_re
    attn_pf --> p_re
    attn_pv --> p_re

    f_re --> global_gate
    v_re --> global_gate
    p_re --> global_gate
    
    global_gate --> pooling

    %% Stage 6: Output Heads
    subgraph Stage6 ["Stage 6: Output Heads (GRL & Classification)"]
        stress["Stress Output Head (Binary)"]
        conf["Confidence Score Head"]
        grl(("Gradient Reversal Layer"))
        subj["Adversarial Subject Identity Head"]
    end

    pooling --> stress
    pooling --> conf
    pooling --> grl
    grl -->|Identifies Subject| subj
    
    grl -.-> |Unlearns Identity| pooling
    
    classDef outputHead fill:#2b2d42,stroke:#ef233c,stroke-width:2px,color:#fff;
    class stress,conf,subj outputHead;
```

## CNNBaseline (21K params)

```mermaid
%%{init: {"theme": "dark", "flowchart": {"defaultRenderer": "elk"}} }%%
flowchart TD
    %% Stage 1: Raw Sub-Modality Inputs
    subgraph Stage1 ["Stage 1: Sub-Modality Inputs (9 Streams)"]
        direction LR
        Eye(["Eye Data (30×9)"])
        Mouth(["Mouth Data (30×6)"])
        GFace(["Global Face (30×18)"])
        Pros(["Prosody (30×8)"])
        MFCC(["MFCC (30×13)"])
        Qual(["Voice Quality (30×2)"])
        Card(["Cardio (30×2)"])
        EDA(["EDA (30×3)"])
        Soma(["Somatic (30×8)"])
    end

    %% Stage 2: Stream Concatenation
    subgraph Stage2 ["Stage 2: Early Feature Concatenation"]
        Concat["Channel Concatenation (30 × 69)"]
    end

    Eye --> Concat
    Mouth --> Concat
    GFace --> Concat
    Pros --> Concat
    MFCC --> Concat
    Qual --> Concat
    Card --> Concat
    EDA --> Concat
    Soma --> Concat

    %% Stage 3: Shared Conv1D Backbone Stack
    subgraph Stage3 ["Stage 3: Shared Conv1D Feature Extractor"]
        direction TB
        C1["Conv1D Block 1 (69 → 64, k=3, BN, ReLU)"]
        C2["Conv1D Block 2 (64 → 32, k=3, BN, ReLU)"]
        C3["Conv1D Block 3 (32 → 16, k=3, BN, ReLU)"]
        
        C1 --> C2
        C2 --> C3
    end

    Concat --> C1

    %% Stage 4: Global Temporal Pooling
    subgraph Stage4 ["Stage 4: Global Temporal Pooling"]
        GAP(("Global Average Pooling (16-d)"))
    end

    C3 --> GAP

    %% Stage 5: Output Heads
    subgraph Stage5 ["Stage 5: Output Heads"]
        stress["Stress Output Head (Linear 16 → 2)"]
        conf["Dummy Confidence Head (Constant 1.0)"]
    end

    GAP --> stress
    GAP --> conf

    classDef outputHead fill:#2b2d42,stroke:#ef233c,stroke-width:2px,color:#fff;
    classDef dummyHead fill:#3d405b,stroke:#8d99ae,stroke-width:1.5px,color:#fff;
    class stress outputHead;
    class conf dummyHead;
```

## CNNBaseline+GRL (23K params)

```mermaid
%%{init: {"theme": "dark", "flowchart": {"defaultRenderer": "elk"}} }%%
flowchart TD
    %% Stage 1: Sub-Modality Inputs
    subgraph Stage1 ["Stage 1: Sub-Modality Inputs (9 Streams)"]
        direction LR
        Eye(["Eye Data (30×9)"])
        Mouth(["Mouth Data (30×6)"])
        GFace(["Global Face (30×18)"])
        Pros(["Prosody (30×8)"])
        MFCC(["MFCC (30×13)"])
        Qual(["Voice Quality (30×2)"])
        Card(["Cardio (30×2)"])
        EDA(["EDA (30×3)"])
        Soma(["Somatic (30×8)"])
    end

    %% Stage 2: Stream Concatenation
    subgraph Stage2 ["Stage 2: Early Feature Concatenation"]
        Concat["Channel Concatenation (30 × 69)"]
    end

    Eye --> Concat
    Mouth --> Concat
    GFace --> Concat
    Pros --> Concat
    MFCC --> Concat
    Qual --> Concat
    Card --> Concat
    EDA --> Concat
    Soma --> Concat

    %% Stage 3: Shared Conv1D Backbone Stack
    subgraph Stage3 ["Stage 3: Shared Conv1D Feature Extractor"]
        direction TB
        C1["Conv1D Block 1 (69 → 64, k=3, BN, ReLU)"]
        C2["Conv1D Block 2 (64 → 32, k=3, BN, ReLU)"]
        C3["Conv1D Block 3 (32 → 16, k=3, BN, ReLU)"]
        
        C1 --> C2
        C2 --> C3
    end

    Concat --> C1

    %% Stage 4: Global Temporal Pooling
    subgraph Stage4 ["Stage 4: Global Temporal Pooling"]
        GAP(("Global Average Pooling (16-d)"))
    end

    C3 --> GAP

    %% Stage 5: Output Heads & Adversarial Disentanglement
    subgraph Stage5 ["Stage 5: Output Heads & Adversarial Subject Invariance"]
        stress["Stress Output Head (Linear 16 → 2)"]
        conf["Dummy Confidence Head (Constant 1.0)"]
        grl(("Gradient Reversal Layer (α=0.02)"))
        subj["Adversarial Subject Head (Linear 16 → 91)"]
    end

    GAP --> stress
    GAP --> conf
    GAP --> grl
    grl -->|Identifies Subject| subj
    grl -.->|Unlearns Identity| GAP

    classDef outputHead fill:#2b2d42,stroke:#ef233c,stroke-width:2px,color:#fff;
    classDef dummyHead fill:#3d405b,stroke:#8d99ae,stroke-width:1.5px,color:#fff;
    classDef grlHead fill:#6b705c,stroke:#a5a58d,stroke-width:1.5px,color:#fff;
    
    class stress outputHead;
    class conf dummyHead;
    class subj outputHead;
```

## ConvMoE-MF (8.8K params) — Production Target

```mermaid
%%{init: {"theme": "dark", "flowchart": {"defaultRenderer": "elk"}} }%%
flowchart TD
    %% Stage 1: Sub-Modality Raw Stream Grouping
    subgraph Stage1 ["Stage 1: Raw Modality Grouping (9 Streams)"]
        direction LR
        Eye(["Eye (30×9)"])
        Mouth(["Mouth (30×6)"])
        GFace(["Global Face (30×18)"])
        
        Pros(["Prosody (30×8)"])
        MFCC(["MFCC (30×13)"])
        Qual(["Voice Quality (30×2)"])
        
        Card(["Cardio (30×2)"])
        EDA(["EDA (30×3)"])
        Soma(["Somatic (30×8)"])
    end

    %% Stage 2: Modality-Specific Convolutional Encoders
    subgraph Stage2 ["Stage 2: Modality-Specific Conv1D Encoders"]
        subgraph FaceEnc ["Face Modality Branch (33-d)"]
            f_cat["Concat Face Streams (33-d)"]
            f_conv["2× Conv1D (33 → 16 → 8)"]
            f_gap(("GAP (8-d)"))
            f_cat --> f_conv --> f_gap
        end
        
        subgraph VoiceEnc ["Voice Modality Branch (23-d)"]
            v_cat["Concat Voice Streams (23-d)"]
            v_conv["2× Conv1D (23 → 16 → 8)"]
            v_gap(("GAP (8-d)"))
            v_cat --> v_conv --> v_gap
        end
        
        subgraph PhysioEnc ["Physio Modality Branch (13-d)"]
            p_cat["Concat Physio Streams (13-d)"]
            p_conv["1× Conv1D (13 → 8)"]
            p_gap(("GAP (8-d)"))
            p_cat --> p_conv --> p_gap
        end
    end

    Eye --> f_cat
    Mouth --> f_cat
    GFace --> f_cat

    Pros --> v_cat
    MFCC --> v_cat
    Qual --> v_cat

    Card --> p_cat
    EDA --> p_cat
    Soma --> p_cat

    %% Stage 3: Mixture-of-Experts (MoE) Gated Fusion
    subgraph Stage3 ["Stage 3: Mixture-of-Experts (MoE) Gated Fusion"]
        m_concat["Modality Concat (24-d)"]
        router{"Learned Router (Linear 24→4 + Softmax)"}
        
        subgraph Experts ["Parallel Expert Layers"]
            e1["Expert 1 (Linear 24 → 16 → 8)"]
            e2["Expert 2 (Linear 24 → 16 → 8)"]
            e3["Expert 3 (Linear 24 → 16 → 8)"]
            e4["Expert 4 (Linear 24 → 16 → 8)"]
        end
        
        w_sum(("Weighted Softmax Combination"))
        fused["Fused Latent Embedding (8-d)"]
    end

    f_gap --> m_concat
    v_gap --> m_concat
    p_gap --> m_concat

    m_concat --> router
    m_concat --> e1
    m_concat --> e2
    m_concat --> e3
    m_concat --> e4

    router -->|Routing Weights| w_sum
    e1 --> w_sum
    e2 --> w_sum
    e3 --> w_sum
    e4 --> w_sum

    w_sum --> fused

    %% Stage 4: Multi-Task Output Heads & Dual Adversarial Disentanglement
    subgraph Stage4 ["Stage 4: Multi-Task Heads & Dual GRL Disentanglement"]
        stress["Stress Output Head (Linear 8 → 2)"]
        conf["Confidence Score Head (Linear 8 → 1 + Sigmoid)"]
        
        grl_subj(("Subject GRL (α=0.02)"))
        subj_head["Subject Identity Head (Linear 8 → 91)"]
        
        grl_ds(("Dataset GRL (α=α_ds)"))
        ds_head["Dataset Origin Head (Linear 8 → 3)"]
    end

    fused --> stress
    fused --> conf
    fused --> grl_subj
    grl_subj -->|Identifies Subject| subj_head
    grl_subj -.->|Unlearns Subject| fused

    fused --> grl_ds
    grl_ds -->|Identifies Dataset Origin| ds_head
    grl_ds -.->|Unlearns Dataset Domain| fused

    classDef outputHead fill:#2b2d42,stroke:#ef233c,stroke-width:2px,color:#fff;
    class stress outputHead;
    class conf outputHead;
    class subj_head outputHead;
    class ds_head outputHead;
```

---

## Side-by-Side Summary

| Feature | SSVB-CASA-AIS | CNNBaseline | CNNBaseline+GRL | ConvMoE-MF |
|---------|:-------------:|:-----------:|:---------------:|:----------:|
| Params | ~41K | ~21K | ~23K | **8.8K** |
| Encoder | 9× SequenceExpert (Conv1D+Attn+GRU) | 3× Conv1D shared | 3× Conv1D shared | 3× Conv1D modality-specific |
| Fusion | Intra-modality gating + 6× Cross-Attn + 10-expert MoE | Concat + 3× Conv1D | Concat + 3× Conv1D | 4-expert MoE |
| Subject GRL | ✅ α=0.02 | ❌ | ✅ α=0.02 | ✅ α=0.02 |
| Dataset GRL | ❌ | ❌ | ❌ | ✅ α=swept |
| Confidence Head | ✅ | ❌ (dummy) | ❌ (dummy) | ✅ |
| SSL Pretraining | ✅ Contrastive | ❌ | ❌ | ✅ Contrastive |
| Identity Suppression | Subject only | None | Subject only | Subject + Dataset |
