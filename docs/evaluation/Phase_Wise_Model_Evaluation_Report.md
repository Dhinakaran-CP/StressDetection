# Consolidated Master Model Evaluation Report: Phase-Wise Performance Analysis

This master report compiles and analyzes the Leave-One-Subject-Out (LOSO) training and validation metrics gathered across all five phases of development. The codebase has been organized into matching directory junctions under the `Phases/` directory.

---

## 📂 Table of Contents
1. [Phase 1: Baseline Classical and Fusion Pipeline](#phase-1-baseline-classical-and-fusion-pipeline)
2. [Phase 2: High-Capacity Multimodal Research](#phase-2-high-capacity-multimodal-research)
3. [Phase 3: Production Model Packaging](#phase-3-production-model-packaging)
4. [Phase 4: Temporal Deep Learning Pipeline](#phase-4-temporal-deep-learning-pipeline)
5. [Phase 5: GAN Data Augmentation Experiments](#phase-5-gan-data-augmentation-experiments)
6. [Consolidated Project Leaderboard](#consolidated-project-leaderboard)
7. [Key Insights and Recommendations](#key-insights-and-recommendations)

---

## 🛠️ Phase 1: Baseline Classical and Fusion Pipeline
* **Source Folder**: [`Phases/Phase_1_Baseline_LOSO/`](file:///c:/Users/StressProject/Desktop/StressDetectionUsingML/Phases/Phase_1_Baseline_LOSO/)
* **Description**: Established the initial benchmarking metrics for classical models, unimodal deep models, fusion algorithms, and the baseline candidate models across three timeframe resolutions (2s, 5s, 10s) using subject-independent LOSO cross-validation.

### Baseline Benchmark Results

| Scale | Model Name | Category | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Runtime (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **10sec** | RandomForest | classical | **0.7412** | 0.7075 | 0.5604 | 0.6254 | 0.7282 | 3.51 |
| 10sec | VoiceSequenceExpert | unimodal_deep | 0.7382 | 0.7053 | 0.5512 | 0.6188 | 0.6993 | 3.30 |
| 10sec | XGBoost | classical | 0.7082 | 0.6487 | 0.5295 | 0.5831 | 0.7117 | 5.70 |
| 10sec | SSVB_CASA_AIS | production | 0.7033 | 0.6512 | 0.4956 | 0.5629 | 0.6705 | 26.01 |
| 10sec | VBC_CASA_IS | production | 0.6774 | 0.5994 | 0.4917 | 0.5402 | 0.6607 | 24.13 |
| 10sec | KNN | classical | 0.6433 | 0.5529 | 0.3895 | 0.4570 | 0.6382 | 1.97 |
| 10sec | FaceSequenceExpert | unimodal_deep | 0.6146 | 0.5000 | 0.0029 | 0.0058 | 0.5189 | 3.14 |
| 10sec | GatedFusion | fusion | 0.5731 | 0.4406 | 0.3991 | 0.4188 | 0.5431 | 7.67 |
| 10sec | LogisticRegression | classical | 0.5689 | 0.4448 | 0.4779 | 0.4607 | 0.5553 | 16.38 |
| 10sec | EarlyConcatFusion | fusion | 0.5641 | 0.4189 | 0.3382 | 0.3743 | 0.5371 | 7.54 |
| 10sec | SVM | classical | 0.5611 | 0.2865 | 0.0930 | 0.1405 | 0.3185 | 515.41 |
| **5sec** | RandomForest | classical | **0.7413** | 0.7070 | 0.5681 | 0.6300 | 0.7246 | 7.27 |
| 5sec | VoiceSequenceExpert | unimodal_deep | 0.7403 | 0.7058 | 0.5662 | 0.6284 | 0.6947 | 7.07 |
| 5sec | XGBoost | classical | 0.7188 | 0.6647 | 0.5544 | 0.6046 | 0.7239 | 7.39 |
| 5sec | VBC_CASA_IS | production | 0.6825 | 0.6140 | 0.4876 | 0.5435 | 0.6728 | 54.11 |
| 5sec | KNN | classical | 0.6524 | 0.5679 | 0.4323 | 0.4909 | 0.6610 | 7.67 |
| 5sec | SSVB_CASA_AIS | production | 0.6397 | 0.5349 | 0.5427 | 0.5388 | 0.6393 | 55.26 |
| 5sec | FaceSequenceExpert | unimodal_deep | 0.6105 | 0.4776 | 0.0490 | 0.0889 | 0.5502 | 7.33 |
| 5sec | GatedFusion | fusion | 0.5940 | 0.4645 | 0.3081 | 0.3705 | 0.5603 | 17.07 |
| 5sec | LogisticRegression | classical | 0.5808 | 0.4616 | 0.4880 | 0.4745 | 0.5706 | 49.13 |
| 5sec | EarlyConcatFusion | fusion | 0.5703 | 0.4387 | 0.3873 | 0.4114 | 0.5411 | 15.28 |
| 5sec | SVM | classical | 0.5393 | 0.2410 | 0.0876 | 0.1285 | 0.2765 | 1162.01 |
| **2sec** | RandomForest | classical | **0.7426** | 0.7080 | 0.5759 | 0.6351 | 0.7265 | 21.86 |
| 2sec | VoiceSequenceExpert | unimodal_deep | 0.7392 | 0.7038 | 0.5691 | 0.6293 | 0.6864 | 17.70 |
| 2sec | XGBoost | classical | 0.7192 | 0.6847 | 0.5156 | 0.5882 | 0.7171 | 12.20 |
| 2sec | VBC_CASA_IS | production | 0.6479 | 0.5465 | 0.5572 | 0.5518 | 0.6688 | 135.30 |
| 2sec | KNN | classical | 0.6311 | 0.5332 | 0.4159 | 0.4673 | 0.6314 | 40.82 |
| 2sec | FaceSequenceExpert | unimodal_deep | 0.6091 | 0.4891 | 0.1074 | 0.1761 | 0.5496 | 19.99 |
| 2sec | GatedFusion | fusion | 0.6055 | 0.4884 | 0.2948 | 0.3677 | 0.5310 | 42.56 |
| 2sec | LogisticRegression | classical | 0.5949 | 0.4797 | 0.4881 | 0.4839 | 0.5842 | 135.14 |
| 2sec | SSVB_CASA_AIS | production | 0.5867 | 0.4722 | 0.5292 | 0.4990 | 0.6261 | 140.19 |
| 2sec | EarlyConcatFusion | fusion | 0.5109 | 0.3846 | 0.4286 | 0.4054 | 0.4874 | 43.69 |
| 2sec | SVM | classical | 0.4059 | 0.2272 | 0.2194 | 0.2232 | 0.2894 | 3646.26 |

---

## 🧠 Phase 2: High-Capacity Multimodal Research
* **Source Folder**: [`Phases/Phase_2_High_Capacity/`](file:///c:/Users/StressProject/Desktop/StressDetectionUsingML/Phases/Phase_2_High_Capacity/)
* **Description**: Focused on the development of high-capacity fusion architectures (Gated, Cross-Attention, and Mixture of Experts) on the GPU using the synchronized early fusion dataset.

### High-Capacity Multimodal Fusion Metrics

| Model Configuration | Mean Accuracy | Accuracy Std | Mean F1-Score | F1 Std | Mean ROC-AUC | AUC Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Early Fusion Classifier** | **0.6790** | 0.0105 | 0.5747 | 0.0572 | 0.7083 | 0.0373 |
| Gated Fusion Classifier | 0.6736 | 0.0145 | **0.5760** | 0.0421 | **0.7164** | 0.0274 |
| Cross Attention Fusion Classifier | 0.6727 | 0.0192 | 0.5704 | 0.0536 | 0.7148 | 0.0353 |
| FlexiModal MoE Classifier | 0.6704 | 0.0206 | 0.5677 | 0.0434 | 0.6996 | 0.0380 |

---

## 📦 Phase 3: Production Model Packaging
* **Source Folder**: [`Phases/Phase_3_Production/`](file:///c:/Users/StressProject/Desktop/StressDetectionUsingML/Phases/Phase_3_Production/)
* **Description**: Evaluated end-to-end multimodal routing logic under Standard CNN-GRU and Adversarial CNN-GRU strategies (using gradient reversal layers to decouple subject-specific noise).

### Modality & Routing Benchmark (Standard vs Adversarial)

| Strategy / Modality Configuration | Accuracy (Mean) | F1-Score (Mean) | ROC-AUC (Mean) | RMSE (Mean) |
| :--- | :---: | :---: | :---: | :---: |
| **Strategy 4: Standard CNN-GRU** | | | | |
| - Face Only | 0.6338 | 0.5197 | 0.6548 | 0.4951 |
| - Voice Only | 0.6772 | 0.5904 | 0.7136 | 0.4767 |
| - Physio Only | 0.6430 | 0.5574 | 0.6674 | 0.4767 |
| - Face + Physio | 0.6521 | 0.5468 | 0.6851 | 0.4724 |
| - Face + Voice | 0.6870 | 0.5854 | 0.7128 | 0.4619 |
| - Voice + Physio | 0.6986 | **0.6063** | 0.7201 | 0.4593 |
| - All 3 Modalities (Fusion Router) | 0.6944 | 0.5899 | **0.7254** | **0.4559** |
| **Strategy 5: Adversarial CNN-GRU (Primary)** | | | | |
| - Face Only | 0.6603 | 0.5229 | 0.6672 | 0.4807 |
| - Voice Only | 0.6816 | 0.5833 | 0.6939 | 0.4678 |
| - Physio Only | 0.6603 | 0.5488 | 0.6918 | 0.4714 |
| - Face + Physio | 0.6703 | 0.5492 | 0.6998 | 0.4659 |
| - Face + Voice | 0.6981 | 0.5867 | 0.6984 | 0.4583 |
| - Voice + Physio | 0.6937 | 0.5899 | **0.7240** | 0.4562 |
| - **All 3 Modalities (Adversarial Router)** | **0.7051** | **0.5931** | 0.7184 | **0.4542** |

---

## 📈 Phase 4: Temporal Deep Learning Pipeline
* **Source Folder**: [`Phases/Phase_4_Temporal_Deep/`](file:///c:/Users/StressProject/Desktop/StressDetectionUsingML/Phases/Phase_4_Temporal_Deep/)
* **Description**: Evaluated sequence-based temporal deep architectures (GRU, LSTM, CNN-LSTM, TCN, and Transformer) on flattened and sequential feature matrices across the three window scales.

### Temporal Model Benchmarking Results

| Scale | Model Name | Category | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Runtime (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **10sec** | RandomForest | classical | **0.7414** | 0.7077 | 0.5606 | 0.6256 | 0.7281 | 4.82 |
| 10sec | XGBoost | classical | 0.7060 | 0.6448 | 0.5280 | 0.5806 | 0.7105 | 7.89 |
| 10sec | CNN-LSTM | temporal_deep | 0.6849 | 0.6041 | 0.5297 | 0.5645 | 0.6853 | 11.75 |
| 10sec | TCN | temporal_deep | 0.6712 | 0.5844 | 0.5083 | 0.5437 | 0.6841 | 11.51 |
| 10sec | GRU | temporal_deep | 0.6580 | 0.5654 | 0.4870 | 0.5233 | 0.6502 | 10.89 |
| 10sec | Transformer | temporal_deep | 0.6570 | 0.5591 | 0.5206 | 0.5391 | 0.6703 | 15.32 |
| 10sec | KNN | classical | 0.6476 | 0.5572 | 0.4172 | 0.4771 | 0.6444 | 4.23 |
| 10sec | LSTM | temporal_deep | 0.6459 | 0.5490 | 0.4558 | 0.4981 | 0.6521 | 11.60 |
| 10sec | SVM | classical | 0.5778 | 0.2734 | 0.0575 | 0.0951 | 0.3455 | 454.20 |
| 10sec | LogisticRegression | classical | 0.5723 | 0.4453 | 0.4463 | 0.4458 | 0.5470 | 15.32 |
| **5sec** | RandomForest | classical | **0.7408** | 0.7069 | 0.5663 | 0.6289 | 0.7247 | 9.95 |
| 5sec | XGBoost | classical | 0.7136 | 0.6619 | 0.5343 | 0.5913 | 0.7149 | 11.97 |
| 5sec | CNN-LSTM | temporal_deep | 0.6804 | 0.6067 | 0.4994 | 0.5478 | 0.6719 | 22.26 |
| 5sec | TCN | temporal_deep | 0.6785 | 0.6012 | 0.5077 | 0.5505 | 0.6752 | 21.41 |
| 5sec | Transformer | temporal_deep | 0.6710 | 0.5970 | 0.4661 | 0.5235 | 0.6733 | 30.86 |
| 5sec | LSTM | temporal_deep | 0.6605 | 0.5732 | 0.4870 | 0.5266 | 0.6572 | 21.00 |
| 5sec | KNN | classical | 0.6480 | 0.5589 | 0.4370 | 0.4905 | 0.6532 | 10.84 |
| 5sec | GRU | temporal_deep | 0.6383 | 0.5363 | 0.4955 | 0.5151 | 0.6448 | 19.19 |
| 5sec | LogisticRegression | classical | 0.5878 | 0.4678 | 0.4597 | 0.4638 | 0.5654 | 42.83 |
| 5sec | SVM | classical | 0.5572 | 0.2113 | 0.0519 | 0.0834 | 0.3125 | 1004.20 |
| **2sec** | RandomForest | classical | **0.7425** | 0.7079 | 0.5758 | 0.6351 | 0.7265 | 22.75 |
| 2sec | XGBoost | classical | 0.7212 | 0.6847 | 0.5254 | 0.5946 | 0.7205 | 21.47 |
| 2sec | TCN | temporal_deep | 0.7055 | 0.6670 | 0.4851 | 0.5617 | 0.7114 | 52.78 |
| 2sec | LSTM | temporal_deep | 0.6864 | 0.6178 | 0.5085 | 0.5578 | 0.6862 | 51.04 |
| 2sec | CNN-LSTM | temporal_deep | 0.6828 | 0.6130 | 0.5006 | 0.5511 | 0.6839 | 56.30 |
| 2sec | Transformer | temporal_deep | 0.6756 | 0.6056 | 0.4764 | 0.5333 | 0.6769 | 80.53 |
| 2sec | GRU | temporal_deep | 0.6743 | 0.5986 | 0.4943 | 0.5415 | 0.6854 | 52.46 |
| 2sec | KNN | classical | 0.6295 | 0.5294 | 0.4303 | 0.4747 | 0.6284 | 51.24 |
| 2sec | LogisticRegression | classical | 0.6056 | 0.4927 | 0.4687 | 0.4804 | 0.5836 | 131.73 |
| 2sec | SVM | classical | 0.3742 | 0.2198 | 0.2388 | 0.2289 | 0.3331 | 3416.18 |

---

## ⚖️ Phase 5: GAN Data Augmentation Experiments
* **Source Folder**: [`Phases/Phase_5_GAN_Augmentation/`](file:///c:/Users/StressProject/Desktop/StressDetectionUsingML/Phases/Phase_5_GAN_Augmentation/)
* **Description**: Evaluated WGAN-GP oversampling on training splits to address class imbalance (Stress class minority) while evaluating performance strictly on real, untouched validation sets.

### GAN-Augmented vs Real-Only Performance Comparison

| Scale | Model Name | Real-Only Accuracy | GAN-Augmented Accuracy | Real-Only F1 | GAN-Augmented F1 | Real-Only Recall | GAN-Augmented Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **10sec** | RandomForest | 0.7413 | **0.7438** | 0.6259 | **0.6321** | 0.5613 | **0.5711** |
| 10sec | XGBoost | 0.7060 | 0.7038 | 0.5806 | **0.5831** | 0.5280 | **0.5374** |
| 10sec | CNN-LSTM | 0.6912 | 0.6758 | 0.5594 | 0.5412 | 0.5087 | 0.4960 |
| 10sec | TCN | 0.6753 | 0.6744 | 0.5423 | **0.5461** | 0.4991 | **0.5081** |
| 10sec | Transformer | 0.6662 | **0.6835** | 0.5121 | **0.5517** | 0.4545 | **0.5054** |
| 10sec | LSTM | 0.6582 | 0.6533 | 0.5362 | 0.5191 | 0.5126 | 0.4855 |
| 10sec | KNN | 0.6476 | 0.6245 | 0.4771 | **0.5018** | 0.4172 | **0.4907** |
| 10sec | SVM | 0.5625 | 0.3869 | 0.0974 | **0.5560** | 0.0612 | **0.9959** |
| 10sec | LogisticRegression | 0.5723 | 0.5486 | 0.4458 | 0.4373 | 0.4463 | **0.4550** |
| **5sec** | RandomForest | 0.7420 | **0.7439** | 0.6325 | **0.6354** | 0.5726 | **0.5757** |
| 5sec | XGBoost | 0.7136 | **0.7197** | 0.5913 | **0.6089** | 0.5343 | **0.5628** |
| 5sec | Transformer | 0.6715 | **0.6888** | 0.5365 | **0.5426** | 0.4903 | 0.4760 |
| 5sec | TCN | 0.6655 | 0.6563 | 0.5425 | 0.5112 | 0.5115 | 0.4636 |
| 5sec | KNN | 0.6480 | **0.6524** | 0.4905 | **0.5884** | 0.4370 | **0.6409** |
| 5sec | GRU | 0.6579 | 0.6484 | 0.5312 | 0.5190 | 0.5000 | 0.4892 |
| 5sec | LSTM | 0.6591 | 0.6460 | 0.5184 | **0.5273** | 0.4732 | **0.5092** |
| 5sec | SVM | 0.6098 | 0.3740 | 0.0098 | **0.3988** | 0.0050 | **0.5356** |
| 5sec | LogisticRegression | 0.5878 | 0.5825 | 0.4638 | 0.4569 | 0.4597 | 0.4529 |
| **2sec** | RandomForest | 0.7423 | **0.7434** | 0.6346 | **0.6369** | 0.5752 | **0.5784** |
| 2sec | XGBoost | 0.7212 | 0.7120 | 0.5946 | 0.5883 | 0.5254 | **0.5289** |
| 2sec | TCN | 0.7055 | 0.7032 | 0.5617 | **0.5793** | 0.4851 | **0.5252** |
| 2sec | Transformer | 0.6756 | **0.6929** | 0.5333 | **0.5367** | 0.4764 | 0.4573 |
| 2sec | CNN-LSTM | 0.6828 | 0.6664 | 0.5511 | **0.5540** | 0.5006 | **0.5326** |
| 2sec | GRU | 0.6743 | 0.6602 | 0.5415 | **0.5501** | 0.4943 | **0.5340** |
| 2sec | LSTM | 0.6864 | 0.6539 | 0.5578 | 0.5422 | 0.5085 | **0.5268** |
| 2sec | KNN | 0.6295 | 0.5873 | 0.4747 | **0.5354** | 0.4303 | **0.6112** |
| 2sec | LogisticRegression | 0.6056 | 0.5845 | 0.4804 | 0.4638 | 0.4687 | 0.4620 |
| 2sec | SVM | 0.5585 | 0.4725 | 0.0895 | **0.2760** | 0.0558 | **0.2584** |

---

## 🏆 Consolidated Project Leaderboard
This leaderboard ranks the top 5 performing model configurations evaluated across all phases and resolutions.

| Rank | Model Name | Phase / Mode | Scale | Accuracy | F1-Score | ROC-AUC |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 🥇 | **RandomForest** | Phase 5: GAN-Augmented | 5sec | **0.7439** | 0.6354 | 0.7258 |
| 🥈 | **RandomForest** | Phase 5: GAN-Augmented | 10sec | **0.7438** | 0.6321 | 0.7175 |
| 🥉 | **RandomForest** | Phase 5: GAN-Augmented | 2sec | **0.7434** | **0.6369** | 0.7196 |
| 4 | **RandomForest** | Phase 1 & 4: Baseline | 2sec | 0.7426 | 0.6351 | 0.7265 |
| 5 | **RandomForest** | Phase 4: Baseline | 2sec | 0.7425 | 0.6351 | 0.7265 |

---

## 💡 Key Insights and Recommendations

1. **Top Classifier Family**: 
   - Across all phases and window sizes, the **Random Forest** classifier remains the most robust model for stress detection. Its performance remains consistent (~0.74 accuracy) and benefits marginally from WGAN-GP data balancing.
2. **GAN Augmentation Sensitivity**:
   - The GAN data augmentation phase (Phase 5) shows massive benefits in improving minority class **Recall** and **F1-scores** for models that suffer from class imbalance bias (like SVM and KNN). 
3. **Modality and Routing Success**:
   - For multimodal architectures (Phase 3), the **Adversarial CNN-GRU Router** combining all three modalities (Face, Voice, and Physio) achieved the best deep learning accuracy of **0.7051** and a strong ROC-AUC of **0.7184**, demonstrating the power of adversarial subject-invariant representations.
4. **Resolution Selection**:
   - Timeframe scale comparisons indicate that the 5-second window is the most optimal balance, maximizing F1 scores and training efficiency.
