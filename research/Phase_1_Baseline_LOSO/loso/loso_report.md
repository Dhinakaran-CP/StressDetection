# Leave-One-Subject-Out (LOSO) Stress Detection Leaderboard

**Validation Mode:** FULL_LOSO
**Feature File Ingested:** C:\Users\StressProject\Desktop\StressDetectionUsingML\loso_evaluation_results\stress_features_fusion_5s.csv
**Total Records:** 27506
**Unique Subjects:** 65

| Model Name | Category | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Random Forest | Classical | 0.7423 | 0.7080 | 0.5708 | 0.6321 | 0.7137 |
| Voice Sequence Expert | Unimodal Expert | 0.7423 | 0.7087 | 0.5694 | 0.6315 | 0.6526 |
| XGBoost | Classical | 0.7116 | 0.6494 | 0.5569 | 0.5996 | 0.7077 |
| SSVB-CASA-AIS | Production | 0.6711 | 0.5894 | 0.5001 | 0.5411 | 0.6706 |
| KNN | Classical | 0.6445 | 0.5529 | 0.4345 | 0.4866 | 0.6489 |
| VBC-CASA-IS | Production | 0.6300 | 0.5256 | 0.4701 | 0.4963 | 0.6256 |
| Face Sequence Expert | Unimodal Expert | 0.6108 | 0.4922 | 0.1208 | 0.1939 | 0.5124 |
| Logistic Regression | Classical | 0.5973 | 0.4815 | 0.5002 | 0.4907 | 0.5773 |
| Early Concat Fusion | Early Fusion | 0.5786 | 0.4530 | 0.4191 | 0.4354 | 0.5730 |
| Gated Fusion | Early Fusion | 0.5357 | 0.3896 | 0.3484 | 0.3679 | 0.5053 |
| SVM | Classical | 0.4946 | 0.2421 | 0.1424 | 0.1793 | 0.3008 |


*Plots for top performing models have been generated inside the results directory.*