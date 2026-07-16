# Leave-One-Subject-Out (LOSO) Stress Detection Leaderboard

**Validation Mode:** RANDOM_SPLIT
**Feature File Ingested:** C:\Users\StressProject\Desktop\StressDetectionUsingML\loso_evaluation_results\stress_features_fusion_5s.csv
**Total Records:** 27506
**Unique Subjects:** 65

| Model Name | Category | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| XGBoost | Classical | 0.9517 | 0.9604 | 0.9131 | 0.9362 | 0.9918 |
| KNN | Classical | 0.9335 | 0.9292 | 0.8968 | 0.9127 | 0.9784 |
| Random Forest | Classical | 0.8953 | 0.8917 | 0.8308 | 0.8602 | 0.9666 |
| Logistic Regression | Classical | 0.8468 | 0.8161 | 0.7810 | 0.7981 | 0.9106 |
| VBC-CASA-IS | Production | 0.7859 | 0.7069 | 0.7651 | 0.7348 | 0.8784 |
| SSVB-CASA-AIS | Production | 0.7807 | 0.7707 | 0.6186 | 0.6863 | 0.8678 |
| Voice Sequence Expert | Unimodal Expert | 0.7423 | 0.7077 | 0.5713 | 0.6322 | 0.7180 |
| Gated Fusion | Early Fusion | 0.7008 | 0.6202 | 0.5889 | 0.6042 | 0.7694 |
| Early Concat Fusion | Early Fusion | 0.6751 | 0.5851 | 0.5567 | 0.5705 | 0.7193 |
| Face Sequence Expert | Unimodal Expert | 0.6245 | 0.5732 | 0.1237 | 0.2035 | 0.5807 |
| SVM | Classical | 0.5998 | 0.3177 | 0.0279 | 0.0514 | 0.2898 |


*Plots for top performing models have been generated inside the results directory.*