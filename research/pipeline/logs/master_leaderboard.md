# Master Leaderboard

This document summarizes the Leave-One-Subject-Out (LOSO) cross-validation performance of all trained models in the Model Zoo.

## StressID Dataset Leaderboard

| Model Archetype | Accuracy | Precision | Recall | F1-Score | AUC-ROC | F1 Std Dev | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| logistic_regression | 0.6892 | 0.6661 | 0.5377 | 0.5565 | 0.7440 | 0.2225 |  |
| lightgbm | 0.6731 | 0.6363 | 0.5896 | 0.5683 | 0.7372 | 0.2356 |  |
| mlp | 0.6834 | 0.6404 | 0.6061 | 0.5857 | 0.7459 | 0.2303 | ⭐ **Top Performer** |
| temporal | 0.6812 | 0.6238 | 0.6138 | 0.5798 | 0.7480 | 0.2279 |  |


## EmpathicSchool Dataset Leaderboard

| Model Archetype | Accuracy | Precision | Recall | F1-Score | AUC-ROC | F1 Std Dev | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| logistic_regression | 0.8102 | 0.2746 | 0.0571 | 0.0557 | 0.5901 | 0.1488 |  |
| lightgbm | 0.8694 | 0.2703 | 0.1362 | 0.1667 | 0.6000 | 0.2912 | ⭐ **Top Performer** |
| mlp | 0.8369 | 0.3209 | 0.0724 | 0.1060 | 0.5639 | 0.1803 |  |
| temporal | 0.8141 | 0.2856 | 0.0892 | 0.1213 | 0.5440 | 0.2007 |  |

