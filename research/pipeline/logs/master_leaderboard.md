# Master Leaderboard

This document summarizes the Leave-One-Subject-Out (LOSO) cross-validation performance of all trained models in the Model Zoo.

## StressID Dataset Leaderboard

| Model Archetype | Accuracy | Precision | Recall | F1-Score | AUC-ROC | F1 Std Dev | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| logistic_regression | 0.6892 | 0.6661 | 0.5377 | 0.5565 | 0.7440 | 0.2225 |  |
| lightgbm | 0.6731 | 0.6363 | 0.5896 | 0.5683 | 0.7372 | 0.2356 |  |
| mlp | 0.6863 | 0.6388 | 0.6037 | 0.5879 | 0.7440 | 0.2307 | ⭐ **Top Performer** |
| temporal | 0.6784 | 0.6298 | 0.5983 | 0.5759 | 0.7441 | 0.2266 |  |


## EmpathicSchool Dataset Leaderboard

| Model Archetype | Accuracy | Precision | Recall | F1-Score | AUC-ROC | F1 Std Dev | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| logistic_regression | 0.8102 | 0.2746 | 0.0571 | 0.0557 | 0.5901 | 0.1488 |  |
| lightgbm | 0.8694 | 0.2703 | 0.1362 | 0.1667 | 0.6000 | 0.2912 | ⭐ **Top Performer** |
| mlp | 0.8385 | 0.3397 | 0.0611 | 0.0943 | 0.5397 | 0.1514 |  |
| temporal | 0.8085 | 0.2544 | 0.0841 | 0.1144 | 0.5389 | 0.1902 |  |

