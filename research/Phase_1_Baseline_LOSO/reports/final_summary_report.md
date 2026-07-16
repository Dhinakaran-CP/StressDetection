# Master Model Benchmarking final Summary Report

This report compiles performance comparisons across all window scales (2s, 5s, 10s) and model architectures.

| Scale | Model Name | Category | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Runtime-Seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10sec | RandomForest | classical | 0.7412 | 0.7075 | 0.5604 | 0.6254 | 0.7282 | 3.5123 |
| 10sec | VoiceSequenceExpert | unimodal_deep | 0.7382 | 0.7053 | 0.5512 | 0.6188 | 0.6993 | 3.3047 |
| 10sec | XGBoost | classical | 0.7082 | 0.6487 | 0.5295 | 0.5831 | 0.7117 | 5.7010 |
| 10sec | SSVB_CASA_AIS | production | 0.7033 | 0.6512 | 0.4956 | 0.5629 | 0.6705 | 26.0115 |
| 10sec | VBC_CASA_IS | production | 0.6774 | 0.5994 | 0.4917 | 0.5402 | 0.6607 | 24.1327 |
| 10sec | KNN | classical | 0.6433 | 0.5529 | 0.3895 | 0.4570 | 0.6382 | 1.9705 |
| 10sec | FaceSequenceExpert | unimodal_deep | 0.6146 | 0.5000 | 0.0029 | 0.0058 | 0.5189 | 3.1351 |
| 10sec | GatedFusion | fusion | 0.5731 | 0.4406 | 0.3991 | 0.4188 | 0.5431 | 7.6739 |
| 10sec | LogisticRegression | classical | 0.5689 | 0.4448 | 0.4779 | 0.4607 | 0.5553 | 16.3778 |
| 10sec | EarlyConcatFusion | fusion | 0.5641 | 0.4189 | 0.3382 | 0.3743 | 0.5371 | 7.5367 |
| 10sec | SVM | classical | 0.5611 | 0.2865 | 0.0930 | 0.1405 | 0.3185 | 515.4145 |
| 2sec | RandomForest | classical | 0.7426 | 0.7080 | 0.5759 | 0.6351 | 0.7265 | 21.8643 |
| 2sec | VoiceSequenceExpert | unimodal_deep | 0.7392 | 0.7038 | 0.5691 | 0.6293 | 0.6864 | 17.6974 |
| 2sec | XGBoost | classical | 0.7192 | 0.6847 | 0.5156 | 0.5882 | 0.7171 | 12.2006 |
| 2sec | VBC_CASA_IS | production | 0.6479 | 0.5465 | 0.5572 | 0.5518 | 0.6688 | 135.3046 |
| 2sec | KNN | classical | 0.6311 | 0.5332 | 0.4159 | 0.4673 | 0.6314 | 40.8151 |
| 2sec | FaceSequenceExpert | unimodal_deep | 0.6091 | 0.4891 | 0.1074 | 0.1761 | 0.5496 | 19.9950 |
| 2sec | GatedFusion | fusion | 0.6055 | 0.4884 | 0.2948 | 0.3677 | 0.5310 | 42.5566 |
| 2sec | LogisticRegression | classical | 0.5949 | 0.4797 | 0.4881 | 0.4839 | 0.5842 | 135.1402 |
| 2sec | SSVB_CASA_AIS | production | 0.5867 | 0.4722 | 0.5292 | 0.4990 | 0.6261 | 140.1853 |
| 2sec | EarlyConcatFusion | fusion | 0.5109 | 0.3846 | 0.4286 | 0.4054 | 0.4874 | 43.6930 |
| 2sec | SVM | classical | 0.4059 | 0.2272 | 0.2194 | 0.2232 | 0.2894 | 3646.2569 |
| 5sec | RandomForest | classical | 0.7413 | 0.7070 | 0.5681 | 0.6300 | 0.7246 | 7.2748 |
| 5sec | VoiceSequenceExpert | unimodal_deep | 0.7403 | 0.7058 | 0.5662 | 0.6284 | 0.6947 | 7.0698 |
| 5sec | XGBoost | classical | 0.7188 | 0.6647 | 0.5544 | 0.6046 | 0.7239 | 7.3923 |
| 5sec | VBC_CASA_IS | production | 0.6825 | 0.6140 | 0.4876 | 0.5435 | 0.6728 | 54.1147 |
| 5sec | KNN | classical | 0.6524 | 0.5679 | 0.4323 | 0.4909 | 0.6610 | 7.6712 |
| 5sec | SSVB_CASA_AIS | production | 0.6397 | 0.5349 | 0.5427 | 0.5388 | 0.6393 | 55.2644 |
| 5sec | FaceSequenceExpert | unimodal_deep | 0.6105 | 0.4776 | 0.0490 | 0.0889 | 0.5502 | 7.3263 |
| 5sec | GatedFusion | fusion | 0.5940 | 0.4645 | 0.3081 | 0.3705 | 0.5603 | 17.0733 |
| 5sec | LogisticRegression | classical | 0.5808 | 0.4616 | 0.4880 | 0.4745 | 0.5706 | 49.1294 |
| 5sec | EarlyConcatFusion | fusion | 0.5703 | 0.4387 | 0.3873 | 0.4114 | 0.5411 | 15.2793 |
| 5sec | SVM | classical | 0.5393 | 0.2410 | 0.0876 | 0.1285 | 0.2765 | 1162.0131 |


*All plots and detailed reports have been categorized into the outputs/ directory.*