# Master Model Pipeline Comparison Report

This report compiles performance comparisons across all window scales (2s, 5s, 10s) and model architectures.

| Scale | Model Name | Category | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Runtime-Seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10sec | RandomForest | classical | 0.7414 | 0.7077 | 0.5606 | 0.6256 | 0.7281 | 4.8183 |
| 10sec | XGBoost | classical | 0.7060 | 0.6448 | 0.5280 | 0.5806 | 0.7105 | 7.8901 |
| 10sec | CNN-LSTM | temporal_deep | 0.6849 | 0.6041 | 0.5297 | 0.5645 | 0.6853 | 11.7461 |
| 10sec | TCN | temporal_deep | 0.6712 | 0.5844 | 0.5083 | 0.5437 | 0.6841 | 11.5117 |
| 10sec | GRU | temporal_deep | 0.6580 | 0.5654 | 0.4870 | 0.5233 | 0.6502 | 10.8904 |
| 10sec | Transformer | temporal_deep | 0.6570 | 0.5591 | 0.5206 | 0.5391 | 0.6703 | 15.3192 |
| 10sec | KNN | classical | 0.6476 | 0.5572 | 0.4172 | 0.4771 | 0.6444 | 4.2340 |
| 10sec | LSTM | temporal_deep | 0.6459 | 0.5490 | 0.4558 | 0.4981 | 0.6521 | 11.5994 |
| 10sec | SVM | classical | 0.5778 | 0.2734 | 0.0575 | 0.0951 | 0.3455 | 454.2019 |
| 10sec | LogisticRegression | classical | 0.5723 | 0.4453 | 0.4463 | 0.4458 | 0.5470 | 15.3221 |
| 2sec | RandomForest | classical | 0.7425 | 0.7079 | 0.5758 | 0.6351 | 0.7265 | 22.7533 |
| 2sec | XGBoost | classical | 0.7212 | 0.6847 | 0.5254 | 0.5946 | 0.7205 | 21.4663 |
| 2sec | TCN | temporal_deep | 0.7055 | 0.6670 | 0.4851 | 0.5617 | 0.7114 | 52.7769 |
| 2sec | LSTM | temporal_deep | 0.6864 | 0.6178 | 0.5085 | 0.5578 | 0.6862 | 51.0394 |
| 2sec | CNN-LSTM | temporal_deep | 0.6828 | 0.6130 | 0.5006 | 0.5511 | 0.6839 | 56.2953 |
| 2sec | Transformer | temporal_deep | 0.6756 | 0.6056 | 0.4764 | 0.5333 | 0.6769 | 80.5334 |
| 2sec | GRU | temporal_deep | 0.6743 | 0.5986 | 0.4943 | 0.5415 | 0.6854 | 52.4640 |
| 2sec | KNN | classical | 0.6295 | 0.5294 | 0.4303 | 0.4747 | 0.6284 | 51.2399 |
| 2sec | LogisticRegression | classical | 0.6056 | 0.4927 | 0.4687 | 0.4804 | 0.5836 | 131.7288 |
| 2sec | SVM | classical | 0.3742 | 0.2198 | 0.2388 | 0.2289 | 0.3331 | 3416.1793 |
| 5sec | RandomForest | classical | 0.7408 | 0.7069 | 0.5663 | 0.6289 | 0.7247 | 9.9514 |
| 5sec | XGBoost | classical | 0.7136 | 0.6619 | 0.5343 | 0.5913 | 0.7149 | 11.9724 |
| 5sec | CNN-LSTM | temporal_deep | 0.6804 | 0.6067 | 0.4994 | 0.5478 | 0.6719 | 22.2585 |
| 5sec | TCN | temporal_deep | 0.6785 | 0.6012 | 0.5077 | 0.5505 | 0.6752 | 21.4075 |
| 5sec | Transformer | temporal_deep | 0.6710 | 0.5970 | 0.4661 | 0.5235 | 0.6733 | 30.8608 |
| 5sec | LSTM | temporal_deep | 0.6605 | 0.5732 | 0.4870 | 0.5266 | 0.6572 | 20.9952 |
| 5sec | KNN | classical | 0.6480 | 0.5589 | 0.4370 | 0.4905 | 0.6532 | 10.8353 |
| 5sec | GRU | temporal_deep | 0.6383 | 0.5363 | 0.4955 | 0.5151 | 0.6448 | 19.1910 |
| 5sec | LogisticRegression | classical | 0.5878 | 0.4678 | 0.4597 | 0.4638 | 0.5654 | 42.8319 |
| 5sec | SVM | classical | 0.5572 | 0.2113 | 0.0519 | 0.0834 | 0.3125 | 1004.2047 |


*All plots and detailed reports have been categorized into the outputs/ directory.*