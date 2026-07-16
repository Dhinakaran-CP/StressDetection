# GAN Experiment Comparison Report

This report compiles performance comparisons across all window scales (2s, 5s, 10s) and model architectures, comparing the Real-Only baseline with the GAN-Augmented training splits.

| Scale | Experiment Mode | Model Name | Category | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Runtime-Seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10sec | gan_augmented | RandomForest | classical | 0.7438 | 0.7078 | 0.5711 | 0.6321 | 0.7175 | 40.6840 |
| 10sec | gan_augmented | XGBoost | classical | 0.7038 | 0.6373 | 0.5374 | 0.5831 | 0.7097 | 31.0008 |
| 10sec | gan_augmented | Transformer | temporal_deep | 0.6835 | 0.6075 | 0.5054 | 0.5517 | 0.6760 | 37.1918 |
| 10sec | gan_augmented | CNN-LSTM | temporal_deep | 0.6758 | 0.5954 | 0.4960 | 0.5412 | 0.6813 | 26.8705 |
| 10sec | gan_augmented | TCN | temporal_deep | 0.6744 | 0.5902 | 0.5081 | 0.5461 | 0.6750 | 31.7450 |
| 10sec | gan_augmented | LSTM | temporal_deep | 0.6533 | 0.5577 | 0.4855 | 0.5191 | 0.6675 | 30.7043 |
| 10sec | gan_augmented | GRU | temporal_deep | 0.6461 | 0.5465 | 0.4806 | 0.5114 | 0.6587 | 26.8899 |
| 10sec | gan_augmented | KNN | classical | 0.6245 | 0.5135 | 0.4907 | 0.5018 | 0.6241 | 27.5803 |
| 10sec | gan_augmented | LogisticRegression | classical | 0.5486 | 0.4208 | 0.4550 | 0.4373 | 0.4994 | 55.8551 |
| 10sec | gan_augmented | SVM | classical | 0.3869 | 0.3856 | 0.9959 | 0.5560 | 0.4164 | 64.8013 |
| 10sec | real_only | RandomForest | classical | 0.7413 | 0.7071 | 0.5613 | 0.6259 | 0.7296 | 5.3070 |
| 10sec | real_only | XGBoost | classical | 0.7060 | 0.6448 | 0.5280 | 0.5806 | 0.7105 | 7.2085 |
| 10sec | real_only | CNN-LSTM | temporal_deep | 0.6912 | 0.6214 | 0.5087 | 0.5594 | 0.7005 | 11.5178 |
| 10sec | real_only | GRU | temporal_deep | 0.6893 | 0.6135 | 0.5241 | 0.5653 | 0.6845 | 12.0671 |
| 10sec | real_only | TCN | temporal_deep | 0.6753 | 0.5937 | 0.4991 | 0.5423 | 0.6868 | 12.0194 |
| 10sec | real_only | Transformer | temporal_deep | 0.6662 | 0.5865 | 0.4545 | 0.5121 | 0.6684 | 19.3879 |
| 10sec | real_only | LSTM | temporal_deep | 0.6582 | 0.5621 | 0.5126 | 0.5362 | 0.6690 | 11.5112 |
| 10sec | real_only | KNN | classical | 0.6476 | 0.5572 | 0.4172 | 0.4771 | 0.6444 | 4.4553 |
| 10sec | real_only | LogisticRegression | classical | 0.5723 | 0.4453 | 0.4463 | 0.4458 | 0.5470 | 17.2076 |
| 10sec | real_only | SVM | classical | 0.5625 | 0.2377 | 0.0612 | 0.0974 | 0.3565 | 29.6016 |
| 2sec | gan_augmented | RandomForest | classical | 0.7434 | 0.7084 | 0.5784 | 0.6369 | 0.7196 | 252.1382 |
| 2sec | gan_augmented | XGBoost | classical | 0.7120 | 0.6626 | 0.5289 | 0.5883 | 0.7109 | 229.7364 |
| 2sec | gan_augmented | TCN | temporal_deep | 0.7032 | 0.6458 | 0.5252 | 0.5793 | 0.7161 | 177.3751 |
| 2sec | gan_augmented | Transformer | temporal_deep | 0.6929 | 0.6495 | 0.4573 | 0.5367 | 0.6985 | 197.4102 |
| 2sec | gan_augmented | CNN-LSTM | temporal_deep | 0.6664 | 0.5773 | 0.5326 | 0.5540 | 0.6763 | 134.5572 |
| 2sec | gan_augmented | GRU | temporal_deep | 0.6602 | 0.5673 | 0.5340 | 0.5501 | 0.6677 | 164.2697 |
| 2sec | gan_augmented | LSTM | temporal_deep | 0.6539 | 0.5585 | 0.5268 | 0.5422 | 0.6632 | 168.4763 |
| 2sec | gan_augmented | KNN | classical | 0.5873 | 0.4764 | 0.6112 | 0.5354 | 0.6097 | 269.6237 |
| 2sec | gan_augmented | LogisticRegression | classical | 0.5845 | 0.4657 | 0.4620 | 0.4638 | 0.5375 | 384.0064 |
| 2sec | gan_augmented | SVM | classical | 0.4725 | 0.2961 | 0.2584 | 0.2760 | 0.3814 | 289.3044 |
| 2sec | real_only | RandomForest | classical | 0.7423 | 0.7078 | 0.5752 | 0.6346 | 0.7330 | 25.1821 |
| 2sec | real_only | XGBoost | classical | 0.7212 | 0.6847 | 0.5254 | 0.5946 | 0.7205 | 22.4652 |
| 2sec | real_only | TCN | temporal_deep | 0.7055 | 0.6670 | 0.4851 | 0.5617 | 0.7114 | 61.2866 |
| 2sec | real_only | LSTM | temporal_deep | 0.6864 | 0.6178 | 0.5085 | 0.5578 | 0.6862 | 61.6975 |
| 2sec | real_only | CNN-LSTM | temporal_deep | 0.6828 | 0.6130 | 0.5006 | 0.5511 | 0.6839 | 66.4601 |
| 2sec | real_only | Transformer | temporal_deep | 0.6756 | 0.6056 | 0.4764 | 0.5333 | 0.6769 | 93.9500 |
| 2sec | real_only | GRU | temporal_deep | 0.6743 | 0.5986 | 0.4943 | 0.5415 | 0.6854 | 62.2546 |
| 2sec | real_only | KNN | classical | 0.6295 | 0.5294 | 0.4303 | 0.4747 | 0.6284 | 53.2783 |
| 2sec | real_only | LogisticRegression | classical | 0.6056 | 0.4927 | 0.4687 | 0.4804 | 0.5836 | 143.5246 |
| 2sec | real_only | SVM | classical | 0.5585 | 0.2265 | 0.0558 | 0.0895 | 0.3647 | 62.0562 |
| 5sec | gan_augmented | RandomForest | classical | 0.7439 | 0.7090 | 0.5757 | 0.6354 | 0.7258 | 89.7115 |
| 5sec | gan_augmented | XGBoost | classical | 0.7197 | 0.6634 | 0.5628 | 0.6089 | 0.7184 | 81.1136 |
| 5sec | gan_augmented | Transformer | temporal_deep | 0.6888 | 0.6308 | 0.4760 | 0.5426 | 0.6910 | 79.8911 |
| 5sec | gan_augmented | CNN-LSTM | temporal_deep | 0.6673 | 0.5795 | 0.5169 | 0.5464 | 0.6664 | 67.6051 |
| 5sec | gan_augmented | TCN | temporal_deep | 0.6563 | 0.5698 | 0.4636 | 0.5112 | 0.6621 | 68.6092 |
| 5sec | gan_augmented | KNN | classical | 0.6524 | 0.5439 | 0.6409 | 0.5884 | 0.6752 | 87.3321 |
| 5sec | gan_augmented | GRU | temporal_deep | 0.6484 | 0.5526 | 0.4892 | 0.5190 | 0.6530 | 61.5060 |
| 5sec | gan_augmented | LSTM | temporal_deep | 0.6460 | 0.5467 | 0.5092 | 0.5273 | 0.6499 | 53.9931 |
| 5sec | gan_augmented | LogisticRegression | classical | 0.5825 | 0.4610 | 0.4529 | 0.4569 | 0.5455 | 127.9022 |
| 5sec | gan_augmented | SVM | classical | 0.3740 | 0.3177 | 0.5356 | 0.3988 | 0.3795 | 116.3581 |
| 5sec | real_only | RandomForest | classical | 0.7420 | 0.7063 | 0.5726 | 0.6325 | 0.7299 | 9.8207 |
| 5sec | real_only | XGBoost | classical | 0.7136 | 0.6619 | 0.5343 | 0.5913 | 0.7149 | 10.7287 |
| 5sec | real_only | Transformer | temporal_deep | 0.6715 | 0.5923 | 0.4903 | 0.5365 | 0.6612 | 36.1804 |
| 5sec | real_only | TCN | temporal_deep | 0.6655 | 0.5774 | 0.5115 | 0.5425 | 0.6757 | 23.2971 |
| 5sec | real_only | LSTM | temporal_deep | 0.6591 | 0.5732 | 0.4732 | 0.5184 | 0.6658 | 24.1414 |
| 5sec | real_only | CNN-LSTM | temporal_deep | 0.6584 | 0.5708 | 0.4798 | 0.5213 | 0.6653 | 27.9905 |
| 5sec | real_only | GRU | temporal_deep | 0.6579 | 0.5666 | 0.5000 | 0.5312 | 0.6614 | 23.6086 |
| 5sec | real_only | KNN | classical | 0.6480 | 0.5589 | 0.4370 | 0.4905 | 0.6532 | 11.9321 |
| 5sec | real_only | SVM | classical | 0.6098 | 0.3029 | 0.0050 | 0.0098 | 0.3301 | 39.7907 |
| 5sec | real_only | LogisticRegression | classical | 0.5878 | 0.4678 | 0.4597 | 0.4638 | 0.5654 | 44.3809 |


*All plots and detailed reports have been categorized into the gan_pipeline_run/outputs/ directory.*