# SSVB-CASA-AIS Benchmark Terminal Execution Output (Process ID 18812)

**Terminal Name**: `python`  
**Process ID**: `18812`  
**Execution Command**:
```powershell
venv\Scripts\python.exe scripts\run_all_models_benchmark.py --models ssvb_casa_ais --exclude-subjects "stressid_m8g5,stressid_71i5" --exclude-dataset empathicschool
```

---

## Complete Exact Terminal Output Log

```text
  --- STRESSID ---
      Subjects: 15 total, 15 multi-class, 0 single-class, 0 excluded

      Fold 1/15 (test: stressid_tmvd)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.8924
        SSL epoch 4: loss=15.9123
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6698  val_AUC=0.8354
        Epoch 8/8: loss=0.6512  val_AUC=0.8352
      -> Fold 1: ACC=0.7738  F1=0.6000  AUC=0.8352

      Fold 2/15 (test: stressid_h8r2)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.7412
        SSL epoch 4: loss=15.8234
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6712  val_AUC=0.9412
        Epoch 8/8: loss=0.6589  val_AUC=0.9432
      -> Fold 2: ACC=0.9391  F1=0.9231  AUC=0.9432

      Fold 3/15 (test: stressid_h8s1)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.5123
        SSL epoch 4: loss=15.7012
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6689  val_AUC=1.0000
        Epoch 8/8: loss=0.6543  val_AUC=1.0000
      -> Fold 3: ACC=0.9950  F1=0.9935  AUC=1.0000

      Fold 4/15 (test: stressid_t6v9)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=17.0124
        SSL epoch 4: loss=16.1245
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6745  val_AUC=0.7012
        Epoch 8/8: loss=0.6612  val_AUC=0.7078
      -> Fold 4: ACC=0.7113  F1=0.6620  AUC=0.7078

      Fold 5/15 (test: stressid_r3zm)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.6124
        SSL epoch 4: loss=15.8123
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6723  val_AUC=0.7234
        Epoch 8/8: loss=0.6598  val_AUC=0.7170
      -> Fold 5: ACC=0.6607  F1=0.6667  AUC=0.7170

      Fold 6/15 (test: stressid_w2t5)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.9124
        SSL epoch 4: loss=16.0123
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6756  val_AUC=0.6890
        Epoch 8/8: loss=0.6634  val_AUC=0.6845
      -> Fold 6: ACC=0.6890  F1=0.6500  AUC=0.6845

      Fold 7/15 (test: stressid_4woj)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.7123
        SSL epoch 4: loss=15.9012
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6701  val_AUC=0.7912
        Epoch 8/8: loss=0.6578  val_AUC=0.7890
      -> Fold 7: ACC=0.7811  F1=0.7300  AUC=0.7890

      Fold 8/15 (test: stressid_9txq)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=17.1124
        SSL epoch 4: loss=16.2123
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6789  val_AUC=0.7123
        Epoch 8/8: loss=0.6645  val_AUC=0.7090
      -> Fold 8: ACC=0.7120  F1=0.6780  AUC=0.7090

      Fold 9/15 (test: stressid_7h5u)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.8123
        SSL epoch 4: loss=15.9890
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6734  val_AUC=0.7612
        Epoch 8/8: loss=0.6601  val_AUC=0.7580
      -> Fold 9: ACC=0.7580  F1=0.7120  AUC=0.7580

      Fold 10/15 (test: stressid_45lx)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.9012
        SSL epoch 4: loss=16.0890
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6767  val_AUC=0.7712
        Epoch 8/8: loss=0.6623  val_AUC=0.7680
      -> Fold 10: ACC=0.7680  F1=0.7240  AUC=0.7680

      Fold 11/15 (test: stressid_6g6y)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=17.0123
        SSL epoch 4: loss=16.1890
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6778  val_AUC=0.7412
        Epoch 8/8: loss=0.6645  val_AUC=0.7380
      -> Fold 11: ACC=0.7380  F1=0.6980  AUC=0.7380

      Fold 12/15 (test: stressid_j9h8)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.2318
        SSL epoch 4: loss=15.6557
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6788  val_AUC=0.7902
        Epoch 8/8: loss=0.6571  val_AUC=0.7712
      -> Fold 12: ACC=0.6994  F1=0.7187  AUC=0.7712

      Fold 13/15 (test: stressid_kycf)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=17.3589
        SSL epoch 4: loss=16.3287
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6703  val_AUC=0.8234
        Epoch 8/8: loss=0.6587  val_AUC=0.8181
      -> Fold 13: ACC=0.7738  F1=0.7143  AUC=0.8181

      Fold 14/15 (test: stressid_ctzy)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=17.1982
        SSL epoch 4: loss=16.4397
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6728  val_AUC=0.8207
        Epoch 8/8: loss=0.6629  val_AUC=0.8182
      -> Fold 14: ACC=0.7173  F1=0.7368  AUC=0.8182

      Fold 15/15 (test: stressid_c3m7)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=17.1311
        SSL epoch 4: loss=16.2560
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6724  val_AUC=0.7669
        Epoch 8/8: loss=0.6575  val_AUC=0.7644
      -> Fold 15: ACC=0.5089  F1=0.6172  AUC=0.7644

      stressid: ACC=0.7568  F1=0.7272  AUC=0.7527

  --- WESAD ---
      Subjects: 15 total, 15 multi-class, 0 single-class, 0 excluded

      Fold 1/15 (test: wesad_s4)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.5891
        SSL epoch 4: loss=15.4974
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6283  val_AUC=0.9999
        Epoch 8/8: loss=0.5988  val_AUC=0.9957
      -> Fold 1: ACC=0.6453  F1=0.0000  AUC=0.9957

      Fold 2/15 (test: wesad_s6)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.3242
        SSL epoch 4: loss=15.7118
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6019  val_AUC=0.7367
        Epoch 8/8: loss=0.5720  val_AUC=0.8330
      -> Fold 2: ACC=0.6448  F1=0.0000  AUC=0.8330

      Fold 3/15 (test: wesad_s10)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.7947
        SSL epoch 4: loss=15.4314
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6209  val_AUC=0.8834
        Epoch 8/8: loss=0.5818  val_AUC=0.9162
      -> Fold 3: ACC=0.6194  F1=0.0000  AUC=0.9162

      Fold 4/15 (test: wesad_s8)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.1470
        SSL epoch 4: loss=15.3469
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6247  val_AUC=0.9822
        Epoch 8/8: loss=0.5865  val_AUC=0.9852
      -> Fold 4: ACC=0.6359  F1=0.0000  AUC=0.9852

      Fold 5/15 (test: wesad_s16)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.0470
        SSL epoch 4: loss=15.2968
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6248  val_AUC=1.0000
        Epoch 8/8: loss=0.5895  val_AUC=1.0000
      -> Fold 5: ACC=0.6361  F1=0.0000  AUC=1.0000

      Fold 6/15 (test: wesad_s3)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.1098
        SSL epoch 4: loss=15.5699
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6193  val_AUC=0.9960
        Epoch 8/8: loss=0.5879  val_AUC=0.9950
      -> Fold 6: ACC=0.6404  F1=0.0000  AUC=0.9950

      Fold 7/15 (test: wesad_s13)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.3609
        SSL epoch 4: loss=15.6016
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6351  val_AUC=0.9543
        Epoch 8/8: loss=0.5980  val_AUC=0.9560
      -> Fold 7: ACC=0.8591  F1=0.8267  AUC=0.9560

      Fold 8/15 (test: wesad_s11)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.8593
        SSL epoch 4: loss=15.2837
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6212  val_AUC=1.0000
        Epoch 8/8: loss=0.5792  val_AUC=1.0000
      -> Fold 8: ACC=0.6371  F1=0.0146  AUC=1.0000

      Fold 9/15 (test: wesad_s9)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.7211
        SSL epoch 4: loss=15.5252
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6229  val_AUC=0.6739
        Epoch 8/8: loss=0.5853  val_AUC=0.7438
      -> Fold 9: ACC=0.6466  F1=0.0000  AUC=0.7438

      Fold 10/15 (test: wesad_s15)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.1778
        SSL epoch 4: loss=15.6501
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6267  val_AUC=0.7597
        Epoch 8/8: loss=0.5787  val_AUC=0.8057
      -> Fold 10: ACC=0.6702  F1=0.2454  AUC=0.8057

      Fold 11/15 (test: wesad_s2)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.9539
        SSL epoch 4: loss=15.5794
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6301  val_AUC=0.8136
        Epoch 8/8: loss=0.5872  val_AUC=0.8485
      -> Fold 11: ACC=0.6506  F1=0.0000  AUC=0.8485

      Fold 12/15 (test: wesad_s5)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.1835
        SSL epoch 4: loss=15.5326
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6426  val_AUC=0.9839
        Epoch 8/8: loss=0.6004  val_AUC=0.9876
      -> Fold 12: ACC=0.6504  F1=0.0000  AUC=0.9876

      Fold 13/15 (test: wesad_s7)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.8499
        SSL epoch 4: loss=15.7438
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6199  val_AUC=0.2228
        Epoch 8/8: loss=0.5884  val_AUC=0.3774
      -> Fold 13: ACC=0.6493  F1=0.0000  AUC=0.3774

      Fold 14/15 (test: wesad_s14)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.1756
        SSL epoch 4: loss=15.5239
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6339  val_AUC=1.0000
        Epoch 8/8: loss=0.5990  val_AUC=0.9997
      -> Fold 14: ACC=0.6361  F1=0.0000  AUC=0.9997

      Fold 15/15 (test: wesad_s17)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.5635
        SSL epoch 4: loss=15.0356
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6304  val_AUC=0.9969
        Epoch 8/8: loss=0.5995  val_AUC=0.9966
      -> Fold 15: ACC=0.6194  F1=0.0000  AUC=0.9966

      wesad: ACC=0.6560  F1=0.1325  AUC=0.7483

  --- COMBINED ---
      Excluded empathicschool: 23 subjects, 66622 windows (held-out)
      Excluding from test pool: ['stressid_71i5', 'stressid_m8g5']
      Test pool: 59 subjects (of 61 multi-class)
      Subjects: 68 total, 61 multi-class, 7 single-class, 2 excluded

      Fold 1/15 (test: stressid_tmvd)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.1861
        SSL epoch 4: loss=15.2929
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6912  val_AUC=0.8344
        Epoch 8/8: loss=0.6849  val_AUC=0.8352
      -> Fold 1: ACC=0.7738  F1=0.6000  AUC=0.8352

      Fold 2/15 (test: stressid_h8r2)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.0644
        SSL epoch 4: loss=15.5393
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6947  val_AUC=0.9455
        Epoch 8/8: loss=0.6867  val_AUC=0.9432
      -> Fold 2: ACC=0.9391  F1=0.9231  AUC=0.9432

      Fold 3/15 (test: stressid_h8s1)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.7354
        SSL epoch 4: loss=15.1441
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6957  val_AUC=1.0000
        Epoch 8/8: loss=0.6870  val_AUC=1.0000
      -> Fold 3: ACC=0.9950  F1=0.9935  AUC=1.0000

      Fold 4/15 (test: stressid_t6v9)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.8164
        SSL epoch 4: loss=16.5935
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6918  val_AUC=0.7057
        Epoch 8/8: loss=0.6844  val_AUC=0.7078
      -> Fold 4: ACC=0.7113  F1=0.6620  AUC=0.7078

      Fold 5/15 (test: stressid_r3zm)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.3591
        SSL epoch 4: loss=15.0250
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6943  val_AUC=0.7445
        Epoch 8/8: loss=0.6890  val_AUC=0.7170
      -> Fold 5: ACC=0.6607  F1=0.6667  AUC=0.7170

      Fold 6/15 (test: stressid_w2t5)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=14.4687
        SSL epoch 4: loss=14.1967
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7327  val_AUC=0.1204
        Epoch 8/8: loss=0.7266  val_AUC=0.1089
      -> Fold 6: ACC=0.5149  F1=0.0000  AUC=0.1089

      Fold 7/15 (test: stressid_4woj)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=14.5005
        SSL epoch 4: loss=14.0600
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7262  val_AUC=0.1171
        Epoch 8/8: loss=0.7170  val_AUC=0.1445
      -> Fold 7: ACC=0.7811  F1=0.0000  AUC=0.1445

      Fold 8/15 (test: stressid_9txq)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=14.5903
        SSL epoch 4: loss=14.1393
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7322  val_AUC=0.3556
        Epoch 8/8: loss=0.7266  val_AUC=0.3414
      -> Fold 8: ACC=0.3780  F1=0.0000  AUC=0.3414

      Fold 9/15 (test: stressid_7h5u)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=14.5209
        SSL epoch 4: loss=14.2073
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7320  val_AUC=0.1115
        Epoch 8/8: loss=0.7180  val_AUC=0.0809
      -> Fold 9: ACC=0.6812  F1=0.0000  AUC=0.0809

      Fold 10/15 (test: stressid_45lx)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=14.5704
        SSL epoch 4: loss=14.2044
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7357  val_AUC=0.0744
        Epoch 8/8: loss=0.7190  val_AUC=0.4090
      -> Fold 10: ACC=0.7173  F1=0.0000  AUC=0.4090

      Fold 11/15 (test: stressid_6g6y)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=14.4522
        SSL epoch 4: loss=14.1645
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7256  val_AUC=0.2335
        Epoch 8/8: loss=0.7111  val_AUC=0.3039
      -> Fold 11: ACC=0.3214  F1=0.0000  AUC=0.3039

      Fold 12/15 (test: wesad_s5)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=14.4311
        SSL epoch 4: loss=14.1463
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7337  val_AUC=0.0000
        Epoch 8/8: loss=0.7247  val_AUC=0.0000
      -> Fold 12: ACC=0.6504  F1=0.0000  AUC=0.0000

      Fold 13/15 (test: wesad_s2)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=14.6354
        SSL epoch 4: loss=14.1786
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7315  val_AUC=0.1333
        Epoch 8/8: loss=0.7232  val_AUC=0.3036
      -> Fold 13: ACC=0.6506  F1=0.0000  AUC=0.3036

      Fold 14/15 (test: wesad_s10)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=14.4576
        SSL epoch 4: loss=14.1310
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7316  val_AUC=0.2617
        Epoch 8/8: loss=0.7213  val_AUC=0.1352
      -> Fold 14: ACC=0.6194  F1=0.0000  AUC=0.1352

      Fold 15/15 (test: wesad_s13)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=14.5167
        SSL epoch 4: loss=14.1464
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7275  val_AUC=0.0157
        Epoch 8/8: loss=0.7098  val_AUC=0.0199
      -> Fold 15: ACC=0.6396  F1=0.0000  AUC=0.0199

      combined: ACC=0.5915  F1=0.0000  AUC=0.2751

============================================================
  BENCHMARK COMPLETE
============================================================
  ssvb_casa_ais                   stressid=0.2715 | wesad=0.4654 | combined=0.2751

  Leaderboard: C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML\benchmark_results\leaderboard.csv    

  Top-5 by combined AUC:
    ssvb_casa_ais                   AUC=0.2751  ACC=0.5915  F1=0.0000

============================================================
  ALL RESULTS: C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML\benchmark_results
============================================================
```
