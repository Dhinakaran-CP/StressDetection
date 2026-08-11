# Benchmark Terminal Execution Output (3 Models: CNNBaseline+GRL, CNNBaseline, ConvMoE-MF)

```text
PS C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML> venv\Scripts\python.exe scripts\run_all_models_benchmark.py --models cnn_baseline_grl --exclude-subjects "stressid_m8g5,stressid_71i5" --exclude-dataset empathicschool
Device: cuda
Device: cuda

============================================================
  UNIFIED RESEARCH MODEL BENCHMARK
============================================================
  Device: cuda
  Models: 1
    - cnn_baseline_grl               (phase3)  CNNBaseline+GRL: 1D-CNN with adversarial subject head
============================================================
  stressid: 16974 windows, 53 subjects
  wesad: 5517 windows, 15 subjects
  combined: 89113 windows, 91 subjects
  Excluding subjects from test pool: ['stressid_m8g5', 'stressid_71i5']

============================================================
  MODEL: cnn_baseline_grl (phase3)
  CNNBaseline+GRL: 1D-CNN with adversarial subject head
============================================================

  --- STRESSID ---
      Excluding from test pool: ['stressid_71i5', 'stressid_m8g5']
      Test pool: 44 subjects (of 46 multi-class)
      Subjects: 53 total, 46 multi-class, 7 single-class, 2 excluded

      Fold 1/15 (test: stressid_tmvd)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9487  val_AUC=0.7939
        Epoch 8/8: loss=0.9250  val_AUC=0.8281
      -> Fold 1: ACC=0.5893  F1=0.4524  AUC=0.8281

      Fold 2/15 (test: stressid_h8r2)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9524  val_AUC=0.9508
        Epoch 8/8: loss=0.9364  val_AUC=0.9417
      -> Fold 2: ACC=0.9359  F1=0.9187  AUC=0.9417

      Fold 3/15 (test: stressid_h8s1)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9527  val_AUC=1.0000
        Epoch 8/8: loss=0.9319  val_AUC=1.0000
      -> Fold 3: ACC=0.8458  F1=0.7480  AUC=1.0000

      Fold 4/15 (test: stressid_t6v9)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9469  val_AUC=0.6656
        Epoch 8/8: loss=0.9246  val_AUC=0.6454
      -> Fold 4: ACC=0.7113  F1=0.6620  AUC=0.6454

      Fold 5/15 (test: stressid_r3zm)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9509  val_AUC=0.6956
        Epoch 8/8: loss=0.9304  val_AUC=0.7026
      -> Fold 5: ACC=0.6607  F1=0.6667  AUC=0.7026

      Fold 6/15 (test: stressid_w2t5)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9494  val_AUC=0.6728
        Epoch 8/8: loss=0.9240  val_AUC=0.6721
      -> Fold 6: ACC=0.7976  F1=0.7703  AUC=0.6721

      Fold 7/15 (test: stressid_4woj)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9513  val_AUC=0.8838
        Epoch 8/8: loss=0.9299  val_AUC=0.9244
      -> Fold 7: ACC=0.8358  F1=0.7273  AUC=0.9244

      Fold 8/15 (test: stressid_9txq)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9497  val_AUC=0.6720
        Epoch 8/8: loss=0.9302  val_AUC=0.6985
      -> Fold 8: ACC=0.6607  F1=0.6667  AUC=0.6985

      Fold 9/15 (test: stressid_7h5u)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9511  val_AUC=0.9316
        Epoch 8/8: loss=0.9306  val_AUC=0.9545
      -> Fold 9: ACC=0.8557  F1=0.8054  AUC=0.9545

      Fold 10/15 (test: stressid_45lx)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9481  val_AUC=0.8947
        Epoch 8/8: loss=0.9258  val_AUC=0.8923
      -> Fold 10: ACC=0.8869  F1=0.8333  AUC=0.8923

      Fold 11/15 (test: stressid_6g6y)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9540  val_AUC=0.6427
        Epoch 8/8: loss=0.9301  val_AUC=0.6702
      -> Fold 11: ACC=0.6458  F1=0.6827  AUC=0.6702

      Fold 12/15 (test: stressid_j9h8)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9398  val_AUC=0.6035
        Epoch 8/8: loss=0.9181  val_AUC=0.4253
      -> Fold 12: ACC=0.3363  F1=0.0591  AUC=0.4253

      Fold 13/15 (test: stressid_kycf)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9503  val_AUC=0.7366
        Epoch 8/8: loss=0.9276  val_AUC=0.7324
      -> Fold 13: ACC=0.7738  F1=0.7143  AUC=0.7324

      Fold 14/15 (test: stressid_ctzy)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9492  val_AUC=0.8369
        Epoch 8/8: loss=0.9252  val_AUC=0.8713
      -> Fold 14: ACC=0.7173  F1=0.7368  AUC=0.8713

      Fold 15/15 (test: stressid_c3m7)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9420  val_AUC=0.7301
        Epoch 8/8: loss=0.9215  val_AUC=0.7051
      -> Fold 15: ACC=0.5119  F1=0.6204  AUC=0.7051

      stressid: ACC=0.7084  F1=0.6675  AUC=0.6762

  --- WESAD ---
      Subjects: 15 total, 15 multi-class, 0 single-class, 0 excluded

      Fold 1/15 (test: wesad_s4)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6327  val_AUC=0.9609
        Epoch 8/8: loss=0.5792  val_AUC=0.9799
      -> Fold 1: ACC=0.7570  F1=0.4790  AUC=0.9799

      Fold 2/15 (test: wesad_s6)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.5749  val_AUC=0.8890
        Epoch 8/8: loss=0.5121  val_AUC=0.8757
      -> Fold 2: ACC=0.7432  F1=0.5155  AUC=0.8757

      Fold 3/15 (test: wesad_s10)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.5896  val_AUC=0.8436
        Epoch 8/8: loss=0.5411  val_AUC=0.8628
      -> Fold 3: ACC=0.8110  F1=0.7410  AUC=0.8628

      Fold 4/15 (test: wesad_s8)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.5867  val_AUC=0.8546
        Epoch 8/8: loss=0.5359  val_AUC=0.8857
      -> Fold 4: ACC=0.7663  F1=0.7171  AUC=0.8857

      Fold 5/15 (test: wesad_s16)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6032  val_AUC=1.0000
        Epoch 8/8: loss=0.5552  val_AUC=1.0000
      -> Fold 5: ACC=0.9838  F1=0.9783  AUC=1.0000

      Fold 6/15 (test: wesad_s3)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.5908  val_AUC=0.9944
        Epoch 8/8: loss=0.5371  val_AUC=0.9940
      -> Fold 6: ACC=0.8989  F1=0.8407  AUC=0.9940

      Fold 7/15 (test: wesad_s13)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.5882  val_AUC=0.9871
        Epoch 8/8: loss=0.5392  val_AUC=0.9846
      -> Fold 7: ACC=0.6694  F1=0.6856  AUC=0.9846

      Fold 8/15 (test: wesad_s11)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.5910  val_AUC=1.0000
        Epoch 8/8: loss=0.5415  val_AUC=1.0000
      -> Fold 8: ACC=0.8683  F1=0.8474  AUC=1.0000

      Fold 9/15 (test: wesad_s9)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.5797  val_AUC=0.9000
        Epoch 8/8: loss=0.5346  val_AUC=0.9288
      -> Fold 9: ACC=0.7945  F1=0.6193  AUC=0.9288

      Fold 10/15 (test: wesad_s15)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.5981  val_AUC=0.9192
        Epoch 8/8: loss=0.5530  val_AUC=0.9546
      -> Fold 10: ACC=0.8901  F1=0.8313  AUC=0.9546

      Fold 11/15 (test: wesad_s2)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.5441  val_AUC=0.6091
        Epoch 8/8: loss=0.4961  val_AUC=0.4422
      -> Fold 11: ACC=0.6506  F1=0.0889  AUC=0.4422

      Fold 12/15 (test: wesad_s5)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.5818  val_AUC=0.9963
        Epoch 8/8: loss=0.5418  val_AUC=0.9968
      -> Fold 12: ACC=0.9485  F1=0.9218  AUC=0.9968

      Fold 13/15 (test: wesad_s7)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6161  val_AUC=0.9984
        Epoch 8/8: loss=0.5656  val_AUC=1.0000
      -> Fold 13: ACC=0.3589  F1=0.5224  AUC=1.0000

      Fold 14/15 (test: wesad_s14)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6286  val_AUC=1.0000
        Epoch 8/8: loss=0.5856  val_AUC=0.9998
      -> Fold 14: ACC=0.6577  F1=0.6801  AUC=0.9998

      Fold 15/15 (test: wesad_s17)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6050  val_AUC=0.9916
        Epoch 8/8: loss=0.5526  val_AUC=0.9856
      -> Fold 15: ACC=0.9396  F1=0.9256  AUC=0.9856

      wesad: ACC=0.7836  F1=0.7135  AUC=0.8479

  --- COMBINED ---
      Excluded empathicschool: 23 subjects, 66622 windows (held-out)
      Excluding from test pool: ['stressid_71i5', 'stressid_m8g5']
      Test pool: 59 subjects (of 61 multi-class)
      Subjects: 68 total, 61 multi-class, 7 single-class, 2 excluded

      Fold 1/15 (test: stressid_tmvd)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9602  val_AUC=0.8081
        Epoch 8/8: loss=0.9320  val_AUC=0.8278
      -> Fold 1: ACC=0.7321  F1=0.5588  AUC=0.8278

      Fold 2/15 (test: stressid_h8r2)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9636  val_AUC=0.9420
        Epoch 8/8: loss=0.9358  val_AUC=0.9472
      -> Fold 2: ACC=0.9391  F1=0.9231  AUC=0.9472

      Fold 3/15 (test: stressid_h8s1)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9650  val_AUC=1.0000
        Epoch 8/8: loss=0.9396  val_AUC=1.0000
      -> Fold 3: ACC=0.9353  F1=0.9078  AUC=1.0000

      Fold 4/15 (test: stressid_t6v9)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9600  val_AUC=0.6875
        Epoch 8/8: loss=0.9301  val_AUC=0.6513
      -> Fold 4: ACC=0.7083  F1=0.6573  AUC=0.6513

      Fold 5/15 (test: stressid_r3zm)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9592  val_AUC=0.6951
        Epoch 8/8: loss=0.9323  val_AUC=0.7286
      -> Fold 5: ACC=0.6607  F1=0.6667  AUC=0.7286

      Fold 6/15 (test: stressid_w2t5)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9628  val_AUC=0.6731
        Epoch 8/8: loss=0.9307  val_AUC=0.6970
      -> Fold 6: ACC=0.7232  F1=0.7103  AUC=0.6970

      Fold 7/15 (test: stressid_4woj)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9698  val_AUC=0.8943
        Epoch 8/8: loss=0.9362  val_AUC=0.8971
      -> Fold 7: ACC=0.8358  F1=0.7273  AUC=0.8971

      Fold 8/15 (test: stressid_9txq)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9598  val_AUC=0.6880
        Epoch 8/8: loss=0.9355  val_AUC=0.6765
      -> Fold 8: ACC=0.6012  F1=0.5786  AUC=0.6765

      Fold 9/15 (test: stressid_7h5u)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9707  val_AUC=0.9228
        Epoch 8/8: loss=0.9396  val_AUC=0.9466
      -> Fold 9: ACC=0.8691  F1=0.8186  AUC=0.9466

      Fold 10/15 (test: stressid_45lx)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9632  val_AUC=0.9226
        Epoch 8/8: loss=0.9311  val_AUC=0.8973
      -> Fold 10: ACC=0.8958  F1=0.8430  AUC=0.8973

      Fold 11/15 (test: stressid_6g6y)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9635  val_AUC=0.6791
        Epoch 8/8: loss=0.9314  val_AUC=0.6699
      -> Fold 11: ACC=0.6071  F1=0.6526  AUC=0.6699

      Fold 12/15 (test: wesad_s5)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9594  val_AUC=1.0000
        Epoch 8/8: loss=0.9307  val_AUC=0.9999
      -> Fold 12: ACC=0.8916  F1=0.8165  AUC=0.9999

      Fold 13/15 (test: wesad_s2)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9607  val_AUC=0.9492
        Epoch 8/8: loss=0.9331  val_AUC=0.9449
      -> Fold 13: ACC=0.6562  F1=0.0320  AUC=0.9449

      Fold 14/15 (test: wesad_s10)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9542  val_AUC=0.5313
        Epoch 8/8: loss=0.9255  val_AUC=0.5820
      -> Fold 14: ACC=0.3701  F1=0.5402  AUC=0.5820

      Fold 15/15 (test: wesad_s13)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9651  val_AUC=0.9177
        Epoch 8/8: loss=0.9337  val_AUC=0.9371
      -> Fold 15: ACC=0.4905  F1=0.5822  AUC=0.9371

      combined: ACC=0.7127  F1=0.6623  AUC=0.7196

============================================================
  BENCHMARK COMPLETE
============================================================
  cnn_baseline_grl                stressid=0.6762 | wesad=0.8479 | combined=0.7196

  Leaderboard: C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML\benchmark_results\leaderboard.csv    

  Top-5 by combined AUC:
    cnn_baseline_grl                AUC=0.7196  ACC=0.7127  F1=0.6623

============================================================
  ALL RESULTS: C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML\benchmark_results
============================================================
PS C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML> venv\Scripts\python.exe scripts\run_all_models_benchmark.py --models cnn_baseline --exclude-subjects "stressid_m8g5,stressid_71i5" --exclude-dataset empathicschool
Device: cuda
Device: cuda

============================================================
  UNIFIED RESEARCH MODEL BENCHMARK
============================================================
  Device: cuda
  Models: 1
    - cnn_baseline                   (phase3)  CNNBaseline: plain 1D-CNN, no GRL
============================================================
  stressid: 16974 windows, 53 subjects
  wesad: 5517 windows, 15 subjects
  combined: 89113 windows, 91 subjects
  Excluding subjects from test pool: ['stressid_m8g5', 'stressid_71i5']

============================================================
  MODEL: cnn_baseline (phase3)
  CNNBaseline: plain 1D-CNN, no GRL
============================================================

  --- STRESSID ---
      Excluding from test pool: ['stressid_71i5', 'stressid_m8g5']
      Test pool: 44 subjects (of 46 multi-class)
      Subjects: 53 total, 46 multi-class, 7 single-class, 2 excluded

      Fold 1/15 (test: stressid_tmvd)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9530  val_AUC=0.8273
        Epoch 8/8: loss=0.9348  val_AUC=0.8193
      -> Fold 1: ACC=0.5357  F1=0.4222  AUC=0.8193

      Fold 2/15 (test: stressid_h8r2)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9564  val_AUC=0.9634
        Epoch 8/8: loss=0.9426  val_AUC=0.9435
      -> Fold 2: ACC=0.9359  F1=0.9187  AUC=0.9435

      Fold 3/15 (test: stressid_h8s1)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9573  val_AUC=0.9932
        Epoch 8/8: loss=0.9391  val_AUC=1.0000
      -> Fold 3: ACC=0.9950  F1=0.9935  AUC=1.0000

      Fold 4/15 (test: stressid_t6v9)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9497  val_AUC=0.6865
        Epoch 8/8: loss=0.9358  val_AUC=0.6708
      -> Fold 4: ACC=0.7113  F1=0.6620  AUC=0.6708

      Fold 5/15 (test: stressid_r3zm)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9488  val_AUC=0.6800
        Epoch 8/8: loss=0.9339  val_AUC=0.7028
      -> Fold 5: ACC=0.6607  F1=0.6667  AUC=0.7028

      Fold 6/15 (test: stressid_w2t5)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9529  val_AUC=0.6904
        Epoch 8/8: loss=0.9389  val_AUC=0.6935
      -> Fold 6: ACC=0.7708  F1=0.7475  AUC=0.6935

      Fold 7/15 (test: stressid_4woj)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9566  val_AUC=0.9179
        Epoch 8/8: loss=0.9391  val_AUC=0.9344
      -> Fold 7: ACC=0.8358  F1=0.7273  AUC=0.9344

      Fold 8/15 (test: stressid_9txq)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9486  val_AUC=0.6937
        Epoch 8/8: loss=0.9292  val_AUC=0.6953
      -> Fold 8: ACC=0.6577  F1=0.6628  AUC=0.6953

      Fold 9/15 (test: stressid_7h5u)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9594  val_AUC=0.9311
        Epoch 8/8: loss=0.9427  val_AUC=0.9496
      -> Fold 9: ACC=0.8591  F1=0.8091  AUC=0.9496

      Fold 10/15 (test: stressid_45lx)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9552  val_AUC=0.8893
        Epoch 8/8: loss=0.9363  val_AUC=0.8896
      -> Fold 10: ACC=0.8869  F1=0.8333  AUC=0.8896

      Fold 11/15 (test: stressid_6g6y)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9507  val_AUC=0.6699
        Epoch 8/8: loss=0.9270  val_AUC=0.6129
      -> Fold 11: ACC=0.6131  F1=0.6524  AUC=0.6129

      Fold 12/15 (test: stressid_j9h8)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9496  val_AUC=0.8697
        Epoch 8/8: loss=0.9310  val_AUC=0.4765
      -> Fold 12: ACC=0.3333  F1=0.0508  AUC=0.4765

      Fold 13/15 (test: stressid_kycf)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9494  val_AUC=0.7431
        Epoch 8/8: loss=0.9318  val_AUC=0.7648
      -> Fold 13: ACC=0.7619  F1=0.6947  AUC=0.7648

      Fold 14/15 (test: stressid_ctzy)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9496  val_AUC=0.7720
        Epoch 8/8: loss=0.9330  val_AUC=0.8799
      -> Fold 14: ACC=0.7173  F1=0.7368  AUC=0.8799

      Fold 15/15 (test: stressid_c3m7)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9399  val_AUC=0.7033
        Epoch 8/8: loss=0.9270  val_AUC=0.7168
      -> Fold 15: ACC=0.5268  F1=0.6362  AUC=0.7168

      stressid: ACC=0.7067  F1=0.6699  AUC=0.6867

  --- WESAD ---
      Subjects: 15 total, 15 multi-class, 0 single-class, 0 excluded

      Fold 1/15 (test: wesad_s4)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6549  val_AUC=0.9735
        Epoch 8/8: loss=0.6083  val_AUC=0.9848
      -> Fold 1: ACC=0.7849  F1=0.5650  AUC=0.9848

      Fold 2/15 (test: wesad_s6)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.5756  val_AUC=0.9029
        Epoch 8/8: loss=0.5289  val_AUC=0.8970
      -> Fold 2: ACC=0.7486  F1=0.5208  AUC=0.8970

      Fold 3/15 (test: wesad_s10)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6042  val_AUC=0.9454
        Epoch 8/8: loss=0.5680  val_AUC=0.9666
      -> Fold 3: ACC=0.8530  F1=0.7627  AUC=0.9666

      Fold 4/15 (test: wesad_s8)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.5870  val_AUC=0.9007
        Epoch 8/8: loss=0.5409  val_AUC=0.9189
      -> Fold 4: ACC=0.8533  F1=0.7823  AUC=0.9189

      Fold 5/15 (test: wesad_s16)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6277  val_AUC=1.0000
        Epoch 8/8: loss=0.5895  val_AUC=1.0000
      -> Fold 5: ACC=1.0000  F1=1.0000  AUC=1.0000

      Fold 6/15 (test: wesad_s3)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6504  val_AUC=0.9953
        Epoch 8/8: loss=0.6037  val_AUC=0.9943
      -> Fold 6: ACC=0.9382  F1=0.9091  AUC=0.9943

      Fold 7/15 (test: wesad_s13)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.5926  val_AUC=0.9847
        Epoch 8/8: loss=0.5505  val_AUC=0.9912
      -> Fold 7: ACC=0.5989  F1=0.6425  AUC=0.9912

      Fold 8/15 (test: wesad_s11)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6326  val_AUC=1.0000
        Epoch 8/8: loss=0.5752  val_AUC=1.0000
      -> Fold 8: ACC=0.8871  F1=0.8662  AUC=1.0000

      Fold 9/15 (test: wesad_s9)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6449  val_AUC=0.8961
        Epoch 8/8: loss=0.6035  val_AUC=0.9277
      -> Fold 9: ACC=0.8411  F1=0.7411  AUC=0.9277

      Fold 10/15 (test: wesad_s15)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6017  val_AUC=0.9534
        Epoch 8/8: loss=0.5540  val_AUC=0.9629
      -> Fold 10: ACC=0.9249  F1=0.8923  AUC=0.9629

      Fold 11/15 (test: wesad_s2)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6039  val_AUC=0.5645
        Epoch 8/8: loss=0.5484  val_AUC=0.3335
      -> Fold 11: ACC=0.6648  F1=0.1690  AUC=0.3335

      Fold 12/15 (test: wesad_s5)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6375  val_AUC=0.9931
        Epoch 8/8: loss=0.5905  val_AUC=0.9970
      -> Fold 12: ACC=0.9485  F1=0.9212  AUC=0.9970

      Fold 13/15 (test: wesad_s7)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6305  val_AUC=0.9101
        Epoch 8/8: loss=0.5874  val_AUC=0.9476
      -> Fold 13: ACC=0.5753  F1=0.6229  AUC=0.9476

      Fold 14/15 (test: wesad_s14)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6157  val_AUC=0.9999
        Epoch 8/8: loss=0.5687  val_AUC=0.9999
      -> Fold 14: ACC=0.6280  F1=0.6618  AUC=0.9999

      Fold 15/15 (test: wesad_s17)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6189  val_AUC=0.9879
        Epoch 8/8: loss=0.5816  val_AUC=0.9826
      -> Fold 15: ACC=0.9239  F1=0.9079  AUC=0.9826

      wesad: ACC=0.8122  F1=0.7469  AUC=0.8813

  --- COMBINED ---
      Excluded empathicschool: 23 subjects, 66622 windows (held-out)
      Excluding from test pool: ['stressid_71i5', 'stressid_m8g5']
      Test pool: 59 subjects (of 61 multi-class)
      Subjects: 68 total, 61 multi-class, 7 single-class, 2 excluded

      Fold 1/15 (test: stressid_tmvd)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9905  val_AUC=0.8395
        Epoch 8/8: loss=0.9538  val_AUC=0.8442
      -> Fold 1: ACC=0.6131  F1=0.4672  AUC=0.8442

      Fold 2/15 (test: stressid_h8r2)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9819  val_AUC=0.9556
        Epoch 8/8: loss=0.9602  val_AUC=0.9572
      -> Fold 2: ACC=0.9359  F1=0.9187  AUC=0.9572

      Fold 3/15 (test: stressid_h8s1)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9894  val_AUC=1.0000
        Epoch 8/8: loss=0.9643  val_AUC=1.0000
      -> Fold 3: ACC=1.0000  F1=1.0000  AUC=1.0000

      Fold 4/15 (test: stressid_t6v9)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9738  val_AUC=0.7141
        Epoch 8/8: loss=0.9556  val_AUC=0.6819
      -> Fold 4: ACC=0.7083  F1=0.6573  AUC=0.6819

      Fold 5/15 (test: stressid_r3zm)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9789  val_AUC=0.6850
        Epoch 8/8: loss=0.9534  val_AUC=0.7237
      -> Fold 5: ACC=0.6607  F1=0.6667  AUC=0.7237

      Fold 6/15 (test: stressid_w2t5)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9778  val_AUC=0.6873
        Epoch 8/8: loss=0.9550  val_AUC=0.6912
      -> Fold 6: ACC=0.7560  F1=0.7355  AUC=0.6912

      Fold 7/15 (test: stressid_4woj)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9779  val_AUC=0.9392
        Epoch 8/8: loss=0.9572  val_AUC=0.9366
      -> Fold 7: ACC=0.8358  F1=0.7273  AUC=0.9366

      Fold 8/15 (test: stressid_9txq)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9759  val_AUC=0.6961
        Epoch 8/8: loss=0.9510  val_AUC=0.6931
      -> Fold 8: ACC=0.6250  F1=0.6087  AUC=0.6931

      Fold 9/15 (test: stressid_7h5u)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9714  val_AUC=0.9398
        Epoch 8/8: loss=0.9442  val_AUC=0.9533
      -> Fold 9: ACC=0.8423  F1=0.7814  AUC=0.9533

      Fold 10/15 (test: stressid_45lx)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9762  val_AUC=0.9084
        Epoch 8/8: loss=0.9529  val_AUC=0.9002
      -> Fold 10: ACC=0.8810  F1=0.8230  AUC=0.9002

      Fold 11/15 (test: stressid_6g6y)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9752  val_AUC=0.6878
        Epoch 8/8: loss=0.9560  val_AUC=0.6147
      -> Fold 11: ACC=0.6161  F1=0.6649  AUC=0.6147

      Fold 12/15 (test: wesad_s5)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9811  val_AUC=1.0000
        Epoch 8/8: loss=0.9586  val_AUC=0.9999
      -> Fold 12: ACC=0.8618  F1=0.7536  AUC=0.9999

      Fold 13/15 (test: wesad_s2)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9757  val_AUC=0.9461
        Epoch 8/8: loss=0.9514  val_AUC=0.9390
      -> Fold 13: ACC=0.6562  F1=0.0320  AUC=0.9390

      Fold 14/15 (test: wesad_s10)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9715  val_AUC=0.4977
        Epoch 8/8: loss=0.9478  val_AUC=0.5430
      -> Fold 14: ACC=0.3753  F1=0.5458  AUC=0.5430

      Fold 15/15 (test: wesad_s13)
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.9774  val_AUC=0.9283
        Epoch 8/8: loss=0.9520  val_AUC=0.9500
      -> Fold 15: ACC=0.6043  F1=0.6386  AUC=0.9500

      combined: ACC=0.7156  F1=0.6655  AUC=0.7143

============================================================
  BENCHMARK COMPLETE
============================================================
  cnn_baseline                    stressid=0.6867 | wesad=0.8813 | combined=0.7143

  Leaderboard: C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML\benchmark_results\leaderboard.csv    

  Top-5 by combined AUC:
    cnn_baseline                    AUC=0.7143  ACC=0.7156  F1=0.6655

============================================================
  ALL RESULTS: C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML\benchmark_results
============================================================
PS C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML> PS C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML> venv\Scripts\python.exe scripts\run_all_models_benchmark.py --models conv_moe_mf --exclude-subjects "stressid_m8g5,stressid_71i5" --exclude-dataset empathicschool
Device: cuda
Device: cuda

============================================================
  UNIFIED RESEARCH MODEL BENCHMARK
============================================================
  Device: cuda
  Models: 1
    - conv_moe_mf                    (phase3)  ConvMoE-MF: light conv encoders, MoE fusion, dual GRL
============================================================
  stressid: 16974 windows, 53 subjects
  wesad: 5517 windows, 15 subjects
  combined: 89113 windows, 91 subjects
  Excluding subjects from test pool: ['stressid_m8g5', 'stressid_71i5']

============================================================
  MODEL: conv_moe_mf (phase3)
  ConvMoE-MF: light conv encoders, MoE fusion, dual GRL
============================================================

  --- STRESSID ---
      Excluding from test pool: ['stressid_71i5', 'stressid_m8g5']
      Test pool: 44 subjects (of 46 multi-class)
      Subjects: 53 total, 46 multi-class, 7 single-class, 2 excluded

      Fold 1/15 (test: stressid_tmvd)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.3266
        SSL epoch 4: loss=15.6893
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6741  val_AUC=0.8345
        Epoch 8/8: loss=0.6579  val_AUC=0.8358
      -> Fold 1: ACC=0.7738  F1=0.6000  AUC=0.8358

      Fold 2/15 (test: stressid_h8r2)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=17.2020
        SSL epoch 4: loss=16.4151
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6799  val_AUC=0.9529
        Epoch 8/8: loss=0.6580  val_AUC=0.9473
      -> Fold 2: ACC=0.9391  F1=0.9231  AUC=0.9473

      Fold 3/15 (test: stressid_h8s1)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.8702
        SSL epoch 4: loss=16.2329
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6661  val_AUC=1.0000
        Epoch 8/8: loss=0.6578  val_AUC=1.0000
      -> Fold 3: ACC=0.9950  F1=0.9935  AUC=1.0000

      Fold 4/15 (test: stressid_t6v9)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=17.3467
        SSL epoch 4: loss=16.6925
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6686  val_AUC=0.6657
        Epoch 8/8: loss=0.6575  val_AUC=0.6567
      -> Fold 4: ACC=0.7113  F1=0.6620  AUC=0.6567

      Fold 5/15 (test: stressid_r3zm)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.0719
        SSL epoch 4: loss=15.7156
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6697  val_AUC=0.7382
        Epoch 8/8: loss=0.6588  val_AUC=0.7387
      -> Fold 5: ACC=0.6607  F1=0.6667  AUC=0.7387

      Fold 6/15 (test: stressid_w2t5)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=17.0796
        SSL epoch 4: loss=16.1256
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6806  val_AUC=0.8177
        Epoch 8/8: loss=0.6641  val_AUC=0.8229
      -> Fold 6: ACC=0.7976  F1=0.7703  AUC=0.8229

      Fold 7/15 (test: stressid_4woj)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=17.0411
        SSL epoch 4: loss=16.2957
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6712  val_AUC=0.8231
        Epoch 8/8: loss=0.6644  val_AUC=0.8279
      -> Fold 7: ACC=0.8358  F1=0.7273  AUC=0.8279

      Fold 8/15 (test: stressid_9txq)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=17.6464
        SSL epoch 4: loss=15.7488
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6746  val_AUC=0.7341
        Epoch 8/8: loss=0.6625  val_AUC=0.7372
      -> Fold 8: ACC=0.6607  F1=0.6667  AUC=0.7372

      Fold 9/15 (test: stressid_7h5u)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=17.7965
        SSL epoch 4: loss=16.1003
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6776  val_AUC=0.9198
        Epoch 8/8: loss=0.6613  val_AUC=0.9305
      -> Fold 9: ACC=0.8691  F1=0.8282  AUC=0.9305

      Fold 10/15 (test: stressid_45lx)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.7630
        SSL epoch 4: loss=15.9046
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6712  val_AUC=0.8712
        Epoch 8/8: loss=0.6607  val_AUC=0.8706
      -> Fold 10: ACC=0.8750  F1=0.8125  AUC=0.8706

      Fold 11/15 (test: stressid_6g6y)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=17.1318
        SSL epoch 4: loss=16.0931
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6673  val_AUC=0.8794
        Epoch 8/8: loss=0.6551  val_AUC=0.8661
      -> Fold 11: ACC=0.6875  F1=0.7009  AUC=0.8661

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
        SSL epoch 2: loss=16.1885
        SSL epoch 4: loss=15.3972
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6947  val_AUC=0.8281
        Epoch 8/8: loss=0.6883  val_AUC=0.8113
      -> Fold 6: ACC=0.7976  F1=0.7703  AUC=0.8113

      Fold 7/15 (test: stressid_4woj)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.1493
        SSL epoch 4: loss=15.3627
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6912  val_AUC=0.8283
        Epoch 8/8: loss=0.6857  val_AUC=0.8803
      -> Fold 7: ACC=0.8358  F1=0.7273  AUC=0.8803

      Fold 8/15 (test: stressid_9txq)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.7684
        SSL epoch 4: loss=15.0183
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6957  val_AUC=0.7602
        Epoch 8/8: loss=0.6873  val_AUC=0.7672
      -> Fold 8: ACC=0.6607  F1=0.6667  AUC=0.7672

      Fold 9/15 (test: stressid_7h5u)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.8931
        SSL epoch 4: loss=15.2852
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6940  val_AUC=0.9288
        Epoch 8/8: loss=0.6876  val_AUC=0.9370
      -> Fold 9: ACC=0.8691  F1=0.8282  AUC=0.9370

      Fold 10/15 (test: stressid_45lx)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.5734
        SSL epoch 4: loss=16.4437
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6976  val_AUC=0.8812
        Epoch 8/8: loss=0.6914  val_AUC=0.8864
      -> Fold 10: ACC=0.8839  F1=0.8282  AUC=0.8864

      Fold 11/15 (test: stressid_6g6y)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.3329
        SSL epoch 4: loss=14.8798
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6909  val_AUC=0.8307
        Epoch 8/8: loss=0.6829  val_AUC=0.8171
      -> Fold 11: ACC=0.6964  F1=0.7119  AUC=0.8171

      Fold 12/15 (test: wesad_s5)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.4020
        SSL epoch 4: loss=15.0644
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7008  val_AUC=0.0398
        Epoch 8/8: loss=0.6897  val_AUC=0.0005
      -> Fold 12: ACC=0.6504  F1=0.0000  AUC=0.0005

      Fold 13/15 (test: wesad_s2)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=16.5817
        SSL epoch 4: loss=16.4333
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6966  val_AUC=0.8902
        Epoch 8/8: loss=0.6874  val_AUC=0.5000
      -> Fold 13: ACC=0.6506  F1=0.0000  AUC=0.5000

      Fold 14/15 (test: wesad_s10)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.6429
        SSL epoch 4: loss=15.0561
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6898  val_AUC=0.8734
        Epoch 8/8: loss=0.6848  val_AUC=0.8094
      -> Fold 14: ACC=0.6194  F1=0.0000  AUC=0.8094

      Fold 15/15 (test: wesad_s13)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.7382
        SSL epoch 4: loss=14.9375
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6927  val_AUC=0.1831
        Epoch 8/8: loss=0.6860  val_AUC=0.1383
      -> Fold 15: ACC=0.6396  F1=0.0000  AUC=0.1383

      combined: ACC=0.7452  F1=0.6285  AUC=0.6943

============================================================
  BENCHMARK COMPLETE
============================================================
  conv_moe_mf                     stressid=0.7527 | wesad=0.7483 | combined=0.6943

  Leaderboard: C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML\benchmark_results\leaderboard.csv    

  Top-5 by combined AUC:
    conv_moe_mf                     AUC=0.6943  ACC=0.7452  F1=0.6285

============================================================
  ALL RESULTS: C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML\benchmark_results
============================================================

```


# SSVB-CASA-AIS Exact Terminal Log

```text
PS C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML> venv\Scripts\python.exe scripts\run_all_models_benchmark.py --models ssvb_casa_ais --exclude-subjects "stressid_m8g5,stressid_71i5" --exclude-dataset empathicschool
Device: cuda
Device: cuda

============================================================
  UNIFIED RESEARCH MODEL BENCHMARK
============================================================
  Device: cuda
  Models: 1
    - ssvb_casa_ais                  (phase3)  Full SSVB-CASA-AIS: 9 experts, cross-attention, global MoE, GRL
============================================================
  stressid: 16974 windows, 53 subjects
  wesad: 5517 windows, 15 subjects
  combined: 89113 windows, 91 subjects
  Excluding subjects from test pool: ['stressid_m8g5', 'stressid_71i5']

============================================================
  MODEL: ssvb_casa_ais (phase3)
  Full SSVB-CASA-AIS: 9 experts, cross-attention, global MoE, GRL
============================================================

  --- STRESSID ---
      Excluding from test pool: ['stressid_71i5', 'stressid_m8g5']
      Test pool: 44 subjects (of 46 multi-class)
      Subjects: 53 total, 46 multi-class, 7 single-class, 2 excluded

      Fold 1/15 (test: stressid_tmvd)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.0789
        SSL epoch 4: loss=14.8132
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7256  val_AUC=0.1583
        Epoch 8/8: loss=0.7091  val_AUC=0.1568
      -> Fold 1: ACC=0.8304  F1=0.0000  AUC=0.1568

      Fold 2/15 (test: stressid_h8r2)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=14.8487
        SSL epoch 4: loss=14.6465
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7192  val_AUC=0.0453
        Epoch 8/8: loss=0.7086  val_AUC=0.0512
      -> Fold 2: ACC=0.6346  F1=0.0000  AUC=0.0512

      Fold 3/15 (test: stressid_h8s1)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.0222
        SSL epoch 4: loss=14.7669
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7143  val_AUC=0.0000
        Epoch 8/8: loss=0.7033  val_AUC=0.0000
      -> Fold 3: ACC=0.6169  F1=0.0000  AUC=0.0000

      Fold 4/15 (test: stressid_t6v9)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.1242
        SSL epoch 4: loss=14.8122
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7188  val_AUC=0.3249
        Epoch 8/8: loss=0.7050  val_AUC=0.5926
      -> Fold 4: ACC=0.5417  F1=0.0000  AUC=0.5926

      Fold 5/15 (test: stressid_r3zm)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.0540
        SSL epoch 4: loss=14.7710
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7216  val_AUC=0.2906
        Epoch 8/8: loss=0.7032  val_AUC=0.2786
      -> Fold 5: ACC=0.3780  F1=0.0000  AUC=0.2786

      Fold 6/15 (test: stressid_w2t5)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.1084
        SSL epoch 4: loss=14.7782
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7152  val_AUC=0.1293
        Epoch 8/8: loss=0.7054  val_AUC=0.1209
      -> Fold 6: ACC=0.5149  F1=0.0000  AUC=0.1209

      Fold 7/15 (test: stressid_4woj)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.1506
        SSL epoch 4: loss=14.7926
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7245  val_AUC=0.1067
        Epoch 8/8: loss=0.7063  val_AUC=0.8720
      -> Fold 7: ACC=0.7811  F1=0.0000  AUC=0.8720

      Fold 8/15 (test: stressid_9txq)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.0806
        SSL epoch 4: loss=14.7545
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7118  val_AUC=0.3481
        Epoch 8/8: loss=0.7059  val_AUC=0.3512
      -> Fold 8: ACC=0.3780  F1=0.0000  AUC=0.3512

      Fold 9/15 (test: stressid_7h5u)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=14.8877
        SSL epoch 4: loss=14.5684
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7238  val_AUC=0.0646
        Epoch 8/8: loss=0.7057  val_AUC=0.0691
      -> Fold 9: ACC=0.6812  F1=0.0000  AUC=0.0691

      Fold 10/15 (test: stressid_45lx)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.0661
        SSL epoch 4: loss=14.8189
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7168  val_AUC=0.0348
        Epoch 8/8: loss=0.7064  val_AUC=0.0501
      -> Fold 10: ACC=0.7173  F1=0.0000  AUC=0.0501

      Fold 11/15 (test: stressid_6g6y)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.0669
        SSL epoch 4: loss=14.7947
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7146  val_AUC=0.1066
        Epoch 8/8: loss=0.7052  val_AUC=0.1881
      -> Fold 11: ACC=0.3214  F1=0.0000  AUC=0.1881

      Fold 12/15 (test: stressid_j9h8)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.0898
        SSL epoch 4: loss=14.8091
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7152  val_AUC=0.1851
        Epoch 8/8: loss=0.7025  val_AUC=0.3201
      -> Fold 12: ACC=0.3155  F1=0.0000  AUC=0.3201

      Fold 13/15 (test: stressid_kycf)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.0889
        SSL epoch 4: loss=14.8425
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7191  val_AUC=0.2435
        Epoch 8/8: loss=0.7072  val_AUC=0.1960
      -> Fold 13: ACC=0.6042  F1=0.0000  AUC=0.1960

      Fold 14/15 (test: stressid_ctzy)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.1139
        SSL epoch 4: loss=14.8212
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7154  val_AUC=0.3199
        Epoch 8/8: loss=0.7024  val_AUC=0.1414
      -> Fold 14: ACC=0.3214  F1=0.0000  AUC=0.1414

      Fold 15/15 (test: stressid_c3m7)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.0900
        SSL epoch 4: loss=14.8348
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7222  val_AUC=0.3103
        Epoch 8/8: loss=0.7014  val_AUC=0.2682
      -> Fold 15: ACC=0.1131  F1=0.0000  AUC=0.2682

      stressid: ACC=0.5042  F1=0.0000  AUC=0.2715

  --- WESAD ---
      Subjects: 15 total, 15 multi-class, 0 single-class, 0 excluded

      Fold 1/15 (test: wesad_s4)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.1914
        SSL epoch 4: loss=14.9131
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6022  val_AUC=0.8820
        Epoch 8/8: loss=0.5974  val_AUC=0.9190
      -> Fold 1: ACC=0.6453  F1=0.0000  AUC=0.9190

      Fold 2/15 (test: wesad_s6)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.2602
        SSL epoch 4: loss=15.0202
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6022  val_AUC=0.0896
        Epoch 8/8: loss=0.5967  val_AUC=0.1067
      -> Fold 2: ACC=0.6448  F1=0.0000  AUC=0.1067

      Fold 3/15 (test: wesad_s10)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.1885
        SSL epoch 4: loss=15.0536
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.5998  val_AUC=0.7106
        Epoch 8/8: loss=0.5934  val_AUC=0.6527
      -> Fold 3: ACC=0.6194  F1=0.0000  AUC=0.6527

      Fold 4/15 (test: wesad_s8)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.2405
        SSL epoch 4: loss=15.0110
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.5997  val_AUC=0.4251
        Epoch 8/8: loss=0.5915  val_AUC=0.3721
      -> Fold 4: ACC=0.6359  F1=0.0000  AUC=0.3721

      Fold 5/15 (test: wesad_s16)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.2231
        SSL epoch 4: loss=14.9762
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6002  val_AUC=0.0000
        Epoch 8/8: loss=0.5945  val_AUC=0.0000
      -> Fold 5: ACC=0.6361  F1=0.0000  AUC=0.0000

      Fold 6/15 (test: wesad_s3)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.3278
        SSL epoch 4: loss=15.0744
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6023  val_AUC=0.7316
        Epoch 8/8: loss=0.5965  val_AUC=0.5919
      -> Fold 6: ACC=0.6404  F1=0.0000  AUC=0.5919

      Fold 7/15 (test: wesad_s13)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.1562
        SSL epoch 4: loss=14.8564
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6030  val_AUC=0.5057
        Epoch 8/8: loss=0.5980  val_AUC=0.4333
      -> Fold 7: ACC=0.6396  F1=0.0000  AUC=0.4333

      Fold 8/15 (test: wesad_s11)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.2865
        SSL epoch 4: loss=14.9374
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6030  val_AUC=0.0000
        Epoch 8/8: loss=0.5970  val_AUC=0.0000
      -> Fold 8: ACC=0.6344  F1=0.0000  AUC=0.0000

      Fold 9/15 (test: wesad_s9)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.2255
        SSL epoch 4: loss=14.9365
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.5998  val_AUC=0.6582
        Epoch 8/8: loss=0.5928  val_AUC=0.5878
      -> Fold 9: ACC=0.6466  F1=0.0000  AUC=0.5878

      Fold 10/15 (test: wesad_s15)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.3379
        SSL epoch 4: loss=15.0528
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6008  val_AUC=0.5185
        Epoch 8/8: loss=0.5947  val_AUC=0.4615
      -> Fold 10: ACC=0.6300  F1=0.0000  AUC=0.4615

      Fold 11/15 (test: wesad_s2)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.4893
        SSL epoch 4: loss=15.2201
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6004  val_AUC=0.7738
        Epoch 8/8: loss=0.5935  val_AUC=0.7295
      -> Fold 11: ACC=0.6506  F1=0.0000  AUC=0.7295

      Fold 12/15 (test: wesad_s5)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.2467
        SSL epoch 4: loss=14.8955
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6009  val_AUC=0.0734
        Epoch 8/8: loss=0.5946  val_AUC=0.0642
      -> Fold 12: ACC=0.6504  F1=0.0000  AUC=0.0642

      Fold 13/15 (test: wesad_s7)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.3116
        SSL epoch 4: loss=15.1272
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6010  val_AUC=0.0301
        Epoch 8/8: loss=0.5944  val_AUC=0.0286
      -> Fold 13: ACC=0.6493  F1=0.0000  AUC=0.0286

      Fold 14/15 (test: wesad_s14)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.2656
        SSL epoch 4: loss=15.0443
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6000  val_AUC=0.0020
        Epoch 8/8: loss=0.5925  val_AUC=0.0025
      -> Fold 14: ACC=0.6361  F1=0.0000  AUC=0.0025

      Fold 15/15 (test: wesad_s17)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=15.1047
        SSL epoch 4: loss=14.9028
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.6022  val_AUC=0.2148
        Epoch 8/8: loss=0.5954  val_AUC=0.1506
      -> Fold 15: ACC=0.6194  F1=0.0000  AUC=0.1506

      wesad: ACC=0.6384  F1=0.0000  AUC=0.4654

  --- COMBINED ---
      Excluded empathicschool: 23 subjects, 66622 windows (held-out)
      Excluding from test pool: ['stressid_71i5', 'stressid_m8g5']
      Test pool: 59 subjects (of 61 multi-class)
      Subjects: 68 total, 61 multi-class, 7 single-class, 2 excluded

      Fold 1/15 (test: stressid_tmvd)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=14.3898
        SSL epoch 4: loss=14.1728
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7315  val_AUC=0.1496
        Epoch 8/8: loss=0.7209  val_AUC=0.1791
      -> Fold 1: ACC=0.8304  F1=0.0000  AUC=0.1791

      Fold 2/15 (test: stressid_h8r2)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=14.7348
        SSL epoch 4: loss=14.1252
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7273  val_AUC=0.0432
        Epoch 8/8: loss=0.7198  val_AUC=0.0530
      -> Fold 2: ACC=0.6346  F1=0.0000  AUC=0.0530

      Fold 3/15 (test: stressid_h8s1)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=14.3439
        SSL epoch 4: loss=14.0937
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7314  val_AUC=0.0000
        Epoch 8/8: loss=0.7257  val_AUC=0.0000
      -> Fold 3: ACC=0.6169  F1=0.0000  AUC=0.0000

      Fold 4/15 (test: stressid_t6v9)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=14.7512
        SSL epoch 4: loss=14.1735
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7248  val_AUC=0.3704
        Epoch 8/8: loss=0.7157  val_AUC=0.3932
      -> Fold 4: ACC=0.5417  F1=0.0000  AUC=0.3932

      Fold 5/15 (test: stressid_r3zm)
        SSL pretraining (4 epochs)
        SSL epoch 2: loss=14.4057
        SSL epoch 4: loss=14.1286
        Fine-tuning (8 epochs)
        Epoch 4/8: loss=0.7252  val_AUC=0.3644
        Epoch 8/8: loss=0.7182  val_AUC=0.3869
      -> Fold 5: ACC=0.3780  F1=0.0000  AUC=0.3869

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
