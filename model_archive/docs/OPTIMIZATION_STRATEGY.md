# Comprehensive Multimodal Model Optimization Plan

## 1. Current State Assessment
- **Facial Model**: Currently relies on extracting features (Haar cascades, histograms) from video frames in `StressID Dataset`.
  - *Limitation*: Only ~378 video samples available. Face detection in videos can be inconsistent due to lighting/motion.
- **Voice Model**: Uses ~378 audio samples from `StressID Dataset`.
  - *Status*: Valid, but small dataset.
- **Physiological Model**: Uses ~774 physiological logs from `StressID Dataset`.
  - *Status*: Strongest modality currently due to direct sensor data.
- **Fusion**: Late fusion (averaging probabilities) of 3 Random Forest classifiers trained on the above extracted features.

## 2. Proposed "Specialized Expert" Strategy
To achieve state-of-the-art results for real-time usage, we will train **Specialized Expert Models** for each modality using the best available dataset for that specific task.

### A. Facial Expression Expert (The "Vision Specialist")
**Dataset to Use**: `facesData` (f:\Multimodal_stress_Detection\facesData)
- **Why?**: This dataset contains ~12,275 pre-cropped, labeled images (9,795 Train / 2,480 Test) specifically for "Stress" vs "No Stress". This is **30x larger** than the video dataset and removes the noise of background/lighting found in raw videos.
- **Approach**: 
  1. Create a script `train_facial_expert.py`.
  2. Iterate through `facesData/train/stress` and `facesData/train/nostress`.
  3. Extract facial features using the *exact same* `extract_facial_features` function from `model.py` (to ensure compatibility with the real-time app).
  4. Train a dedicated `RandomForest` classifier on these ~10,000 feature vectors.
  5. Save as `facial_expert_model.pkl`.

### B. Voice Stress Expert (The "Audio Specialist")
**Dataset to Use**: `StressID Dataset` Audio (f:\Multimodal_stress_Detection\StressID\StressID Dataset\Audio)
- **Why?**: The 378 samples are high-quality, long-duration recordings of specific stress tasks (e.g., "Public Speaking", "Mental Math"). This is more valuable for *stress* detection than generic emotion datasets like RAVDESS (which are acted).
- **Approach**: 
  1. Continue using `integrate_dataset.py` to extract features from these files.
  2. Train a dedicated classifier on just these audio features.
  3. Save as `voice_expert_model.pkl`.

### C. Physiological Expert (The "Biosignal Specialist")
**Dataset to Use**: `StressID Dataset` Physio (f:\Multimodal_stress_Detection\StressID\StressID Dataset\Physiological)
- **Why?**: This is the only source of real ECG/EDA data synchronized with the stress tasks.
- **Approach**: 
  1. Continue using `integrate_dataset.py` to extract features.
  2. Train a dedicated classifier.
  3. Save as `physio_expert_model.pkl`.

## 3. Late Fusion Integration
We will modify the main `MultimodalStressDetector` class in `model.py` to load these **three separate model files** instead of one monolithic file.

**Algorithm**:
1. **Input**: Real-time Video Frame + Audio Chunk + (Optional) Physio Data.
2. **Step 1**: Pass Video Frame -> `Facial Expert` -> Get Probability $P_{face}$.
3. **Step 2**: Pass Audio Chunk -> `Voice Expert` -> Get Probability $P_{voice}$.
4. **Step 3**: Pass Physio Data -> `Physio Expert` -> Get Probability $P_{physio}$.
5. **Step 4 (Fusion)**: 
   $$ P_{final} = w_f \cdot P_{face} + w_v \cdot P_{voice} + w_p \cdot P_{physio} $$
   *Where weights $w$ can be dynamic based on confidence or fixed (e.g., 0.4, 0.4, 0.2).*

## 4. Implementation Steps Roadmap
1. **[DONE]** Feature Extraction (Face): Create and run `train_face_expert.py` to process `facesData`.
2. **[DONE]** Feature Extraction (Voice/Physio): Ensure `integrate_dataset.py` is run on the full `StressID` dataset.
3. **[DONE]** Training: Train the 3 experts independently.
4. **[DONE]** Integration: Update `model.py` to load 3 separate `.pkl` files.
5. **[DONE]** Testing: Run `app.py` and verify real-time performance.

## 5. Expected Improvements
- **Accuracy Note (Data Leakage Correction)**: In early phases, facial models tested on external image datasets (`facesData`) or with random splits achieved >70%-90%. However, under **strict Leave-One-Subject-Out (LOSO)** on synchronized video/audio windows, true subject-generalized accuracy for unimodal experts is ~60-65%. The baseline goal for the fused engine is now >60%, reflecting realistic cross-subject generalization without identity leakage.
- **Robustness**: If one modality fails (e.g., camera is dark), the other experts (Voice/Physio) naturally compensate.
- **Real-Time Speed**: Random Forest on extracted features is lightweight (<10ms inference), making it perfect for the user's real-time constraints compared to heavy Deep Learning models.

