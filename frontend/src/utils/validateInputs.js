// frontend/src/utils/validateInputs.js

export function validateAnalysisInputs({ faceFile, voiceFile, eegData, gsrData, eegFile, gsrFile }) {
  const errors = [];

  // Must have at least one modality
  const hasModality = faceFile || voiceFile || eegFile || gsrFile || eegData?.trim() || gsrData?.trim();
  if (!hasModality) {
    errors.push('Provide at least one input: photo, voice recording, or EEG/GSR data.');
  }

  // Face file validation
  if (faceFile) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!validTypes.includes(faceFile.type)) {
      errors.push('Face image must be JPG or PNG format.');
    }
    if (faceFile.size > 10 * 1024 * 1024) {
      errors.push('Face image must be under 10MB.');
    }
  }

  // Voice file validation
  if (voiceFile) {
    const validTypes = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/ogg',
                        'audio/webm', 'audio/m4a'];
    if (!validTypes.some(t => voiceFile.type.includes(t.split('/')[1]))) {
      errors.push('Voice recording must be WAV, MP3, OGG, WebM, or M4A format.');
    }
    if (voiceFile.size > 50 * 1024 * 1024) {
      errors.push('Voice file must be under 50MB.');
    }
  }

  // EEG validation: must be comma-separated numbers
  if (eegData?.trim()) {
    const nums = eegData.trim().split(',').map(Number);
    if (nums.some(isNaN)) {
      errors.push('EEG data must be comma-separated numbers only (e.g., 0.52, 0.61, 0.58).');
    }
    if (nums.length < 10) {
      errors.push(`EEG data needs at least 10 values for reliable analysis (you provided ${nums.length}).`);
    }
  }

  // GSR validation
  if (gsrData?.trim()) {
    const nums = gsrData.trim().split(',').map(Number);
    if (nums.some(isNaN)) {
      errors.push('GSR data must be comma-separated numbers only.');
    }
    if (nums.some(n => n < 0 || n > 100)) {
      errors.push('GSR values should be in the 0–100 µS range.');
    }
  }

  return errors;
}

// Response validation after fetch
export function validateAnalysisResponse(data) {
  const errors = [];

  if (!data) {
    errors.push('Empty response from server.');
    return errors;
  }
  if (!['Low','Moderate','High','Extreme'].includes(data.stress_level)) {
    errors.push(`Invalid stress_level: ${data.stress_level}`);
  }
  
  const score = data.stress_probability != null ? data.stress_probability : data.fused_score;
  if (typeof score !== 'number' || score < 0 || score > 1) {
    errors.push(`Invalid stress score: ${score}`);
  }
  
  const individual = data.individual_predictions || {};
  const face_score = individual.facial != null ? individual.facial : data.face_score;
  const voice_score = individual.voice != null ? individual.voice : data.voice_score;
  const physio_score = individual.physiological != null ? individual.physiological : data.physio_score;
  
  const hasScore = face_score != null || voice_score != null || physio_score != null;
  if (!hasScore) {
    errors.push('No modality scores in response. The model may not have processed the inputs.');
  }

  return errors;
}
