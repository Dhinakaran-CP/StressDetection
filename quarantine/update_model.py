import re

with open('backend/model.py', 'r', encoding='utf-8') as f:
    content = f.read()

# We will modify the predict method to collect errors and return them if probs is empty
predict_method = '''    def predict(self, facial_features=None, voice_features=None, phys_features=None, temp_image_path=None, sensitivity=0.5):
        if not self.is_trained: return {'error': 'Models not loaded'}
        
        probs = []
        preds = {'facial': None, 'voice': None, 'physiological': None}
        errors = {}
        
        # 1. Facial Expert
        if facial_features is not None and self.facial_model:
            try:
                ff = np.array(facial_features).reshape(1, -1)
                if self.facial_scaler:
                    ff = self.facial_scaler.transform(ff)
                f_prob = self.facial_model.predict_proba(ff)[0][1]
                preds['facial'] = f_prob
                
                if temp_image_path:
                    smile = self.detect_smile(temp_image_path)
                    if smile > 0.5:
                        f_prob = max(0.0, f_prob - 0.4) 
                
                probs.append(f_prob)
            except Exception as e: 
                print(f"Facial pred error: {e}")
                errors['facial'] = f"Error: {e}"
        else:
            errors['facial'] = "Missing features" if facial_features is None else "Missing model"
            
        # 2. Voice Expert
        if voice_features is not None and self.voice_model:
            try:
                vf = np.array(voice_features).reshape(1, -1)
                if self.voice_scaler:
                    vf_scaled = self.voice_scaler.transform(vf)
                else:
                    vf_scaled = vf
                v_prob = self.voice_model.predict_proba(vf_scaled)[0][1]
                preds['voice'] = v_prob
                probs.append(v_prob)
            except Exception as e: 
                print(f"Voice pred error: {e}")
                errors['voice'] = f"Error: {e}"
        else:
            errors['voice'] = "Missing features" if voice_features is None else "Missing model"
            
        # 3. Physio Expert
        if phys_features is not None and self.phys_model:
            try:
                pf = np.array(phys_features).reshape(1, -1)
                if self.phys_scaler:
                    pf = self.phys_scaler.transform(pf)
                p_prob = self.phys_model.predict_proba(pf)[0][1]
                preds['physiological'] = p_prob
                probs.append(p_prob)
            except Exception as e: 
                print(f"Physio pred error: {e}")
                errors['physiological'] = f"Error: {e}"
        else:
            errors['physiological'] = "Missing features" if phys_features is None else "Missing model"
            
        if not probs:
            return {'error': f'No valid predictions. {errors}'}'''

# Replace the original predict method
content = re.sub(r'    def predict\(self, facial_features=None, voice_features=None, phys_features=None, temp_image_path=None, sensitivity=0\.5\):.*?        if not probs:\n            return \{\'error\': \'No valid predictions\. Check inputs or model capabilities\.\'\}', predict_method, content, flags=re.DOTALL)

with open('backend/model.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated model.py with error details")
