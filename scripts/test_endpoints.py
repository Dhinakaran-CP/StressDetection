import urllib.request
import json
import sys

BASE_URL = "http://localhost:5000"

def test_endpoint(path, data=None):
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, headers=headers)
    if data is not None:
        req.data = json.dumps(data).encode("utf-8")
        req.method = "POST"
    else:
        req.method = "GET"
    
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            print(f"[PASS] {path} returned status {response.status}")
            return res_data
    except Exception as e:
        print(f"[FAIL] {path} failed: {e}")
        return None

print("=== Running Integration Tests ===")
# 1. Test health
health = test_endpoint("/api/health")
print("Health status:", health.get("status"))

# 2. Test version
version = test_endpoint("/api/model/version")
print("Use deep:", version.get("use_deep"))

# 3. Test fallback
fallback = test_endpoint("/api/fallback/status")
print("Fallback active:", fallback.get("fallback_active"))
print("Active Model:", fallback.get("active_model"))

# 4. Test modality
modality = test_endpoint("/api/modality/status")
print("Modality Status:", modality.get("status"))

# 5. Test real-time prediction
dummy_face = [0.1] * 18
dummy_voice = [0.2] * 12
dummy_physio = [0.3] * 5

predict_payload = {
    "face": dummy_face,
    "voice": dummy_voice,
    "physio": dummy_physio,
    "user_id": "test_user",
    "sensitivity": 0.5
}

prediction = test_endpoint("/api/predict/realtime", data=predict_payload)
if prediction:
    print("Class:", prediction.get("predicted_class"))
    print("Prob:", prediction.get("probability"))
    print("Confidence:", prediction.get("confidence_percentage"))
    print("Fallback:", prediction.get("resilience_status", {}).get("fallback_active"))

# 6. Test explain shap
explanation = test_endpoint("/api/explain/shap", data=predict_payload)
if explanation:
    print("Explain status:", explanation.get("status"))
    print("Explain keys:", list(explanation.get("explainability", {}).keys()))

print("=== Tests Completed ===")
