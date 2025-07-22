import requests
from configFiles.config import API_URL

def get_prediction_all(payload):
    try:
        response = requests.post(f"{API_URL}/predict/all", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return f"Error {response.status_code}"
    except requests.RequestException as e:
        return f"❌ API Request Failed: {e}"