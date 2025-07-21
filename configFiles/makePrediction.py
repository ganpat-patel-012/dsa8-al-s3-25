import requests
from configFiles.config import API_URL

def get_prediction_lstm(payload):
    try:
        response = requests.post(f"{API_URL}/predict/lstm", json=payload)
        if response.status_code == 200:
            return response.json().get("probability_lstm", "N/A")
        else:
            return f"Error {response.status_code}"
    except requests.RequestException as e:
        return f"❌ API Request Failed: {e}"
    

def get_prediction_gru(payload):
    try:
        response = requests.post(f"{API_URL}/predict/gru", json=payload)
        if response.status_code == 200:
            return response.json().get("probability_gru", "N/A")
        else:
            return f"Error {response.status_code}"
    except requests.RequestException as e:
        return f"❌ API Request Failed: {e}"
    
def get_prediction_textcnn(payload):
    try:
        response = requests.post(f"{API_URL}/predict/textcnn", json=payload)
        if response.status_code == 200:
            return response.json().get("probability_textcnn", "N/A")
        else:
            return f"Error {response.status_code}"
    except requests.RequestException as e:
        return f"❌ API Request Failed: {e}"