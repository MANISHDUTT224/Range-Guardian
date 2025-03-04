import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_sensor_data():
    url = f"{BASE_URL}/sensor-data"
    payload = {
        "battery_level": 25.5,
        "latitude": 37.7749,
        "longitude": -122.4194,
        "weather_condition": "clear"
    }
    response = requests.post(url, json=payload)
    print("Sensor Data Response:", json.dumps(response.json(), indent=2))

def test_charging_stations():
    url = f"{BASE_URL}/charging-stations"
    response = requests.get(url)
    print("Charging Stations Response:", json.dumps(response.json(), indent=2))

def test_decision():
    url = f"{BASE_URL}/decision"
    payload = {
        "battery_level": 25.0,
        "latitude": 37.7749,
        "longitude": -122.4194,
        "weather_condition": "clear"
    }
    response = requests.post(url, json=payload)
    print("Decision Response:", json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_sensor_data()
    test_charging_stations()
    test_decision()
