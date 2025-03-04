# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import math
#
# app = FastAPI(title="Range Guardian API (In-Memory)", version="1.0")
#
# # Enable CORS - for development, you can allow all origins by using allow_origins=["*"]
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],  # Add your frontend's exact URL
#     allow_credentials=True,
#     allow_methods=["GET", "POST", "OPTIONS"],  # Restrict to only required methods
#     allow_headers=["Content-Type", "Authorization"],  # Restrict to necessary headers
# )
# from fastapi import Response
#
# @app.options("/{full_path:path}")
# async def preflight_handler(full_path: str, response: Response):
#     response.headers["Access-Control-Allow-Origin"] = "http://localhost:5174"
#     response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
#     response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
#     return response
#
# # In-memory charging station data (Bangalore region)
# charging_stations = [
#     {
#         "id": 1,
#         "name": "Public Station A",
#         "lat": 12.9716,
#         "lon": 77.5946,
#         "status": "available",
#         "cost": "$5",
#         "type": "public"
#     },
#     {
#         "id": 2,
#         "name": "Home Charger B",
#         "lat": 12.9352,
#         "lon": 77.6245,
#         "status": "available",
#         "cost": "$3",
#         "type": "home"
#     }
# ]
#
# # Pydantic models for request payloads
# class SensorData(BaseModel):
#     battery_level: float
#     latitude: float
#     longitude: float
#     speed: float = 0.0
#     elevation: float = 0.0
#     weather_condition: str = "clear"
#
# class DecisionData(BaseModel):
#     battery_level: float
#     latitude: float
#     longitude: float
#     speed: float = 0.0
#     elevation: float = 0.0
#     weather_condition: str = "clear"
#
# # Utility: Haversine formula (in km)
# def haversine_distance(lat1, lon1, lat2, lon2):
#     lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1
#     a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
#     c = 2 * math.asin(math.sqrt(a))
#     r = 6371  # Earth radius in km
#     return c * r
#
# # Utility: Generate a simple linear route between two coordinates
# def generate_route(start, end, num_points=10):
#     lat1, lon1 = start
#     lat2, lon2 = end
#     route = []
#     for i in range(num_points + 1):
#         fraction = i / num_points
#         lat = lat1 + (lat2 - lat1) * fraction
#         lon = lon1 + (lon2 - lon1) * fraction
#         route.append({"lat": lat, "lon": lon})
#     return route
#
# # Endpoint: Simulate sensor data
# @app.post("/sensor-data")
# def sensor_data(data: SensorData):
#     return {
#         "battery": data.battery_level,
#         "lat": data.latitude,
#         "lon": data.longitude,
#         "speed": data.speed,
#         "elevation": data.elevation,
#         "weather_condition": data.weather_condition
#     }
#
# # Endpoint: Retrieve charging stations.
# # If latitude and longitude are provided, calculate distance for each station.
# @app.get("/charging-stations")
# def get_charging_stations(latitude: float = None, longitude: float = None):
#     if latitude is not None and longitude is not None:
#         for station in charging_stations:
#             station["distance"] = round(haversine_distance(latitude, longitude, station["lat"], station["lon"]), 2)
#         sorted_stations = sorted(charging_stations, key=lambda x: x.get("distance", float("inf")))
#         return sorted_stations
#     return charging_stations
#
# # Endpoint: Decision logic based on battery level.
# @app.post("/decision")
# def decision(data: DecisionData):
#     if data.battery_level < 30:
#         nearest_station = None
#         min_distance = float("inf")
#         for station in charging_stations:
#             distance = haversine_distance(data.latitude, data.longitude, station["lat"], station["lon"])
#             if distance < min_distance:
#                 min_distance = distance
#                 nearest_station = station
#
#         if nearest_station:
#             route = generate_route((data.latitude, data.longitude), (nearest_station["lat"], nearest_station["lon"]))
#             return {
#                 "alert": True,
#                 "message": f"Battery low! Nearest station: {nearest_station['name']} at {round(min_distance, 2)} km.",
#                 "route": route,
#                 "station": nearest_station
#             }
#         else:
#             raise HTTPException(status_code=404, detail="No charging station found")
#     else:
#         return {"alert": False, "message": "Battery level is sufficient."}
#
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
from typing import Optional, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor

app = FastAPI(title="Range Guardian API (Complex ML)", version="2.0")

# Enable CORS (for development, allow all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # In production, restrict this list.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory charging station data (example: Bangalore region)
charging_stations = [
    {
        "id": 1,
        "name": "Public Station A",
        "lat": 12.9716,
        "lon": 77.5946,
        "status": "available",
        "cost": "$5",
        "type": "public"
    },
    {
        "id": 2,
        "name": "Home Charger B",
        "lat": 12.9352,
        "lon": 77.6245,
        "status": "available",
        "cost": "$3",
        "type": "home"
    }
]


# ------------------ Pydantic Models ------------------

# Extended sensor data now includes extra features for ML.
class SensorData(BaseModel):
    battery_level: float
    latitude: float
    longitude: float
    speed: float = 0.0  # km/h
    elevation: float = 0.0  # meters
    ambient_temperature: float = 25.0  # Celsius
    wind_speed: float = 0.0  # km/h
    road_gradient: float = 0.0  # percentage or degree of incline


# We use the same model for decision since it comes from sensor data.
class DecisionData(SensorData):
    estimated_range: Optional[float] = Field(
        default=None, description="Estimated range in kilometers based on current battery and consumption rate"
    )
    consumption_rate: Optional[float] = Field(
        default=None, description="Predicted battery consumption rate (% per km)"
    )
    alert: Optional[bool] = Field(
        default=None, description="Indicates whether a charging alert is required"
    )
    message: Optional[str] = Field(
        default=None, description="Decision message with additional details"
    )
    station: Optional[Dict] = Field(
        default=None, description="Details of the nearest charging station"
    )
    route: Optional[List[Dict[str, float]]] = Field(
        default=None, description="Recommended route as a list of coordinate dicts"
    )


# For convenience, you can also create a dedicated model for consumption prediction
class ConsumptionInput(BaseModel):
    speed: float
    elevation: float
    ambient_temperature: float
    wind_speed: float
    road_gradient: float


# ------------------ Improved Machine Learning Model ------------------

def train_improved_model():
    np.random.seed(42)
    num_samples = 1000
    # Generate synthetic features:
    speed = np.random.uniform(10, 50, num_samples)  # km/h
    elevation = np.random.uniform(0, 200, num_samples)  # meters
    ambient_temperature = np.random.uniform(15, 35, num_samples)  # Celsius
    wind_speed = np.random.uniform(0, 15, num_samples)  # km/h
    road_gradient = np.random.uniform(-5, 5, num_samples)  # % incline (can be negative)

    # Define a non-linear synthetic relationship for consumption rate (% per km)
    # Example relationship (plus noise):
    # consumption_rate = 0.04 * speed + 0.002 * (elevation^1.1) + 0.03 * road_gradient +
    #                    0.01 * wind_speed + 0.005 * (ambient_temperature - 25)^2 + noise
    noise = np.random.normal(0, 0.01, num_samples)
    consumption_rate = (
            0.04 * speed +
            0.002 * (elevation ** 1.1) +
            0.03 * road_gradient +
            0.01 * wind_speed +
            0.005 * ((ambient_temperature - 25) ** 2) +
            noise
    )

    # Features matrix X and target y
    X = np.column_stack((speed, elevation, ambient_temperature, wind_speed, road_gradient))
    y = consumption_rate
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model


# Train the improved model at startup
ml_model = train_improved_model()


def predict_consumption(speed: float, elevation: float, ambient_temperature: float, wind_speed: float,
                        road_gradient: float) -> float:
    X_new = np.array([[speed, elevation, ambient_temperature, wind_speed, road_gradient]])
    predicted = ml_model.predict(X_new)[0]
    return predicted


# ------------------ Utility Functions ------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on the Earth."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Earth radius in km
    return c * r
@app.middleware("http")
async def log_requests(request, call_next):
    print(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    print(f"Outgoing response: {response.status_code}")
    return response

def generate_route(start, end, num_points=10):
    """Generate a simple linear route between start and end points."""
    lat1, lon1 = start
    lat2, lon2 = end
    route = []
    for i in range(num_points + 1):
        fraction = i / num_points
        lat = lat1 + (lat2 - lat1) * fraction
        lon = lon1 + (lon2 - lon1) * fraction
        route.append({"lat": lat, "lon": lon})
    return route


# ------------------ API Endpoints ------------------

@app.post("/sensor-data")
def sensor_data(data: SensorData):
    """Return sensor data (echo for now)"""
    return data.dict()


@app.get("/charging-stations")
def get_charging_stations(latitude: float = None, longitude: float = None):
    """Return charging stations; if coordinates are given, include distance calculations."""
    if latitude is not None and longitude is not None:
        for station in charging_stations:
            station["distance"] = round(haversine_distance(latitude, longitude, station["lat"], station["lon"]), 2)
        sorted_stations = sorted(charging_stations, key=lambda x: x.get("distance", float("inf")))
        return sorted_stations
    return charging_stations


@app.post("/predict-consumption")
def predict_consumption_endpoint(data: ConsumptionInput):
    """Return the predicted battery consumption rate (% per km) based on extended features."""
    predicted_rate = predict_consumption(data.speed, data.elevation, data.ambient_temperature, data.wind_speed,
                                         data.road_gradient)
    return {"predicted_consumption_rate": predicted_rate}


@app.post("/decision")
def decision(data: DecisionData):
    """
    Use the improved ML model to predict consumption rate,
    estimate remaining range, and determine if a charging alert is needed.
    """
    consumption_rate = predict_consumption(data.speed, data.elevation, data.ambient_temperature, data.wind_speed,
                                           data.road_gradient)
    if consumption_rate <= 0:
        raise HTTPException(status_code=500, detail="Invalid consumption rate predicted")

    # Estimate remaining range (km) based on current battery level (assuming battery_level in %)
    estimated_range = data.battery_level / consumption_rate

    # Find the nearest charging station from the provided coordinates
    nearest_station = None
    min_distance = float("inf")
    for station in charging_stations:
        distance = haversine_distance(data.latitude, data.longitude, station["lat"], station["lon"])
        if distance < min_distance:
            min_distance = distance
            nearest_station = station

    # Decide if an alert is needed if battery level is below 30% or if estimated range is less than the distance to the nearest station.
    if data.battery_level < 30 or estimated_range < min_distance:
        if nearest_station:
            route = generate_route((data.latitude, data.longitude), (nearest_station["lat"], nearest_station["lon"]))
            alert_message = (
                f"Battery low! Estimated range: {estimated_range:.2f} km. "
                f"Nearest station: {nearest_station['name']} at {round(min_distance, 2)} km."
            )
            return {
                "alert": True,
                "message": alert_message,
                "route": route,
                "station": nearest_station,
                "estimated_range": estimated_range,
                "consumption_rate": consumption_rate
            }
        else:
            raise HTTPException(status_code=404, detail="No charging station found")
    else:
        return {
            "alert": False,
            "message": "Battery level is sufficient.",
            "estimated_range": estimated_range,
            "consumption_rate": consumption_rate
        }

@app.middleware("http")
async def log_requests(request, call_next):
    print(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    print(f"Outgoing response: {response.status_code}")
    return response
from fastapi import Response

@app.options("/{full_path:path}")
async def preflight_handler(full_path: str, response: Response):
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:5173"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
