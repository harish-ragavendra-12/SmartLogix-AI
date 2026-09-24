# ============================================================
# SMARTLOGIX AI - MASTER PREDICTION + CHATBOT API
# ============================================================

import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

import uvicorn


# ============================================================
# ADD SRC TO PYTHON PATH
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

SRC_DIR = (
    PROJECT_ROOT
    / "src"
)

if str(SRC_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(SRC_DIR)
    )


# ============================================================
# IMPORT SMARTLOGIX MASTER AGENT
# ============================================================

from agents.smartlogix_agent import (
    smartlogix_agent
)


# ============================================================
# IMPORT SMARTLOGIX CHATBOT
# ============================================================

from chatbot.smartlogix_assistant import (
    SmartLogixAssistant
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="SmartLogix AI API",
    description=(
        "End-to-end AI logistics API providing "
        "transportation prediction, ETA prediction, "
        "route optimization, predictive maintenance, "
        "and AI-assisted logistics support."
    ),
    version="1.0.0"
)


# ============================================================
# INITIALIZE CHATBOT
# ============================================================

chatbot = SmartLogixAssistant()


# ============================================================
# DELIVERY PREDICTION REQUEST MODEL
# ============================================================

class DeliveryPredictionRequest(
    BaseModel
):

    model_config = ConfigDict(
        populate_by_name=True
    )

    # --------------------------------------------------------
    # ORDER INFORMATION
    # --------------------------------------------------------

    quantity: int

    origin_hub: str

    origin_city: str

    destination_city: str

    destination_state: str

    destination_pincode: int

    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    origin_lat: float

    origin_lon: float

    destination_lat: float

    destination_lon: float

    # --------------------------------------------------------
    # PACKAGE CHARACTERISTICS
    # --------------------------------------------------------

    is_fragile: int

    is_hazmat: int

    cold_chain_required: int

    dimension_length_cm: float

    dimension_width_cm: float

    dimension_height_cm: float

    package_weight_kg: float

    package_volume_cm3: float

    weight_category: str

    # --------------------------------------------------------
    # DELIVERY INFORMATION
    # --------------------------------------------------------

    delivery_priority: str

    payment_mode: str

    order_value_inr: float

    delivery_distance_km: float

    distance_category: str

    priority_score: float

    # --------------------------------------------------------
    # ORDER DATE FEATURES
    # --------------------------------------------------------

    order_year: int

    order_month: int

    order_day: int

    order_day_of_week: int

    order_week: int

    is_weekend: int

    # --------------------------------------------------------
    # WEATHER
    # --------------------------------------------------------

    weather_condition_at_dest: str

    temp_celsius: float

    humidity_percent: float = Field(
        alias="humidity_%"
    )

    precipitation_mm: float = Field(
        alias="Precipitation (mm)"
    )

    wind_speed_kmph: float

    visibility_km: float = Field(
        alias="Visibility_KM"
    )

    condition: str

    # --------------------------------------------------------
    # TRAFFIC
    # --------------------------------------------------------

    traffic_speed_kmph: float

    traffic_level: float

    # --------------------------------------------------------
    # WEATHER / RISK FLAGS
    # --------------------------------------------------------

    high_precipitation: int

    high_wind: int

    low_visibility: int

    # --------------------------------------------------------
    # TRANSPORT MODE
    # --------------------------------------------------------

    transport_mode: str = "Truck"


# ============================================================
# CHATBOT REQUEST MODEL
# ============================================================

class ChatRequest(
    BaseModel
):

    query: str


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "application": "SmartLogix AI",
        "version": "1.0.0",
        "status": "running",
        "message": (
            "SmartLogix AI API is running successfully."
        ),
        "services": {
            "prediction": "/predict",
            "chatbot": "/chat",
            "health": "/health",
            "documentation": "/docs"
        }
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "SmartLogix AI API"
    }


# ============================================================
# MASTER PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict_delivery(
    request: DeliveryPredictionRequest
):

    # --------------------------------------------------------
    # CONVERT REQUEST TO DICTIONARY
    # --------------------------------------------------------

    request_data = (
        request.model_dump(
            by_alias=True
        )
    )

    # --------------------------------------------------------
    # RUN SMARTLOGIX MASTER AGENT
    # --------------------------------------------------------

    result = (
        smartlogix_agent(
            request_data
        )
    )

    # --------------------------------------------------------
    # RETURN FINAL RESULT
    # --------------------------------------------------------

    return result


# ============================================================
# CHATBOT ENDPOINT
# ============================================================

@app.post("/chat")
def chat(
    request: ChatRequest
):

    # --------------------------------------------------------
    # CLEAN USER QUERY
    # --------------------------------------------------------

    query = request.query.strip()

    # --------------------------------------------------------
    # VALIDATE QUERY
    # --------------------------------------------------------

    if not query:

        return {
            "status": "error",
            "message": (
                "Query is required."
            )
        }

    # --------------------------------------------------------
    # RUN SMARTLOGIX ASSISTANT
    # --------------------------------------------------------

    try:

        result = (
            chatbot.ask(
                query
            )
        )

        # ----------------------------------------------------
        # RETURN CHATBOT RESULT
        # ----------------------------------------------------

        return {
            "status": "success",
            "query": result.get(
                "query"
            ),
            "intent": result.get(
                "intent"
            ),
            "response": result.get(
                "response"
            )
        }

    except Exception as error:

        return {
            "status": "error",
            "message": str(error)
        }


# ============================================================
# API INFORMATION
# ============================================================

@app.get("/api/info")
def api_info():

    return {
        "application": "SmartLogix AI",
        "version": "1.0.0",
        "available_services": [
            "Master Agent Prediction",
            "Transportation Prediction",
            "ETA Prediction",
            "Route Optimization",
            "Predictive Maintenance",
            "AI Logistics Chatbot"
        ],
        "endpoints": {
            "root": "/",
            "health": "/health",
            "prediction": "/predict",
            "chatbot": "/chat",
            "documentation": "/docs"
        }
    }


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False
    )