# ============================================================
# SMARTLOGIX AI - STREAMLIT DASHBOARD
# ============================================================

from pathlib import Path
import json
import requests
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartLogix AI",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIGURATION
# ============================================================

FASTAPI_URL = "http://127.0.0.1:8000"

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def api_get(endpoint: str):
    try:
        response = requests.get(
            f"{FASTAPI_URL}{endpoint}",
            timeout=10,
        )
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.ConnectionError:
        return None, (
            "FastAPI server is not running. "
            "Start it with:\n"
            "python -m uvicorn src.api.prediction_api:app --reload"
        )
    except requests.exceptions.RequestException as error:
        return None, str(error)


def api_post(endpoint: str, payload: dict):
    try:
        response = requests.post(
            f"{FASTAPI_URL}{endpoint}",
            json=payload,
            timeout=120,
        )

        try:
            body = response.json()
        except ValueError:
            body = {"raw_response": response.text}

        if response.status_code >= 400:
            return None, body

        return body, None

    except requests.exceptions.ConnectionError:
        return None, (
            "FastAPI server is not running. "
            "Start it with:\n"
            "python -m uvicorn src.api.prediction_api:app --reload"
        )
    except requests.exceptions.RequestException as error:
        return None, str(error)


def load_csv(filename: str):
    path = DATA_PROCESSED / filename

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


def format_value(value):
    if value is None:
        return "N/A"

    if isinstance(value, float):
        return f"{value:,.2f}"

    return str(value)


def extract_prediction_summary(result: dict):
    """
    Safely extract common Master Agent fields without assuming
    an exact response structure.
    """

    summary = {}

    summary["overall_status"] = result.get(
        "overall_status",
        result.get("status", "N/A"),
    )

    delivery = result.get("delivery_prediction", {})
    if isinstance(delivery, dict):
        summary["transport_mode"] = delivery.get(
            "predicted_transport_mode",
            delivery.get("transport_mode", "N/A"),
        )
        summary["eta_hours"] = delivery.get(
            "predicted_eta_hours",
            delivery.get("eta_hours", "N/A"),
        )

    route = result.get("route_optimization", {})
    if isinstance(route, dict):
        summary["route_id"] = route.get(
            "route_id",
            route.get("selected_route_id", "N/A"),
        )
        summary["route_distance"] = route.get(
            "distance_km",
            route.get("route_distance_km", "N/A"),
        )

    vehicle = result.get("vehicle_assignment", {})
    if isinstance(vehicle, dict):
        summary["vehicle_id"] = vehicle.get(
            "vehicle_id",
            "N/A",
        )

    maintenance = result.get("maintenance", {})
    if isinstance(maintenance, dict):
        summary["maintenance_priority"] = maintenance.get(
            "priority",
            maintenance.get("maintenance_priority", "N/A"),
        )

    return summary


# ============================================================
# HEADER
# ============================================================

st.title("🚚 SmartLogix AI")
st.caption(
    "Intelligent Multi-Modal Logistics & Autonomous Delivery Platform"
)

st.markdown(
    """
    **SmartLogix AI Dashboard**

    Integrated with the SmartLogix FastAPI backend for:
    transportation prediction, ETA prediction, route optimization,
    vehicle assignment, predictive maintenance, and AI logistics support.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ System")

if st.sidebar.button("🔄 Check FastAPI"):
    health, error = api_get("/health")

    if error:
        st.sidebar.error(error)
    else:
        st.sidebar.success("FastAPI is healthy")


st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    ### Backend
    `http://127.0.0.1:8000`

    ### API Docs
    `/docs`

    ### Main APIs
    - `/predict`
    - `/chat`
    - `/health`
    """
)


# ============================================================
# FASTAPI STATUS
# ============================================================

health, health_error = api_get("/health")

if health_error:
    st.warning(
        "⚠️ FastAPI is not reachable. Start the FastAPI server "
        "before using Prediction or Chat."
    )
else:
    st.success("🟢 FastAPI connected successfully")


# ============================================================
# LOAD LOCAL DATA FOR OVERVIEW
# ============================================================

orders_df = load_csv("orders_cleaned.csv")
customers_df = load_csv("customers_cleaned.csv")
fleet_df = load_csv("fleet_vehicles_cleaned.csv")
maintenance_df = load_csv("maintenance_history_clean.csv")


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

st.header("📊 Executive Overview")

metric_cols = st.columns(5)

with metric_cols[0]:
    st.metric(
        "Total Orders",
        f"{len(orders_df):,}" if orders_df is not None else "N/A",
    )

with metric_cols[1]:
    st.metric(
        "Customers",
        f"{len(customers_df):,}" if customers_df is not None else "N/A",
    )

with metric_cols[2]:
    st.metric(
        "Fleet Vehicles",
        f"{len(fleet_df):,}" if fleet_df is not None else "N/A",
    )

with metric_cols[3]:
    st.metric(
        "Maintenance Records",
        f"{len(maintenance_df):,}"
        if maintenance_df is not None
        else "N/A",
    )

with metric_cols[4]:
    st.metric(
        "Backend",
        "ONLINE" if health_error is None else "OFFLINE",
    )


# ============================================================
# OVERVIEW CHARTS
# ============================================================

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("📦 Order Status")

    if orders_df is not None and "order_status" in orders_df.columns:
        status_counts = (
            orders_df["order_status"]
            .astype(str)
            .value_counts()
        )
        st.bar_chart(status_counts)
    else:
        st.info("Order status data is unavailable.")


with chart_col2:
    st.subheader("🚛 Transport Mode")

    if orders_df is not None and "transport_mode" in orders_df.columns:
        mode_counts = (
            orders_df["transport_mode"]
            .astype(str)
            .value_counts()
        )
        st.bar_chart(mode_counts)
    else:
        st.info("Transport mode data is unavailable.")


# ============================================================
# DASHBOARD TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔮 Delivery Prediction",
        "🤖 AI Logistics Chatbot",
        "🔧 Fleet & Maintenance",
        "📦 Order Explorer",
    ]
)


# ============================================================
# TAB 1 - DELIVERY PREDICTION
# ============================================================

with tab1:

    st.header("🔮 Smart Delivery Prediction")

    st.write(
        "Send a delivery request to the SmartLogix Master Agent "
        "through the FastAPI `/predict` endpoint."
    )

    st.subheader("Delivery Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        quantity = st.number_input(
            "Quantity",
            min_value=1,
            value=1,
        )

        package_weight = st.number_input(
            "Package Weight (kg)",
            min_value=0.1,
            value=3.5,
            step=0.5,
        )

        delivery_priority = st.selectbox(
            "Delivery Priority",
            ["Standard", "Express", "Urgent"],
        )

        payment_mode = st.selectbox(
            "Payment Mode",
            ["card", "cash", "upi", "wallet"],
        )

    with col2:
        origin_city = st.text_input(
            "Origin City",
            "Chennai",
        )

        destination_city = st.text_input(
            "Destination City",
            "Bengaluru",
        )

        destination_state = st.text_input(
            "Destination State",
            "Karnataka",
        )

        destination_pincode = st.number_input(
            "Destination Pincode",
            min_value=100000,
            max_value=999999,
            value=560001,
        )

    with col3:
        origin_lat = st.number_input(
            "Origin Latitude",
            value=13.0827,
            format="%.4f",
        )

        origin_lon = st.number_input(
            "Origin Longitude",
            value=80.2707,
            format="%.4f",
        )

        destination_lat = st.number_input(
            "Destination Latitude",
            value=12.9716,
            format="%.4f",
        )

        destination_lon = st.number_input(
            "Destination Longitude",
            value=77.5946,
            format="%.4f",
        )

    with st.expander("Advanced Delivery Features"):

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            is_fragile = st.selectbox("Fragile", [0, 1], index=0)
            is_hazmat = st.selectbox("Hazmat", [0, 1], index=0)
            cold_chain = st.selectbox(
                "Cold Chain Required",
                [0, 1],
                index=0,
            )

        with col_b:
            dimension_length = st.number_input(
                "Length (cm)",
                value=30.0,
            )

            dimension_width = st.number_input(
                "Width (cm)",
                value=20.0,
            )

            dimension_height = st.number_input(
                "Height (cm)",
                value=15.0,
            )

        with col_c:
            traffic_speed = st.number_input(
                "Traffic Speed (km/h)",
                value=35.0,
            )

            traffic_level = st.number_input(
                "Traffic Level",
                value=0.5,
                min_value=0.0,
                max_value=1.0,
            )

            weather_condition = st.selectbox(
                "Weather",
                ["Clear", "Cloudy", "Rain", "Storm"],
            )

    st.subheader("Prediction Request")

    prediction_payload = {
        "quantity": int(quantity),
        "origin_hub": "HUB-001",
        "origin_city": origin_city,
        "destination_city": destination_city,
        "destination_state": destination_state,
        "destination_pincode": int(destination_pincode),

        "origin_lat": float(origin_lat),
        "origin_lon": float(origin_lon),
        "destination_lat": float(destination_lat),
        "destination_lon": float(destination_lon),

        "is_fragile": int(is_fragile),
        "is_hazmat": int(is_hazmat),
        "cold_chain_required": int(cold_chain),

        "dimension_length_cm": float(dimension_length),
        "dimension_width_cm": float(dimension_width),
        "dimension_height_cm": float(dimension_height),

        "package_weight_kg": float(package_weight),
        "package_volume_cm3": float(
            dimension_length
            * dimension_width
            * dimension_height
        ),
        "weight_category": (
            "Light" if package_weight <= 5
            else "Medium" if package_weight <= 20
            else "Heavy"
        ),

        "delivery_priority": delivery_priority,
        "payment_mode": payment_mode,
        "order_value_inr": 5000.0,
        "delivery_distance_km": 350.0,
        "distance_category": "Long",
        "priority_score": (
            1.0 if delivery_priority == "Urgent"
            else 0.7 if delivery_priority == "Express"
            else 0.4
        ),

        "order_year": 2026,
        "order_month": 9,
        "order_day": 24,
        "order_day_of_week": 3,
        "order_week": 39,
        "is_weekend": 0,

        "weather_condition_at_dest": weather_condition,
        "temp_celsius": 29.0,
        "humidity_%": 70.0,
        "Precipitation (mm)": (
            12.0 if weather_condition == "Rain" else 0.0
        ),
        "wind_speed_kmph": 15.0,
        "Visibility_KM": 8.0,
        "condition": weather_condition,

        "traffic_speed_kmph": float(traffic_speed),
        "traffic_level": float(traffic_level),

        "high_precipitation": int(
            weather_condition in ["Rain", "Storm"]
        ),
        "high_wind": 0,
        "low_visibility": int(
            weather_condition == "Storm"
        ),

        "transport_mode": "Truck",
    }

    with st.expander("View JSON Payload"):
        st.json(prediction_payload)

    if st.button(
        "🚀 Run SmartLogix Prediction",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "Running SmartLogix Master Agent..."
        ):

            result, error = api_post(
                "/predict",
                prediction_payload,
            )

        if error:
            st.error("Prediction failed.")

            if isinstance(error, dict):
                st.json(error)
            else:
                st.code(str(error))

        else:

            st.success("Prediction completed successfully.")

            summary = extract_prediction_summary(
                result
            )

            result_cols = st.columns(5)

            with result_cols[0]:
                st.metric(
                    "Overall Status",
                    format_value(
                        summary["overall_status"]
                    ),
                )

            with result_cols[1]:
                st.metric(
                    "Transport",
                    format_value(
                        summary["transport_mode"]
                    ),
                )

            with result_cols[2]:
                st.metric(
                    "ETA (hours)",
                    format_value(
                        summary["eta_hours"]
                    ),
                )

            with result_cols[3]:
                st.metric(
                    "Vehicle",
                    format_value(
                        summary["vehicle_id"]
                    ),
                )

            with result_cols[4]:
                st.metric(
                    "Maintenance",
                    format_value(
                        summary["maintenance_priority"]
                    ),
                )

            st.subheader("Master Agent Response")
            st.json(result)


# ============================================================
# TAB 2 - AI CHATBOT
# ============================================================

with tab2:

    st.header("🤖 SmartLogix AI Logistics Assistant")

    st.write(
        "Ask questions about orders, products, reviews, "
        "vehicle maintenance, and logistics information."
    )

    query = st.text_input(
        "Enter your question",
        placeholder=(
            "Example: Track order ORD-011545"
        ),
    )

    example_col1, example_col2, example_col3 = st.columns(3)

    with example_col1:
        if st.button("Track ORD-011545"):
            query = "Track order ORD-011545"

    with example_col2:
        if st.button("Maintenance VEH-0650"):
            query = (
                "Show maintenance information for VEH-0650"
            )

    with example_col3:
        if st.button("Available Products"):
            query = "What products are available?"

    if st.button(
        "💬 Ask SmartLogix",
        type="primary",
        use_container_width=True,
    ):

        if not query.strip():
            st.warning("Please enter a question.")
        else:

            with st.spinner(
                "SmartLogix Assistant is processing..."
            ):

                result, error = api_post(
                    "/chat",
                    {"query": query.strip()},
                )

            if error:
                st.error("Chat request failed.")

                if isinstance(error, dict):
                    st.json(error)
                else:
                    st.code(str(error))

            else:

                if result.get("status") == "success":

                    st.success(
                        f"Intent detected: "
                        f"{result.get('intent', 'N/A')}"
                    )

                    st.subheader("Assistant Response")

                    st.markdown(
                        result.get(
                            "response",
                            "No response returned.",
                        )
                    )

                else:
                    st.error(
                        result.get(
                            "message",
                            "Unknown chatbot error.",
                        )
                    )


# ============================================================
# TAB 3 - FLEET & MAINTENANCE
# ============================================================

with tab3:

    st.header("🔧 Fleet & Maintenance")

    if fleet_df is not None:

        st.subheader("Fleet Overview")

        fleet_cols = st.columns(3)

        with fleet_cols[0]:
            st.metric(
                "Vehicles",
                f"{len(fleet_df):,}",
            )

        with fleet_cols[1]:
            if "transport_mode" in fleet_df.columns:
                st.metric(
                    "Transport Modes",
                    fleet_df["transport_mode"]
                    .nunique(),
                )
            else:
                st.metric(
                    "Transport Modes",
                    "N/A",
                )

        with fleet_cols[2]:
            if "status" in fleet_df.columns:
                st.metric(
                    "Active Vehicles",
                    int(
                        (
                            fleet_df["status"]
                            .astype(str)
                            .str.lower()
                            == "active"
                        ).sum()
                    ),
                )
            else:
                st.metric(
                    "Active Vehicles",
                    "N/A",
                )

        st.dataframe(
            fleet_df.head(100),
            use_container_width=True,
            height=350,
        )

    else:
        st.info(
            "fleet_vehicles_cleaned.csv was not found."
        )

    st.subheader("Maintenance Records")

    if maintenance_df is not None:

        maintenance_cols = st.columns(3)

        with maintenance_cols[0]:
            st.metric(
                "Records",
                f"{len(maintenance_df):,}",
            )

        with maintenance_cols[1]:
            if "cost_inr" in maintenance_df.columns:
                st.metric(
                    "Avg Maintenance Cost",
                    f"₹{maintenance_df['cost_inr'].mean():,.2f}",
                )
            else:
                st.metric(
                    "Avg Maintenance Cost",
                    "N/A",
                )

        with maintenance_cols[2]:
            if "failure_reported" in maintenance_df.columns:
                st.metric(
                    "Failure Records",
                    int(
                        (
                            maintenance_df[
                                "failure_reported"
                            ]
                            .astype(str)
                            .str.lower()
                            == "yes"
                        ).sum()
                    ),
                )
            else:
                st.metric(
                    "Failure Records",
                    "N/A",
                )

        st.dataframe(
            maintenance_df.head(100),
            use_container_width=True,
            height=350,
        )

    else:
        st.info(
            "maintenance_history_clean.csv was not found."
        )


# ============================================================
# TAB 4 - ORDER EXPLORER
# ============================================================

with tab4:

    st.header("📦 Order Explorer")

    if orders_df is None:

        st.warning(
            "orders_cleaned.csv was not found."
        )

    else:

        search_order = st.text_input(
            "Search Order ID",
            placeholder="Example: ORD-011545",
        )

        if search_order.strip():

            if "order_id" in orders_df.columns:

                filtered = orders_df[
                    orders_df["order_id"]
                    .astype(str)
                    .str.contains(
                        search_order.strip(),
                        case=False,
                        na=False,
                    )
                ]

                if len(filtered) > 0:
                    st.success(
                        f"{len(filtered)} matching order(s) found."
                    )

                    st.dataframe(
                        filtered,
                        use_container_width=True,
                    )
                else:
                    st.warning(
                        "No matching order found."
                    )

            else:
                st.info(
                    "The dataset does not contain "
                    "an order_id column."
                )

        else:

            st.dataframe(
                orders_df.head(100),
                use_container_width=True,
                height=400,
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "SmartLogix AI | FastAPI + Streamlit | "
    "Intelligent Logistics & Autonomous Delivery Platform"
)
