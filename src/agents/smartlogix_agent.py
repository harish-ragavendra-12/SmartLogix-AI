# ============================================================
# SMARTLOGIX AI - MASTER AGENT
# ============================================================
#
# End-to-end orchestration:
#
# Delivery Prediction
#        ↓
# Route Optimization
#        ↓
# Vehicle Assignment
#        ↓
# Payload Optimization
#        ↓
# Drone Battery Validation
#        ↓
# Delivery Scheduler
#        ↓
# Maintenance Analysis
#        ↓
# Operational Alert
#        ↓
# Final Dispatch Result
#
# ============================================================

import json
import pandas as pd


# ============================================================
# DELIVERY PREDICTION
# ============================================================

from agents.delivery_prediction_agent import (
    predict_transportation,
    predict_delivery_eta,
    load_transportation_model,
    load_eta_model,
    convert_hours_to_time
)


# ============================================================
# VEHICLE ASSIGNMENT
# ============================================================

from agents.vehicle_assignment_agent import (
    assign_vehicle
)


# ============================================================
# MAINTENANCE
# ============================================================

from agents.maintenance_agent import (
    maintenance_agent
)


# ============================================================
# ROUTE OPTIMIZATION
# ============================================================

from route_optimization.route_optimizer import (
    load_route_data,
    clean_gps_data,
    create_route_summary,
    calculate_route_match_distance,
    select_candidate_routes,
    prepare_traffic_context,
    apply_traffic_adjustment,
    calculate_route_optimization_score,
    rank_routes,
    select_best_route
)


# ============================================================
# PAYLOAD OPTIMIZATION
# ============================================================

from route_optimization.payload_optimizer import (
    evaluate_payload_constraint
)


# ============================================================
# DRONE BATTERY OPTIMIZATION
# ============================================================

from route_optimization.drone_battery_optimizer import (
    evaluate_drone_assignment
)


# ============================================================
# DELIVERY SCHEDULING
# ============================================================

from route_optimization.delivery_scheduler import (
    schedule_deliveries
)


# ============================================================
# CREATE SAMPLE ORDER
# ============================================================

def create_sample_order():

    return {
        "order_id": "ORD-SAMPLE-001",

        "quantity": 2,

        "origin_hub": "HUB-001",

        "origin_city": "Chennai",

        "destination_city": "Bengaluru",

        "destination_state": "Karnataka",

        "destination_lat": 12.9716,

        "destination_lon": 77.5946,

        "destination_pincode": 560001,

        "is_fragile": 0,

        "is_hazmat": 0,

        "cold_chain_required": 0,

        "delivery_priority": "standard",

        "payment_mode": "COD",

        "order_value_inr": 2500,

        "weather_condition_at_dest": "Clear",

        "dimension_length_cm": 30,

        "dimension_width_cm": 20,

        "dimension_height_cm": 15,

        "order_year": 2026,

        "order_month": 9,

        "order_day": 21,

        "order_day_of_week": 0,

        "order_week": 39,

        "is_weekend": 0,

        "package_weight_kg": 3.5,

        "package_volume_cm3": 9000,

        "weight_category": "Light",

        "delivery_distance_km": 350,

        "distance_category": "Medium",

        "priority_score": 2,

        "temp_celsius": 28.5,

        "humidity_%": 68,

        "Precipitation (mm)": 0,

        "wind_speed_kmph": 12.5,

        "Visibility_KM": 9.5,

        "condition": "Clear",

        "traffic_speed_kmph": 32.5,

        "traffic_level": 10.2,

        "high_precipitation": 0,

        "high_wind": 0,

        "low_visibility": 0,

        "transport_mode": "Truck",

        "origin_lat": 13.0827,

        "origin_lon": 80.2707
    }


# ============================================================
# NORMALIZE ORDER DATA
# ============================================================

def normalize_order_data(order_data):

    if isinstance(order_data, list):

        if len(order_data) == 0:

            raise ValueError(
                "order_data list is empty."
            )

        if len(order_data) == 1:

            return order_data[0]

        raise ValueError(
            "order_data list must contain exactly one order."
        )

    if isinstance(order_data, dict):

        return order_data

    raise TypeError(
        "order_data must be a dictionary "
        "or a list containing one dictionary."
    )


# ============================================================
# CREATE ORDER DATAFRAME
# ============================================================

def create_order_dataframe(order_data):

    normalized_order = normalize_order_data(
        order_data
    )

    return pd.DataFrame(
        [normalized_order]
    )


# ============================================================
# DELIVERY PREDICTION
# ============================================================

def run_delivery_prediction(order_data):

    print("\n" + "=" * 70)
    print("DELIVERY PREDICTION")
    print("=" * 70)

    order_data = normalize_order_data(
        order_data
    )

    order_df = pd.DataFrame(
        [order_data]
    )

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    transportation_model = (
        load_transportation_model()
    )

    eta_model = (
        load_eta_model()
    )

    # --------------------------------------------------------
    # TRANSPORTATION PREDICTION
    # --------------------------------------------------------

    predicted_transport_mode = (
        predict_transportation(
            order_df,
            transportation_model
        )
    )

    # --------------------------------------------------------
    # ETA PREDICTION
    # --------------------------------------------------------

    eta_hours = (
        predict_delivery_eta(
            order_df,
            predicted_transport_mode,
            eta_model
        )
    )

    eta_hours = round(
        float(eta_hours),
        2
    )

    eta_minutes = round(
        eta_hours * 60
    )

    eta_duration = (
        convert_hours_to_time(
            eta_hours
        )
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print(
        f"\nPredicted Transport Mode : "
        f"{predicted_transport_mode}"
    )

    print(
        f"Predicted ETA            : "
        f"{eta_hours:.2f} hours"
    )

    print(
        f"ETA Duration             : "
        f"{eta_duration}"
    )

    return {

        "status": "success",

        "transport_mode": (
            predicted_transport_mode
        ),

        "predicted_eta_hours": (
            eta_hours
        ),

        "predicted_eta_minutes": (
            eta_minutes
        ),

        "eta_duration": (
            eta_duration
        )
    }


# ============================================================
# ROUTE OPTIMIZATION
# ============================================================

def run_route_optimization(order_data):

    print("\n" + "=" * 70)
    print("ROUTE OPTIMIZATION")
    print("=" * 70)

    try:

        order_data = normalize_order_data(
            order_data
        )

        # ----------------------------------------------------
        # LOAD ROUTE DATA
        # ----------------------------------------------------

        gps_df, traffic_df = (
            load_route_data()
        )

        # ----------------------------------------------------
        # CLEAN GPS DATA
        # ----------------------------------------------------

        gps_df = clean_gps_data(
            gps_df
        )

        # ----------------------------------------------------
        # CREATE ROUTE SUMMARY
        # ----------------------------------------------------

        route_summary = (
            create_route_summary(
                gps_df
            )
        )

        # ----------------------------------------------------
        # ORIGIN / DESTINATION
        # ----------------------------------------------------

        origin_lat = float(
            order_data["origin_lat"]
        )

        origin_lon = float(
            order_data["origin_lon"]
        )

        destination_lat = float(
            order_data["destination_lat"]
        )

        destination_lon = float(
            order_data["destination_lon"]
        )

        print("\nOrigin:")
        print(
            f"Latitude  : {origin_lat}"
        )
        print(
            f"Longitude : {origin_lon}"
        )

        print("\nDestination:")
        print(
            f"Latitude  : {destination_lat}"
        )
        print(
            f"Longitude : {destination_lon}"
        )

        # ----------------------------------------------------
        # ROUTE MATCH DISTANCE
        # ----------------------------------------------------

        route_summary = (
            calculate_route_match_distance(
                route_summary,
                origin_lat,
                origin_lon,
                destination_lat,
                destination_lon
            )
        )

        # ----------------------------------------------------
        # CANDIDATE ROUTES
        # ----------------------------------------------------

        candidates = (
            select_candidate_routes(
                route_summary,
                origin_lat,
                origin_lon,
                destination_lat,
                destination_lon
            )
        )

        # ----------------------------------------------------
        # TRAFFIC CONTEXT
        # ----------------------------------------------------

        traffic_context = (
            prepare_traffic_context(
                traffic_df
            )
        )

        # ----------------------------------------------------
        # TRAFFIC ADJUSTMENT
        # ----------------------------------------------------

        candidates = (
            apply_traffic_adjustment(
                candidates,
                traffic_context
            )
        )

        # ----------------------------------------------------
        # OPTIMIZATION SCORE
        # ----------------------------------------------------

        candidates = (
            calculate_route_optimization_score(
                candidates
            )
        )

        # ----------------------------------------------------
        # RANK
        # ----------------------------------------------------

        ranked_routes = (
            rank_routes(
                candidates
            )
        )

        # ----------------------------------------------------
        # SELECT BEST ROUTE
        # ----------------------------------------------------

        best_route = (
            select_best_route(
                ranked_routes
            )
        )

        if best_route is None:

            return {
                "status": "failed",
                "message": (
                    "No suitable route was found."
                )
            }

        # ----------------------------------------------------
        # EXTRACT ROUTE INFORMATION
        # ----------------------------------------------------

        route_id = best_route.get(
            "route_id"
        )

        vehicle_id = best_route.get(
            "vehicle_id"
        )

        route_match_distance = best_route.get(
            "route_match_distance_km"
        )

        route_distance = best_route.get(
            "distance_km"
        )

        average_speed = best_route.get(
            "average_speed_kmph"
        )

        estimated_travel_time = best_route.get(
            "estimated_time_hours"
        )

        traffic_adjusted_time = best_route.get(
            "traffic_adjusted_time_hours"
        )

        optimization_score = best_route.get(
            "route_optimization_score"
        )

        route_match_status = best_route.get(
            "route_match_status",
            "UNKNOWN"
        )

        # ----------------------------------------------------
        # SAFE VALUES
        # ----------------------------------------------------

        route_match_distance = (
            float(route_match_distance)
            if route_match_distance is not None
            else None
        )

        route_distance = (
            float(route_distance)
            if route_distance is not None
            else None
        )

        average_speed = (
            float(average_speed)
            if average_speed is not None
            else None
        )

        estimated_travel_time = (
            float(estimated_travel_time)
            if estimated_travel_time is not None
            else None
        )

        traffic_adjusted_time = (
            float(traffic_adjusted_time)
            if traffic_adjusted_time is not None
            else None
        )

        optimization_score = (
            float(optimization_score)
            if optimization_score is not None
            else None
        )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        print("\nSelected Route:")

        print(
            f"Route ID                 : "
            f"{route_id}"
        )

        print(
            f"Vehicle ID               : "
            f"{vehicle_id}"
        )

        print(
            f"Route Match Status       : "
            f"{route_match_status}"
        )

        if route_match_distance is not None:

            print(
                f"Route Match Distance     : "
                f"{route_match_distance:.3f} km"
            )

        if route_distance is not None:

            print(
                f"Route Distance           : "
                f"{route_distance:.3f} km"
            )

        if average_speed is not None:

            print(
                f"Average Speed            : "
                f"{average_speed:.2f} km/h"
            )

        if estimated_travel_time is not None:

            print(
                f"Estimated Travel Time    : "
                f"{estimated_travel_time:.3f} hours"
            )

        if traffic_adjusted_time is not None:

            print(
                f"Traffic Adjusted Time    : "
                f"{traffic_adjusted_time:.3f} hours"
            )

        if optimization_score is not None:

            print(
                f"Optimization Score       : "
                f"{optimization_score:.3f}"
            )

        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        return {

            "status": "success",

            "route_id": route_id,

            "vehicle_id": vehicle_id,

            "route_match_status": (
                route_match_status
            ),

            "route_match_distance_km": (
                round(
                    route_match_distance,
                    3
                )
                if route_match_distance is not None
                else None
            ),

            "route_distance_km": (
                round(
                    route_distance,
                    3
                )
                if route_distance is not None
                else None
            ),

            "average_speed_kmph": (
                round(
                    average_speed,
                    2
                )
                if average_speed is not None
                else None
            ),

            "estimated_travel_time_hours": (
                round(
                    estimated_travel_time,
                    3
                )
                if estimated_travel_time is not None
                else None
            ),

            "traffic_adjusted_time_hours": (
                round(
                    traffic_adjusted_time,
                    3
                )
                if traffic_adjusted_time is not None
                else None
            ),

            "optimization_score": (
                round(
                    optimization_score,
                    3
                )
                if optimization_score is not None
                else None
            )
        }

    except Exception as e:

        print(
            f"\nRoute optimization failed: "
            f"{e}"
        )

        return {
            "status": "failed",
            "message": str(e)
        }


# ============================================================
# VEHICLE ASSIGNMENT
# ============================================================

def run_vehicle_assignment(
    order_data,
    route_result,
    delivery_result
):

    print("\n" + "=" * 70)
    print("VEHICLE ASSIGNMENT")
    print("=" * 70)

    try:

        order = dict(
            normalize_order_data(
                order_data
            )
        )

        # ----------------------------------------------------
        # USE ML PREDICTED TRANSPORT MODE
        # ----------------------------------------------------

        predicted_mode = (
            delivery_result.get(
                "transport_mode"
            )
            if isinstance(
                delivery_result,
                dict
            )
            else None
        )

        if predicted_mode:

            order["transport_mode"] = (
                predicted_mode
            )

        # ----------------------------------------------------
        # HISTORICAL ROUTE VEHICLE
        #
        # If the route optimizer has a real historical vehicle,
        # ask Vehicle Assignment Agent to validate it.
        #
        # If the route optimizer returned UNASSIGNED because
        # direct geospatial fallback was used, let the fleet
        # assignment engine choose a vehicle.
        # ----------------------------------------------------

        preferred_vehicle_id = None

        if isinstance(
            route_result,
            dict
        ):

            route_vehicle_id = (
                route_result.get(
                    "vehicle_id"
                )
            )

            if (
                route_vehicle_id
                and str(route_vehicle_id).upper()
                != "UNASSIGNED"
            ):

                preferred_vehicle_id = (
                    route_vehicle_id
                )

        # ----------------------------------------------------
        # ASSIGN VEHICLE
        # ----------------------------------------------------

        result = assign_vehicle(
            order,
            preferred_vehicle_id=preferred_vehicle_id
        )

        if not isinstance(
            result,
            dict
        ):

            return {
                "status": "failed",
                "message": (
                    "Vehicle assignment returned "
                    "an invalid result."
                )
            }

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        print(
            f"\nAssignment Status       : "
            f"{result.get('status')}"
        )

        print(
            f"Vehicle ID              : "
            f"{result.get('vehicle_id')}"
        )

        print(
            f"Assignment Source       : "
            f"{result.get('assignment_source')}"
        )

        print(
            f"Transport Mode          : "
            f"{result.get('transport_mode')}"
        )

        print(
            f"Availability            : "
            f"{result.get('availability_status')}"
        )

        print(
            f"Capacity                : "
            f"{result.get('capacity_kg')} kg"
        )

        print(
            f"Package Weight          : "
            f"{result.get('package_weight_kg')} kg"
        )

        print(
            f"Capacity Utilization    : "
            f"{result.get('capacity_utilization_pct')}%"
        )

        print(
            f"Capacity Status         : "
            f"{result.get('capacity_status')}"
        )

        print(
            f"Transport Verified      : "
            f"{result.get('transport_mode_verified')}"
        )

        print(
            f"Assignment Score        : "
            f"{result.get('assignment_score')}"
        )

        return result

    except Exception as e:

        print(
            f"\nVehicle assignment failed: "
            f"{e}"
        )

        return {
            "status": "failed",
            "message": str(e)
        }


# ============================================================
# PAYLOAD OPTIMIZATION
# ============================================================

def run_payload_optimization(
    order_data,
    vehicle_assignment
):

    print("\n" + "=" * 70)
    print("PAYLOAD OPTIMIZATION")
    print("=" * 70)

    try:

        order = normalize_order_data(
            order_data
        )

        vehicle_id = (
            vehicle_assignment.get(
                "vehicle_id"
            )
        )

        transport_mode = (
            vehicle_assignment.get(
                "transport_mode"
            )
        )

        capacity = (
            vehicle_assignment.get(
                "capacity_kg"
            )
        )

        vehicle = {

            "vehicle_id": (
                vehicle_id
            ),

            "vehicle_type": (
                transport_mode
            ),

            "transport_mode": (
                transport_mode
            ),

            "payload_capacity_kg": (
                capacity
            )
        }

        result = (
            evaluate_payload_constraint(
                order,
                vehicle
            )
        )

        # ----------------------------------------------------
        # ADD VEHICLE INFORMATION
        # ----------------------------------------------------

        if isinstance(
            result,
            dict
        ):

            result[
                "vehicle_id"
            ] = vehicle_id

            result[
                "transport_mode"
            ] = transport_mode

            # ------------------------------------------------
            # MORE PRECISE UTILIZATION
            # ------------------------------------------------

            package_weight = result.get(
                "package_weight_kg"
            )

            capacity_kg = result.get(
                "capacity_kg"
            )

            if (
                package_weight is not None
                and capacity_kg
                and float(capacity_kg) > 0
            ):

                utilization_pct = (
                    float(package_weight)
                    / float(capacity_kg)
                    * 100
                )

                result[
                    "capacity_utilization_pct"
                ] = round(
                    utilization_pct,
                    2
                )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        print(
            f"\nPayload Status          : "
            f"{result.get('status')}"
        )

        print(
            f"Vehicle ID              : "
            f"{result.get('vehicle_id')}"
        )

        print(
            f"Package Weight          : "
            f"{result.get('package_weight_kg')} kg"
        )

        print(
            f"Vehicle Capacity        : "
            f"{result.get('capacity_kg')} kg"
        )

        print(
            f"Capacity Utilization    : "
            f"{result.get('capacity_utilization_pct')}%"
        )

        print(
            f"Utilization Level       : "
            f"{result.get('utilization_level')}"
        )

        violations = (
            result.get(
                "violations",
                []
            )
        )

        if violations:

            print(
                "\nPayload Violations:"
            )

            for violation in violations:

                print(
                    f"- {violation}"
                )

        return result

    except Exception as e:

        print(
            f"\nPayload optimization failed: "
            f"{e}"
        )

        return {
            "status": "failed",
            "message": str(e)
        }


# ============================================================
# DRONE BATTERY VALIDATION
# ============================================================

def run_drone_battery_validation(
    order_data,
    vehicle_assignment,
    payload_result
):

    print("\n" + "=" * 70)
    print("DRONE BATTERY VALIDATION")
    print("=" * 70)

    try:

        order = normalize_order_data(
            order_data
        )

        # ----------------------------------------------------
        # ACTUAL ASSIGNED TRANSPORT MODE
        # ----------------------------------------------------

        transport_mode = str(
            vehicle_assignment.get(
                "transport_mode",
                ""
            )
        ).strip().lower()

        # ----------------------------------------------------
        # NOT A DRONE
        # ----------------------------------------------------

        if transport_mode != "drone":

            result = {

                "status": "NOT_APPLICABLE",

                "message": (
                    "Drone battery validation is only "
                    "required for drone deliveries."
                )
            }

            print(
                "\nDrone Battery Status    : "
                "NOT_APPLICABLE"
            )

            print(
                "Message                 : "
                f"{result['message']}"
            )

            return result

        # ----------------------------------------------------
        # PAYLOAD MUST BE ELIGIBLE
        # ----------------------------------------------------

        payload_status = (
            payload_result.get(
                "status"
            )
            if isinstance(
                payload_result,
                dict
            )
            else None
        )

        if payload_status != "ELIGIBLE":

            result = {

                "status": "REJECTED",

                "reason": (
                    "Drone battery validation "
                    "cannot proceed because "
                    "payload validation failed."
                )
            }

            print(
                "\nDrone Battery Status    : "
                "REJECTED"
            )

            print(
                f"Reason                  : "
                f"{result['reason']}"
            )

            return result

        # ----------------------------------------------------
        # INPUTS
        # ----------------------------------------------------

        origin_lat = float(
            order["origin_lat"]
        )

        origin_lon = float(
            order["origin_lon"]
        )

        destination_lat = float(
            order["destination_lat"]
        )

        destination_lon = float(
            order["destination_lon"]
        )

        payload_kg = float(
            order.get(
                "package_weight_kg",
                0.0
            )
            or 0.0
        )

        wind_speed_kmph = float(
            order.get(
                "wind_speed_kmph",
                0.0
            )
            or 0.0
        )

        # ----------------------------------------------------
        # DRONE BATTERY ENGINE
        # ----------------------------------------------------

        result = (
            evaluate_drone_assignment(
                origin_lat=origin_lat,
                origin_lon=origin_lon,
                destination_lat=destination_lat,
                destination_lon=destination_lon,
                payload_kg=payload_kg,
                wind_speed_kmph=wind_speed_kmph
            )
        )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        print(
            f"\nDrone Battery Status    : "
            f"{result.get('status')}"
        )

        print(
            f"One-Way Distance        : "
            f"{result.get('one_way_distance_km')} km"
        )

        print(
            f"Round-Trip Distance     : "
            f"{result.get('round_trip_distance_km')} km"
        )

        print(
            f"Payload                 : "
            f"{result.get('payload_kg')} kg"
        )

        print(
            f"Available Battery       : "
            f"{result.get('available_battery_kwh')} kWh"
        )

        print(
            f"Usable Energy           : "
            f"{result.get('usable_energy_kwh')} kWh"
        )

        print(
            f"Required Energy         : "
            f"{result.get('required_energy_kwh')} kWh"
        )

        print(
            f"Energy Margin           : "
            f"{result.get('energy_margin_kwh')} kWh"
        )

        print(
            f"Estimated Flight Time   : "
            f"{result.get('estimated_flight_hours')} hours"
        )

        return result

    except Exception as e:

        print(
            f"\nDrone battery validation failed: "
            f"{e}"
        )

        return {

            "status": "failed",

            "message": str(e)
        }


# ============================================================
# DELIVERY SCHEDULER
# ============================================================

def run_delivery_scheduler(
    order_data,
    route_result
):

    print("\n" + "=" * 70)
    print("DELIVERY SCHEDULER")
    print("=" * 70)

    try:

        order = normalize_order_data(
            order_data
        )

        # ----------------------------------------------------
        # USE ROUTE TRAVEL TIME
        # ----------------------------------------------------

        travel_time = float(
            route_result.get(
                "traffic_adjusted_time_hours",
                route_result.get(
                    "estimated_travel_time_hours",
                    0.0
                )
            )
            or 0.0
        )

        schedule_order = {

            "order_id": (
                order.get(
                    "order_id",
                    "ORD-SAMPLE-001"
                )
            ),

            "delivery_priority": (
                order.get(
                    "delivery_priority",
                    "standard"
                )
            ),

            "estimated_travel_time_hours": (
                travel_time
            )
        }

        result = schedule_deliveries(
            [schedule_order]
        )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        print(
            f"\nScheduler Status        : "
            f"{result.get('status')}"
        )

        print(
            f"Delivery Count          : "
            f"{result.get('delivery_count')}"
        )

        for item in result.get(
            "schedule",
            []
        ):

            print(
                f"\nSequence                : "
                f"{item.get('sequence')}"
            )

            print(
                f"Order ID                : "
                f"{item.get('order_id')}"
            )

            print(
                f"Priority                : "
                f"{item.get('priority')}"
            )

            print(
                f"Planned Departure       : "
                f"{item.get('planned_departure')}"
            )

            print(
                f"Planned Arrival         : "
                f"{item.get('planned_arrival')}"
            )

            print(
                f"Travel Time             : "
                f"{item.get('travel_time_hours')} hours"
            )

        return result

    except Exception as e:

        print(
            f"\nDelivery scheduling failed: "
            f"{e}"
        )

        return {
            "status": "failed",
            "message": str(e)
        }


# ============================================================
# MAINTENANCE ANALYSIS
# ============================================================

def run_maintenance_analysis(
    vehicle_id
):

    print("\n" + "=" * 70)
    print("MAINTENANCE ANALYSIS")
    print("=" * 70)

    try:

        if not vehicle_id:

            return {

                "status": "failed",

                "message": (
                    "No vehicle ID was provided "
                    "for maintenance analysis."
                )
            }

        maintenance_result = (
            maintenance_agent(
                vehicle_id
            )
        )

        # ----------------------------------------------------
        # VEHICLE ID SAFETY
        # ----------------------------------------------------

        if (
            isinstance(
                maintenance_result,
                dict
            )
            and
            "vehicle_id"
            not in maintenance_result
        ):

            maintenance_result[
                "vehicle_id"
            ] = vehicle_id

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        if (
                isinstance(
                    maintenance_result,
                    dict
                )
                and
                maintenance_result.get("status") == "success"
        ):

            evidence = maintenance_result.get(
                "evidence",
                {}
            )

            maintenance_history = (
                maintenance_result.get(
                    "maintenance_history",
                    {}
                )
            )

            predictive = (
                maintenance_result.get(
                    "predictive_maintenance",
                    {}
                )
            )

            print(
                f"\nVehicle ID              : "
                f"{vehicle_id}"
            )

            print(
                f"History Level           : "
                f"{evidence.get('history_level')}"
            )

            print(
                f"Evidence Level          : "
                f"{evidence.get('evidence_level')}"
            )

            print(
                f"Maintenance Count       : "
                f"{maintenance_history.get('maintenance_count')}"
            )

            print(
                f"Failure Count           : "
                f"{maintenance_history.get('failure_count')}"
            )

            print(
                f"Failure Rate            : "
                f"{maintenance_history.get('failure_rate')}"
            )

            print(
                f"Risk Score              : "
                f"{predictive.get('risk_score')}"
            )

            print(
                f"Risk Level              : "
                f"{predictive.get('risk_level')}"
            )

            print(
                f"Priority                : "
                f"{predictive.get('priority')}"
            )

        elif (
            isinstance(
                maintenance_result,
                dict
            )
            and
            maintenance_result.get(
                "status"
            ) == "not_found"
        ):

            print(
                f"\nNo maintenance information "
                f"found for vehicle: "
                f"{vehicle_id}"
            )

        return maintenance_result

    except Exception as e:

        print(
            f"\nMaintenance analysis failed: "
            f"{e}"
        )

        return {

            "status": "failed",

            "message": str(e),

            "vehicle_id": vehicle_id
        }


# ============================================================
# CREATE OPERATIONAL ALERT
# ============================================================

def create_operational_alert(
    maintenance_result,
    route_result,
    payload_result,
    drone_result
):

    alerts = []

    # --------------------------------------------------------
    # ROUTE ALERT
    # --------------------------------------------------------

    if isinstance(
        route_result,
        dict
    ):

        route_match_status = (
            route_result.get(
                "route_match_status"
            )
        )

        if route_match_status == (
            "DIRECT_GEODESIC_FALLBACK"
        ):

            alerts.append(
                "Direct geospatial route fallback "
                "was used. Validate the route with "
                "a road-network routing service "
                "before dispatch."
            )

        elif route_match_status == (
            "RELAXED_MATCH"
        ):

            alerts.append(
                "A relaxed historical route match "
                "was used. Validate route suitability "
                "before dispatch."
            )

    # --------------------------------------------------------
    # PAYLOAD ALERT
    # --------------------------------------------------------

    if isinstance(
        payload_result,
        dict
    ):

        if payload_result.get(
            "status"
        ) == "REJECTED":

            alerts.append(
                "Payload constraints failed. "
                "Vehicle cannot safely carry "
                "the assigned order."
            )

    # --------------------------------------------------------
    # DRONE ALERT
    # --------------------------------------------------------

    if isinstance(
        drone_result,
        dict
    ):

        drone_status = (
            drone_result.get(
                "status"
            )
        )

        if drone_status == "REJECTED":

            alerts.append(
                "Drone battery constraints failed. "
                "The drone should not be dispatched "
                "for this delivery."
            )

        elif drone_status == "failed":

            alerts.append(
                "Drone battery validation failed. "
                "Verify drone energy availability "
                "before dispatch."
            )

    # --------------------------------------------------------
    # MAINTENANCE ALERT
    # --------------------------------------------------------

    if not isinstance(
        maintenance_result,
        dict
    ):

        alerts.append(
            "Maintenance verification "
            "is required before assignment."
        )

    else:

        maintenance_status = (
            maintenance_result.get(
                "status"
            )
        )

        if maintenance_status == "not_found":

            alerts.append(
                "Maintenance verification "
                "is required before assignment."
            )

        elif maintenance_status == "failed":

            alerts.append(
                "Maintenance analysis failed. "
                "Verify maintenance records "
                "before assignment."
            )

        elif maintenance_status == "success":

            risk = (
                maintenance_result.get(
                    "risk",
                    {}
                )
            )

            risk_level = (
                risk.get(
                    "level"
                )
            )

            priority = (
                risk.get(
                    "priority"
                )
            )

            if (
                priority == "CRITICAL"
                or
                risk_level == "HIGH"
            ):

                alerts.append(
                    "High maintenance risk detected. "
                    "Complete preventive inspection "
                    "before critical delivery assignment."
                )

            elif risk_level == "MEDIUM":

                alerts.append(
                    "Medium maintenance risk detected. "
                    "Review maintenance recommendation "
                    "before assignment."
                )

    # --------------------------------------------------------
    # FINAL ALERT
    # --------------------------------------------------------

    if not alerts:

        return (
            "No critical operational alerts."
        )

    return " ".join(
        alerts
    )


# ============================================================
# OVERALL STATUS
# ============================================================

def determine_overall_status(
    delivery_result,
    route_result,
    vehicle_assignment,
    payload_result,
    drone_result,
    scheduler_result,
    maintenance_result
):

    # --------------------------------------------------------
    # HARD FAILURES
    # --------------------------------------------------------

    hard_failure_statuses = {

        "delivery": (
            delivery_result.get("status")
            if isinstance(
                delivery_result,
                dict
            )
            else "failed"
        ),

        "route": (
            route_result.get("status")
            if isinstance(
                route_result,
                dict
            )
            else "failed"
        ),

        "vehicle": (
            vehicle_assignment.get("status")
            if isinstance(
                vehicle_assignment,
                dict
            )
            else "failed"
        ),

        "payload": (
            payload_result.get("status")
            if isinstance(
                payload_result,
                dict
            )
            else "failed"
        ),

        "scheduler": (
            scheduler_result.get("status")
            if isinstance(
                scheduler_result,
                dict
            )
            else "failed"
        )
    }

    if (
        hard_failure_statuses["delivery"]
        == "failed"
    ):

        return "FAILED"

    if (
        hard_failure_statuses["route"]
        == "failed"
    ):

        return "FAILED"

    if (
        hard_failure_statuses["vehicle"]
        != "assigned"
    ):

        return "FAILED"

    if (
        hard_failure_statuses["payload"]
        != "ELIGIBLE"
    ):

        return "FAILED"

    if (
        hard_failure_statuses["scheduler"]
        != "success"
    ):

        return "FAILED"

    # --------------------------------------------------------
    # DRONE CONSTRAINT
    # --------------------------------------------------------

    if isinstance(
        drone_result,
        dict
    ):

        drone_status = (
            drone_result.get(
                "status"
            )
        )

        if drone_status in {
            "REJECTED",
            "failed"
        }:

            return "FAILED"

    # --------------------------------------------------------
    # MAINTENANCE FAILURE
    # --------------------------------------------------------

    if (
        not isinstance(
            maintenance_result,
            dict
        )
        or
        maintenance_result.get(
            "status"
        ) != "success"
    ):

        return "PARTIAL_SUCCESS"

    # --------------------------------------------------------
    # ROUTE WARNING
    # --------------------------------------------------------

    route_match_status = (
        route_result.get(
            "route_match_status"
        )
        if isinstance(
            route_result,
            dict
        )
        else None
    )

    if route_match_status in {
        "DIRECT_GEODESIC_FALLBACK",
        "RELAXED_MATCH"
    }:

        return "SUCCESS_WITH_WARNING"

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    return "SUCCESS"


# ============================================================
# SMARTLOGIX MASTER AGENT
# ============================================================

def smartlogix_agent(
    order_data
):

    print("\n" + "=" * 70)
    print("SMARTLOGIX AI AGENT")
    print("=" * 70)

    normalized_order = (
        normalize_order_data(
            order_data
        )
    )

    # ========================================================
    # 1. DELIVERY PREDICTION
    # ========================================================

    try:

        delivery_result = (
            run_delivery_prediction(
                normalized_order
            )
        )

    except Exception as e:

        print(
            f"\nDelivery prediction failed: "
            f"{e}"
        )

        delivery_result = {

            "status": "failed",

            "message": str(e)
        }

    # ========================================================
    # 2. ROUTE OPTIMIZATION
    # ========================================================

    route_result = (
        run_route_optimization(
            normalized_order
        )
    )

    # ========================================================
    # 3. VEHICLE ASSIGNMENT
    # ========================================================

    vehicle_assignment = (
        run_vehicle_assignment(
            normalized_order,
            route_result,
            delivery_result
        )
    )

    # ========================================================
    # UPDATE ROUTE WITH ACTUAL ASSIGNED VEHICLE
    # ========================================================

    if (
        isinstance(
            route_result,
            dict
        )
        and
        isinstance(
            vehicle_assignment,
            dict
        )
    ):

        assigned_vehicle_id = (
            vehicle_assignment.get(
                "vehicle_id"
            )
        )

        if assigned_vehicle_id:

            route_result[
                "vehicle_id"
            ] = assigned_vehicle_id

            route_result[
                "vehicle_assignment_status"
            ] = vehicle_assignment.get(
                "status"
            )

            route_result[
                "vehicle_assignment_source"
            ] = vehicle_assignment.get(
                "assignment_source"
            )

    # ========================================================
    # 4. PAYLOAD OPTIMIZATION
    # ========================================================

    payload_result = (
        run_payload_optimization(
            normalized_order,
            vehicle_assignment
        )
    )

    # ========================================================
    # 5. DRONE BATTERY VALIDATION
    # ========================================================

    drone_result = (
        run_drone_battery_validation(
            normalized_order,
            vehicle_assignment,
            payload_result
        )
    )

    # ========================================================
    # 6. DELIVERY SCHEDULING
    # ========================================================

    if route_result.get("status") != "success":

        scheduler_result = {
            "status": "failed",
            "message": (
                "Delivery scheduling cannot proceed "
                "because route optimization failed."
            ),
            "delivery_count": 0,
            "schedule": []
        }

    else:

        scheduler_result = run_delivery_scheduler(
            normalized_order,
            route_result
        )

    # ========================================================
    # 7. MAINTENANCE ANALYSIS
    # ========================================================

    assigned_vehicle_id = (
        vehicle_assignment.get(
            "vehicle_id"
        )
        if isinstance(
            vehicle_assignment,
            dict
        )
        else None
    )

    if assigned_vehicle_id:

        maintenance_result = (
            run_maintenance_analysis(
                assigned_vehicle_id
            )
        )

    else:

        maintenance_result = {

            "status": "failed",

            "message": (
                "No vehicle was assigned "
                "for maintenance analysis."
            )
        }

    # ========================================================
    # 8. OPERATIONAL ALERT
    # ========================================================

    operational_alert = (
        create_operational_alert(
            maintenance_result,
            route_result,
            payload_result,
            drone_result
        )
    )

    # ========================================================
    # 9. OVERALL STATUS
    # ========================================================

    overall_status = (
        determine_overall_status(
            delivery_result,
            route_result,
            vehicle_assignment,
            payload_result,
            drone_result,
            scheduler_result,
            maintenance_result
        )
    )

    # ========================================================
    # 10. FINAL RESULT
    # ========================================================

    final_result = {

        "status": overall_status,

        "delivery_prediction": (
            delivery_result
        ),

        "route_optimization": (
            route_result
        ),

        "vehicle_assignment": (
            vehicle_assignment
        ),

        "payload_optimization": (
            payload_result
        ),

        "drone_battery_validation": (
            drone_result
        ),

        "delivery_schedule": (
            scheduler_result
        ),

        "maintenance_analysis": (
            maintenance_result
        ),

        "operational_alert": (
            operational_alert
        )
    }

    return final_result


# ============================================================
# DISPLAY FINAL RESULT
# ============================================================

def display_final_result(
    final_result
):

    print("\n" + "=" * 70)
    print("SMARTLOGIX AI - FINAL RESULT")
    print("=" * 70)

    # ========================================================
    # OVERALL STATUS
    # ========================================================

    print(
        f"\nOverall Status: "
        f"{str(final_result.get('status')).upper()}"
    )

    # ========================================================
    # DELIVERY PREDICTION
    # ========================================================

    print(
        "\n--- DELIVERY PREDICTION ---"
    )

    delivery = (
        final_result.get(
            "delivery_prediction",
            {}
        )
    )

    if delivery.get(
        "status"
    ) == "success":

        print(
            f"Transport Mode: "
            f"{delivery.get('transport_mode')}"
        )

        print(
            f"Predicted ETA: "
            f"{delivery.get('predicted_eta_hours'):.2f} hours"
        )

        print(
            f"ETA Duration: "
            f"{delivery.get('eta_duration')}"
        )

    else:

        print(
            f"Status: "
            f"{delivery.get('status')}"
        )

        print(
            f"Message: "
            f"{delivery.get('message')}"
        )

    # ========================================================
    # ROUTE
    # ========================================================

    print(
        "\n--- ROUTE OPTIMIZATION ---"
    )

    route = (
        final_result.get(
            "route_optimization",
            {}
        )
    )

    if route.get(
        "status"
    ) == "success":

        print(
            f"Route ID: "
            f"{route.get('route_id')}"
        )

        print(
            f"Vehicle ID: "
            f"{route.get('vehicle_id')}"
        )

        print(
            f"Route Match Status: "
            f"{route.get('route_match_status')}"
        )

        print(
            f"Route Distance: "
            f"{route.get('route_distance_km')} km"
        )

        print(
            f"Traffic Adjusted Time: "
            f"{route.get('traffic_adjusted_time_hours')} hours"
        )

        print(
            f"Optimization Score: "
            f"{route.get('optimization_score')}"
        )

    else:

        print(
            f"Status: "
            f"{route.get('status')}"
        )

        print(
            f"Message: "
            f"{route.get('message')}"
        )

    # ========================================================
    # VEHICLE ASSIGNMENT
    # ========================================================

    print(
        "\n--- VEHICLE ASSIGNMENT ---"
    )

    vehicle = (
        final_result.get(
            "vehicle_assignment",
            {}
        )
    )

    print(
        f"Status: "
        f"{vehicle.get('status')}"
    )

    print(
        f"Vehicle ID: "
        f"{vehicle.get('vehicle_id')}"
    )

    print(
        f"Assignment Source: "
        f"{vehicle.get('assignment_source')}"
    )

    print(
        f"Transport Mode: "
        f"{vehicle.get('transport_mode')}"
    )

    print(
        f"Capacity: "
        f"{vehicle.get('capacity_kg')} kg"
    )

    print(
        f"Capacity Utilization: "
        f"{vehicle.get('capacity_utilization_pct')}%"
    )

    print(
        f"Assignment Score: "
        f"{vehicle.get('assignment_score')}"
    )

    # ========================================================
    # PAYLOAD
    # ========================================================

    print(
        "\n--- PAYLOAD OPTIMIZATION ---"
    )

    payload = (
        final_result.get(
            "payload_optimization",
            {}
        )
    )

    print(
        f"Status: "
        f"{payload.get('status')}"
    )

    print(
        f"Package Weight: "
        f"{payload.get('package_weight_kg')} kg"
    )

    print(
        f"Vehicle Capacity: "
        f"{payload.get('capacity_kg')} kg"
    )

    print(
        f"Capacity Utilization: "
        f"{payload.get('capacity_utilization_pct')}%"
    )

    # ========================================================
    # DRONE BATTERY
    # ========================================================

    print(
        "\n--- DRONE BATTERY VALIDATION ---"
    )

    drone = (
        final_result.get(
            "drone_battery_validation",
            {}
        )
    )

    print(
        f"Status: "
        f"{drone.get('status')}"
    )

    if drone.get(
        "status"
    ) == "NOT_APPLICABLE":

        print(
            f"Message: "
            f"{drone.get('message')}"
        )

    else:

        if drone.get(
            "one_way_distance_km"
        ) is not None:

            print(
                f"One-Way Distance: "
                f"{drone.get('one_way_distance_km')} km"
            )

        if drone.get(
            "round_trip_distance_km"
        ) is not None:

            print(
                f"Round-Trip Distance: "
                f"{drone.get('round_trip_distance_km')} km"
            )

        if drone.get(
            "available_battery_kwh"
        ) is not None:

            print(
                f"Available Battery: "
                f"{drone.get('available_battery_kwh')} kWh"
            )

        if drone.get(
            "required_energy_kwh"
        ) is not None:

            print(
                f"Required Energy: "
                f"{drone.get('required_energy_kwh')} kWh"
            )

        if drone.get(
            "energy_margin_kwh"
        ) is not None:

            print(
                f"Energy Margin: "
                f"{drone.get('energy_margin_kwh')} kWh"
            )

    # ========================================================
    # SCHEDULER
    # ========================================================

    print(
        "\n--- DELIVERY SCHEDULER ---"
    )

    scheduler = (
        final_result.get(
            "delivery_schedule",
            {}
        )
    )

    print(
        f"Status: "
        f"{scheduler.get('status')}"
    )

    print(
        f"Delivery Count: "
        f"{scheduler.get('delivery_count')}"
    )

    for item in scheduler.get(
        "schedule",
        []
    ):

        print(
            f"\nOrder ID: "
            f"{item.get('order_id')}"
        )

        print(
            f"Priority: "
            f"{item.get('priority')}"
        )

        print(
            f"Departure: "
            f"{item.get('planned_departure')}"
        )

        print(
            f"Arrival: "
            f"{item.get('planned_arrival')}"
        )

        print(
            f"Service End: "
            f"{item.get('planned_service_end')}"
        )

        print(
            f"Travel Time: "
            f"{item.get('travel_time_hours')} hours"
        )

    # ========================================================
    # MAINTENANCE
    # ========================================================

    print(
        "\n--- MAINTENANCE ANALYSIS ---"
    )

    maintenance = (
        final_result.get(
            "maintenance_analysis",
            {}
        )
    )

    print(
        f"Vehicle ID: "
        f"{maintenance.get('vehicle_id')}"
    )

    maintenance_status = (
        maintenance.get(
            "status"
        )
    )

    if maintenance_status == "success":

        risk = (
            maintenance.get(
                "risk",
                {}
            )
        )

        print(
            f"History Level: "
            f"{maintenance.get('history_level')}"
        )

        print(
            f"Evidence Level: "
            f"{maintenance.get('evidence_level')}"
        )

        print(
            f"Maintenance Count: "
            f"{maintenance.get('maintenance_count')}"
        )

        print(
            f"Risk Score: "
            f"{risk.get('score')}"
        )

        print(
            f"Risk Level: "
            f"{risk.get('level')}"
        )

        print(
            f"Priority: "
            f"{risk.get('priority')}"
        )

    elif maintenance_status == "not_found":

        print(
            "Status: NOT FOUND"
        )

    else:

        print(
            f"Status: "
            f"{str(maintenance_status).upper()}"
        )

        print(
            f"Message: "
            f"{maintenance.get('message')}"
        )

    # ========================================================
    # OPERATIONAL ALERT
    # ========================================================

    print(
        "\n--- OPERATIONAL ALERT ---"
    )

    print(
        final_result.get(
            "operational_alert"
        )
    )

    # ========================================================
    # FINAL JSON
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL JSON RESULT:"
    )

    print(
        json.dumps(
            final_result,
            indent=4,
            default=str
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    sample_order = (
        create_sample_order()
    )

    final_result = (
        smartlogix_agent(
            sample_order
        )
    )

    display_final_result(
        final_result
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
