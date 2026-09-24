"""
SmartLogix AI - Unified Route Optimization Engine

Combines:
1. historical route matching
2. CVRP baseline
3. payload constraints
4. drone battery constraints
5. delivery scheduling

The modules remain independent so each can be tested separately.
"""

import pandas as pd

from src.route_optimization.route_optimizer import (
    optimize_single_route
)

from src.route_optimization.vrp_optimizer import (
    build_greedy_cvrp,
    validate_cvrp_solution
)

from src.route_optimization.payload_optimizer import (
    evaluate_payload_constraint
)

from src.route_optimization.drone_battery_optimizer import (
    evaluate_drone_assignment
)

from src.route_optimization.delivery_scheduler import (
    schedule_deliveries
)


def optimize_order(
    order,
    vehicle=None,
    origin_lat=None,
    origin_lon=None
):
    destination_lat = float(
        order["destination_lat"]
    )
    destination_lon = float(
        order["destination_lon"]
    )

    if origin_lat is None:
        origin_lat = float(
            order["origin_lat"]
        )

    if origin_lon is None:
        origin_lon = float(
            order["origin_lon"]
        )

    result = {
        "status": "success",
        "historical_route": None,
        "payload_constraint": None,
        "drone_constraint": None,
        "schedule": None
    }

    # --------------------------------------------------------
    # HISTORICAL ROUTE
    # --------------------------------------------------------

    try:
        ranked_routes, traffic = optimize_single_route(
            origin_lat,
            origin_lon,
            destination_lat,
            destination_lon
        )

        result["historical_route"] = {
            "status": "success",
            "best_route": ranked_routes.iloc[0].to_dict(),
            "candidate_count": len(ranked_routes),
            "traffic_context": traffic
        }

    except Exception as exc:
        result["historical_route"] = {
            "status": "failed",
            "message": str(exc)
        }

    # --------------------------------------------------------
    # PAYLOAD
    # --------------------------------------------------------

    if vehicle is not None:
        result["payload_constraint"] = (
            evaluate_payload_constraint(
                order,
                vehicle
            )
        )

    # --------------------------------------------------------
    # DRONE
    # --------------------------------------------------------

    if str(
        order.get(
            "transport_mode",
            ""
        )
    ).lower() == "drone":
        result["drone_constraint"] = (
            evaluate_drone_assignment(
                origin_lat,
                origin_lon,
                destination_lat,
                destination_lon,
                payload_kg=float(
                    order.get(
                        "package_weight_kg",
                        0.0
                    )
                )
            )
        )

    return result


def optimize_multiple_orders(
    orders,
    vehicles,
    depot_lat,
    depot_lon
):
    cvrp = build_greedy_cvrp(
        orders,
        vehicles,
        depot_lat,
        depot_lon
    )

    validation = validate_cvrp_solution(
        cvrp
    )

    schedule_input = []

    for route in cvrp["vehicle_routes"]:
        for order_id in route["order_ids"]:
            schedule_input.append({
                "order_id": order_id,
                "delivery_priority": "standard",
                "estimated_travel_time_hours": (
                    route["estimated_time_hours"]
                )
            })

    schedule = schedule_deliveries(
        schedule_input
    )

    return {
        "status": (
            "success"
            if validation["valid"]
            else "failed"
        ),
        "vrp": cvrp,
        "vrp_validation": validation,
        "schedule": schedule
    }
