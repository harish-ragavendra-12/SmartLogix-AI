"""
SmartLogix AI - Capacitated Vehicle Routing Problem (CVRP)

Dependency-free greedy baseline for the project.

Supports:
- multiple delivery orders
- multiple vehicles
- vehicle capacity
- order weight
- priority-aware route construction
- distance/time calculation
- route summaries

This is a practical baseline. It is not a mathematically exact VRP solver.
A later OR-Tools implementation can replace this module without changing
the surrounding interface.
"""

from math import radians, sin, cos, asin, sqrt
from datetime import datetime
import pandas as pd


EARTH_RADIUS_KM = 6371.0


def haversine_km(
    lat1, lon1, lat2, lon2
):
    lat1 = radians(float(lat1))
    lon1 = radians(float(lon1))
    lat2 = radians(float(lat2))
    lon2 = radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    return (
        EARTH_RADIUS_KM
        * 2
        * asin(sqrt(a))
    )


PRIORITY_WEIGHT = {
    "critical": 4,
    "urgent": 3,
    "high": 2,
    "standard": 1,
    "low": 0
}


def _get(row, names, default=None):
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return default


def normalize_orders(orders):
    if isinstance(orders, pd.DataFrame):
        records = orders.to_dict("records")
    elif isinstance(orders, dict):
        records = [orders]
    else:
        records = list(orders)

    normalized = []

    for index, order in enumerate(records, start=1):
        normalized.append({
            "order_id": _get(
                order,
                ["order_id", "id"],
                f"ORDER-{index:04d}"
            ),
            "latitude": float(
                _get(
                    order,
                    [
                        "destination_lat",
                        "delivery_lat",
                        "latitude",
                        "lat"
                    ]
                )
            ),
            "longitude": float(
                _get(
                    order,
                    [
                        "destination_lon",
                        "delivery_lon",
                        "longitude",
                        "lon",
                        "lng"
                    ]
                )
            ),
            "weight_kg": float(
                _get(
                    order,
                    [
                        "package_weight_kg",
                        "weight_kg",
                        "package_weight",
                        "weight"
                    ],
                    0.0
                )
            ),
            "priority": str(
                _get(
                    order,
                    ["delivery_priority", "priority"],
                    "standard"
                )
            ).lower()
        })

    return normalized


def normalize_vehicles(vehicles):
    if isinstance(vehicles, pd.DataFrame):
        records = vehicles.to_dict("records")
    elif isinstance(vehicles, dict):
        records = [vehicles]
    else:
        records = list(vehicles)

    normalized = []

    for index, vehicle in enumerate(records, start=1):
        capacity = _get(
            vehicle,
            [
                "payload_capacity_kg",
                "max_payload_kg",
                "capacity_kg",
                "payload_capacity",
                "capacity"
            ],
            100.0
        )

        normalized.append({
            "vehicle_id": _get(
                vehicle,
                ["vehicle_id", "id"],
                f"VEH-{index:04d}"
            ),
            "capacity_kg": float(capacity)
        })

    return normalized


def _route_distance(
    route_orders,
    depot_lat,
    depot_lon
):
    if not route_orders:
        return 0.0

    total = 0.0
    current_lat = depot_lat
    current_lon = depot_lon

    for order in route_orders:
        total += haversine_km(
            current_lat,
            current_lon,
            order["latitude"],
            order["longitude"]
        )

        current_lat = order["latitude"]
        current_lon = order["longitude"]

    total += haversine_km(
        current_lat,
        current_lon,
        depot_lat,
        depot_lon
    )

    return total


def build_greedy_cvrp(
    orders,
    vehicles,
    depot_lat,
    depot_lon,
    average_speed_kmph=35.0
):
    orders = normalize_orders(orders)
    vehicles = normalize_vehicles(vehicles)

    orders = sorted(
        orders,
        key=lambda x: PRIORITY_WEIGHT.get(
            x["priority"],
            1
        ),
        reverse=True
    )

    remaining = orders.copy()
    routes = []

    for vehicle in vehicles:
        assigned = []
        used_capacity = 0.0

        while remaining:
            feasible = [
                order
                for order in remaining
                if (
                    used_capacity
                    + order["weight_kg"]
                    <= vehicle["capacity_kg"]
                )
            ]

            if not feasible:
                break

            current_lat = (
                depot_lat
                if not assigned
                else assigned[-1]["latitude"]
            )
            current_lon = (
                depot_lon
                if not assigned
                else assigned[-1]["longitude"]
            )

            # Priority first, then nearest feasible order.
            selected = min(
                feasible,
                key=lambda order: (
                    -PRIORITY_WEIGHT.get(
                        order["priority"],
                        1
                    ),
                    haversine_km(
                        current_lat,
                        current_lon,
                        order["latitude"],
                        order["longitude"]
                    )
                )
            )

            assigned.append(selected)
            used_capacity += selected["weight_kg"]
            remaining.remove(selected)

        distance = _route_distance(
            assigned,
            depot_lat,
            depot_lon
        )

        estimated_hours = (
            distance / average_speed_kmph
            if average_speed_kmph > 0
            else None
        )

        routes.append({
            "vehicle_id": vehicle["vehicle_id"],
            "capacity_kg": vehicle["capacity_kg"],
            "used_payload_kg": round(
                used_capacity,
                3
            ),
            "remaining_capacity_kg": round(
                vehicle["capacity_kg"]
                - used_capacity,
                3
            ),
            "order_count": len(assigned),
            "order_ids": [
                order["order_id"]
                for order in assigned
            ],
            "route_distance_km": round(
                distance,
                3
            ),
            "estimated_time_hours": (
                round(estimated_hours, 3)
                if estimated_hours is not None
                else None
            )
        })

    unassigned = [
        order["order_id"]
        for order in remaining
    ]

    return {
        "status": (
            "success"
            if not unassigned
            else "partial_success"
        ),
        "vehicle_routes": routes,
        "unassigned_orders": unassigned,
        "total_orders": len(orders),
        "assigned_orders": (
            len(orders) - len(unassigned)
        )
    }


def validate_cvrp_solution(
    solution
):
    violations = []

    for route in solution.get(
        "vehicle_routes",
        []
    ):
        if (
            route["used_payload_kg"]
            > route["capacity_kg"]
        ):
            violations.append({
                "vehicle_id": route["vehicle_id"],
                "type": "PAYLOAD_CAPACITY",
                "message": (
                    "Assigned payload exceeds vehicle capacity."
                )
            })

    return {
        "valid": len(violations) == 0,
        "violations": violations
    }


def main():
    print("=" * 70)
    print("SMARTLOGIX AI - CVRP DEMO")
    print("=" * 70)

    orders = [
        {
            "order_id": "ORD-001",
            "destination_lat": 12.98,
            "destination_lon": 77.60,
            "package_weight_kg": 4.0,
            "delivery_priority": "high"
        },
        {
            "order_id": "ORD-002",
            "destination_lat": 12.96,
            "destination_lon": 77.58,
            "package_weight_kg": 3.0,
            "delivery_priority": "standard"
        },
        {
            "order_id": "ORD-003",
            "destination_lat": 13.00,
            "destination_lon": 77.62,
            "package_weight_kg": 5.0,
            "delivery_priority": "urgent"
        }
    ]

    vehicles = [
        {
            "vehicle_id": "VEH-001",
            "payload_capacity_kg": 10
        },
        {
            "vehicle_id": "VEH-002",
            "payload_capacity_kg": 10
        }
    ]

    solution = build_greedy_cvrp(
        orders,
        vehicles,
        depot_lat=12.9716,
        depot_lon=77.5946
    )

    print(solution)
    print(validate_cvrp_solution(solution))


if __name__ == "__main__":
    main()
