"""
SmartLogix AI - Payload Constraint Engine

Evaluates whether an order/batch can be assigned to a vehicle.

Handles:
- payload capacity
- package weight
- optional volume capacity
- fragile/hazmat/cold-chain compatibility
- utilization level
"""

import pandas as pd


def _get(data, names, default=None):
    for name in names:
        if name in data and pd.notna(data[name]):
            return data[name]
    return default


def evaluate_payload_constraint(
    order,
    vehicle
):
    package_weight = float(
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
    )

    capacity = float(
        _get(
            vehicle,
            [
                "payload_capacity_kg",
                "max_payload_kg",
                "capacity_kg",
                "payload_capacity",
                "capacity"
            ],
            0.0
        )
    )

    if capacity <= 0:
        return {
            "status": "REJECTED",
            "reason": "Vehicle payload capacity is missing or invalid."
        }

    weight_utilization = (
        package_weight / capacity
    )

    violations = []

    if package_weight > capacity:
        violations.append(
            "Package weight exceeds vehicle payload capacity."
        )

    is_fragile = int(
        _get(order, ["is_fragile", "fragile"], 0)
    )

    is_hazmat = int(
        _get(order, ["is_hazmat", "hazmat"], 0)
    )

    cold_chain = int(
        _get(
            order,
            ["cold_chain_required", "cold_chain"],
            0
        )
    )

    vehicle_type = str(
        _get(
            vehicle,
            ["vehicle_type", "type", "transport_mode"],
            ""
        )
    ).lower()

    # These are conservative business rules. They can be configured later
    # from fleet capability metadata.
    if is_hazmat and vehicle_type in {
        "bike",
        "drone"
    }:
        violations.append(
            "Hazmat order cannot use this vehicle type."
        )

    if cold_chain and vehicle_type == "bike":
        violations.append(
            "Cold-chain order requires a compatible vehicle."
        )

    if violations:
        return {
            "status": "REJECTED",
            "package_weight_kg": package_weight,
            "capacity_kg": capacity,
            "utilization": round(
                weight_utilization,
                3
            ),
            "violations": violations
        }

    if weight_utilization >= 0.90:
        utilization_level = "HIGH"
    elif weight_utilization >= 0.70:
        utilization_level = "MEDIUM"
    else:
        utilization_level = "LOW"

    return {
        "status": "ELIGIBLE",
        "package_weight_kg": package_weight,
        "capacity_kg": capacity,
        "remaining_capacity_kg": round(
            capacity - package_weight,
            3
        ),
        "utilization": round(
            weight_utilization,
            3
        ),
        "utilization_level": utilization_level,
        "violations": []
    }


def evaluate_batch_payload(
    orders,
    vehicle
):
    total_weight = 0.0

    for order in orders:
        total_weight += float(
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
        )

    capacity = float(
        _get(
            vehicle,
            [
                "payload_capacity_kg",
                "max_payload_kg",
                "capacity_kg",
                "payload_capacity",
                "capacity"
            ],
            0.0
        )
    )

    utilization = (
        total_weight / capacity
        if capacity > 0
        else 999
    )

    return {
        "status": (
            "ELIGIBLE"
            if capacity > 0
            and total_weight <= capacity
            else "REJECTED"
        ),
        "total_payload_kg": round(
            total_weight,
            3
        ),
        "capacity_kg": round(
            capacity,
            3
        ),
        "remaining_capacity_kg": round(
            max(capacity - total_weight, 0),
            3
        ),
        "utilization": round(
            utilization,
            3
        )
    }


def main():
    order = {
        "package_weight_kg": 12,
        "is_fragile": 0,
        "is_hazmat": 0,
        "cold_chain_required": 0
    }

    vehicle = {
        "vehicle_id": "VEH-001",
        "vehicle_type": "Truck",
        "payload_capacity_kg": 20
    }

    print(
        evaluate_payload_constraint(
            order,
            vehicle
        )
    )


if __name__ == "__main__":
    main()
