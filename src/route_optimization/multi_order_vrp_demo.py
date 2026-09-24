"""
SmartLogix AI - Multi-Order VRP Demonstration

Purpose:
- Demonstrate multi-order delivery optimization
- Assign multiple orders across multiple vehicles
- Respect vehicle payload capacity
- Consider delivery priority
- Validate the resulting CVRP solution

This uses the existing dependency-free greedy CVRP baseline.
"""

from route_optimization.vrp_optimizer import (
    build_greedy_cvrp,
    validate_cvrp_solution
)


def print_solution(solution, validation):
    print("\n" + "=" * 70)
    print("MULTI-ORDER VRP RESULT")
    print("=" * 70)

    print(f"\nStatus          : {solution['status']}")
    print(f"Total Orders    : {solution['total_orders']}")
    print(f"Assigned Orders : {solution['assigned_orders']}")
    print(
        f"Unassigned      : "
        f"{len(solution['unassigned_orders'])}"
    )

    print("\n" + "-" * 70)
    print("VEHICLE ROUTES")
    print("-" * 70)

    for route in solution["vehicle_routes"]:

        print(f"\nVehicle ID          : {route['vehicle_id']}")
        print(f"Capacity            : {route['capacity_kg']} kg")
        print(
            f"Used Payload        : "
            f"{route['used_payload_kg']} kg"
        )
        print(
            f"Remaining Capacity  : "
            f"{route['remaining_capacity_kg']} kg"
        )
        print(
            f"Order Count         : "
            f"{route['order_count']}"
        )
        print(
            f"Route Distance      : "
            f"{route['route_distance_km']} km"
        )
        print(
            f"Estimated Time      : "
            f"{route['estimated_time_hours']} hours"
        )
        print(
            f"Delivery Sequence   : "
            f"{' -> '.join(route['order_ids'])}"
        )

    print("\n" + "-" * 70)
    print("UNASSIGNED ORDERS")
    print("-" * 70)

    if solution["unassigned_orders"]:
        for order_id in solution["unassigned_orders"]:
            print(f"  - {order_id}")
    else:
        print("  None")

    print("\n" + "-" * 70)
    print("SOLUTION VALIDATION")
    print("-" * 70)

    print(
        f"Valid               : "
        f"{validation['valid']}"
    )

    if validation["violations"]:
        print("\nViolations:")

        for violation in validation["violations"]:
            print(
                f"  - {violation['vehicle_id']}: "
                f"{violation['message']}"
            )
    else:
        print("Violations          : None")


def main():

    print("=" * 70)
    print("SMARTLOGIX AI - MULTI-ORDER VRP")
    print("=" * 70)

    # ----------------------------------------------------------
    # DEPOT
    # ----------------------------------------------------------

    # Chennai depot
    depot_lat = 13.0827
    depot_lon = 80.2707

    # ----------------------------------------------------------
    # MULTIPLE ORDERS
    # ----------------------------------------------------------

    orders = [

        {
            "order_id": "ORD-VRP-001",
            "destination_lat": 12.9716,
            "destination_lon": 77.5946,
            "package_weight_kg": 3.5,
            "delivery_priority": "urgent"
        },

        {
            "order_id": "ORD-VRP-002",
            "destination_lat": 12.2958,
            "destination_lon": 76.6394,
            "package_weight_kg": 8.0,
            "delivery_priority": "high"
        },

        {
            "order_id": "ORD-VRP-003",
            "destination_lat": 11.0168,
            "destination_lon": 76.9558,
            "package_weight_kg": 5.0,
            "delivery_priority": "standard"
        },

        {
            "order_id": "ORD-VRP-004",
            "destination_lat": 13.0827,
            "destination_lon": 80.2707,
            "package_weight_kg": 2.5,
            "delivery_priority": "critical"
        },

        {
            "order_id": "ORD-VRP-005",
            "destination_lat": 12.9165,
            "destination_lon": 79.1325,
            "package_weight_kg": 4.0,
            "delivery_priority": "standard"
        },

        {
            "order_id": "ORD-VRP-006",
            "destination_lat": 11.6643,
            "destination_lon": 78.1460,
            "package_weight_kg": 6.0,
            "delivery_priority": "high"
        },

        {
            "order_id": "ORD-VRP-007",
            "destination_lat": 12.5246,
            "destination_lon": 78.2138,
            "package_weight_kg": 3.0,
            "delivery_priority": "low"
        },

        {
            "order_id": "ORD-VRP-008",
            "destination_lat": 13.3409,
            "destination_lon": 77.1010,
            "package_weight_kg": 4.5,
            "delivery_priority": "standard"
        }
    ]

    # ----------------------------------------------------------
    # VEHICLES
    # ----------------------------------------------------------

    vehicles = [

        {
            "vehicle_id": "VRP-VEH-001",
            "payload_capacity_kg": 15.0
        },

        {
            "vehicle_id": "VRP-VEH-002",
            "payload_capacity_kg": 15.0
        },

        {
            "vehicle_id": "VRP-VEH-003",
            "payload_capacity_kg": 15.0
        }
    ]

    print("\nInput Summary")
    print("-" * 70)

    print(f"Number of Orders : {len(orders)}")
    print(f"Number of Vehicles: {len(vehicles)}")

    total_weight = sum(
        order["package_weight_kg"]
        for order in orders
    )

    total_capacity = sum(
        vehicle["payload_capacity_kg"]
        for vehicle in vehicles
    )

    print(f"Total Order Weight: {total_weight:.2f} kg")
    print(f"Total Capacity    : {total_capacity:.2f} kg")

    # ----------------------------------------------------------
    # RUN CVRP
    # ----------------------------------------------------------

    print("\n" + "=" * 70)
    print("RUNNING GREEDY CVRP")
    print("=" * 70)

    solution = build_greedy_cvrp(
        orders=orders,
        vehicles=vehicles,
        depot_lat=depot_lat,
        depot_lon=depot_lon,
        average_speed_kmph=40.0
    )

    # ----------------------------------------------------------
    # VALIDATE
    # ----------------------------------------------------------

    validation = validate_cvrp_solution(
        solution
    )

    # ----------------------------------------------------------
    # DISPLAY
    # ----------------------------------------------------------

    print_solution(
        solution,
        validation
    )

    # ----------------------------------------------------------
    # FINAL STATUS
    # ----------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL VRP STATUS")
    print("=" * 70)

    if (
        solution["status"] == "success"
        and validation["valid"]
    ):
        print(
            "MULTI-ORDER VRP : SUCCESS"
        )
    elif validation["valid"]:
        print(
            "MULTI-ORDER VRP : PARTIAL SUCCESS"
        )
    else:
        print(
            "MULTI-ORDER VRP : FAILED"
        )


if __name__ == "__main__":
    main()