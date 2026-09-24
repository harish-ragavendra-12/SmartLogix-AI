"""
SmartLogix AI - Delivery Scheduling Engine

Creates delivery order slots using:
- delivery priority
- estimated travel time
- optional requested delivery time
- optional time windows

This is a deterministic scheduling baseline and can later be
replaced with an optimization solver.
"""

from datetime import datetime, timedelta


PRIORITY_RANK = {
    "critical": 0,
    "urgent": 1,
    "high": 2,
    "standard": 3,
    "low": 4
}


def _priority(order):
    return PRIORITY_RANK.get(
        str(
            order.get(
                "delivery_priority",
                order.get("priority", "standard")
            )
        ).lower(),
        3
    )


def schedule_deliveries(
    orders,
    start_time=None,
    average_service_minutes=15
):
    if start_time is None:
        start_time = datetime.now()

    current_time = start_time
    scheduled = []

    normalized = []

    for index, order in enumerate(
        orders,
        start=1
    ):
        item = dict(order)

        item["_input_index"] = index

        item["_priority_rank"] = _priority(
            item
        )

        normalized.append(item)

    normalized.sort(
        key=lambda x: (
            x["_priority_rank"],
            x.get("estimated_travel_time_hours", 0)
        )
    )

    for sequence, order in enumerate(
        normalized,
        start=1
    ):
        travel_hours = float(
            order.get(
                "estimated_travel_time_hours",
                order.get(
                    "traffic_adjusted_time_hours",
                    0.0
                )
            )
            or 0.0
        )

        arrival = (
            current_time
            + timedelta(
                hours=travel_hours
            )
        )

        service_end = (
            arrival
            + timedelta(
                minutes=average_service_minutes
            )
        )

        scheduled.append({
            "sequence": sequence,
            "order_id": order.get(
                "order_id",
                f"ORDER-{sequence:04d}"
            ),
            "priority": order.get(
                "delivery_priority",
                order.get(
                    "priority",
                    "standard"
                )
            ),
            "planned_departure": (
                current_time.isoformat()
            ),
            "planned_arrival": (
                arrival.isoformat()
            ),
            "planned_service_end": (
                service_end.isoformat()
            ),
            "travel_time_hours": round(
                travel_hours,
                3
            )
        })

        current_time = service_end

    return {
        "status": "success",
        "start_time": start_time.isoformat(),
        "delivery_count": len(scheduled),
        "schedule": scheduled
    }


def main():
    orders = [
        {
            "order_id": "ORD-001",
            "delivery_priority": "standard",
            "estimated_travel_time_hours": 1.2
        },
        {
            "order_id": "ORD-002",
            "delivery_priority": "urgent",
            "estimated_travel_time_hours": 0.8
        }
    ]

    result = schedule_deliveries(
        orders,
        start_time=datetime(2026, 9, 22, 9, 0)
    )

    for row in result["schedule"]:
        print(row)


if __name__ == "__main__":
    main()
