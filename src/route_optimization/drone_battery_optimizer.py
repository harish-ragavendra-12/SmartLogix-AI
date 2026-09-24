"""
SmartLogix AI - Drone Battery Constraint Engine

Uses drone telemetry when available and applies conservative
energy constraints.

This module is intentionally separate from route matching so the
same engine can later be connected to a real drone routing service.
"""

import math
import pandas as pd


EARTH_RADIUS_KM = 6371.0

DEFAULT_SPEED_KMPH = 45.0
DEFAULT_BATTERY_KWH = 1.5
DEFAULT_CONSUMPTION_KWH_PER_KM = 0.035
DEFAULT_RESERVE_PERCENT = 20.0


def haversine_km(
    lat1, lon1, lat2, lon2
):
    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return (
        EARTH_RADIUS_KM
        * 2
        * math.asin(math.sqrt(a))
    )


def estimate_drone_energy(
    distance_km,
    payload_kg=0.0,
    wind_speed_kmph=0.0,
    base_consumption_kwh_per_km=DEFAULT_CONSUMPTION_KWH_PER_KM
):
    payload_factor = 1.0 + (
        max(payload_kg, 0.0) * 0.04
    )

    wind_factor = 1.0 + (
        max(wind_speed_kmph, 0.0) * 0.01
    )

    consumption = (
        base_consumption_kwh_per_km
        * payload_factor
        * wind_factor
    )

    return distance_km * consumption


def evaluate_drone_assignment(
    origin_lat,
    origin_lon,
    destination_lat,
    destination_lon,
    payload_kg=0.0,
    available_battery_kwh=DEFAULT_BATTERY_KWH,
    reserve_percent=DEFAULT_RESERVE_PERCENT,
    wind_speed_kmph=0.0,
    speed_kmph=DEFAULT_SPEED_KMPH
):
    one_way_distance = haversine_km(
        origin_lat,
        origin_lon,
        destination_lat,
        destination_lon
    )

    # Conservative assumption: drone returns to origin.
    round_trip_distance = (
        one_way_distance * 2.0
    )

    required_energy = estimate_drone_energy(
        round_trip_distance,
        payload_kg,
        wind_speed_kmph
    )

    reserve_energy = (
        available_battery_kwh
        * reserve_percent
        / 100.0
    )

    usable_energy = (
        available_battery_kwh
        - reserve_energy
    )

    feasible = (
        required_energy <= usable_energy
    )

    estimated_flight_hours = (
        round_trip_distance / speed_kmph
        if speed_kmph > 0
        else None
    )

    return {
        "status": (
            "ELIGIBLE"
            if feasible
            else "REJECTED"
        ),
        "one_way_distance_km": round(
            one_way_distance,
            3
        ),
        "round_trip_distance_km": round(
            round_trip_distance,
            3
        ),
        "payload_kg": round(
            payload_kg,
            3
        ),
        "available_battery_kwh": round(
            available_battery_kwh,
            3
        ),
        "reserve_percent": reserve_percent,
        "usable_energy_kwh": round(
            usable_energy,
            4
        ),
        "required_energy_kwh": round(
            required_energy,
            4
        ),
        "energy_margin_kwh": round(
            usable_energy - required_energy,
            4
        ),
        "estimated_flight_hours": (
            round(
                estimated_flight_hours,
                3
            )
            if estimated_flight_hours is not None
            else None
        ),
        "wind_speed_kmph": wind_speed_kmph
    }


def evaluate_from_telemetry(
    telemetry,
    origin_lat,
    origin_lon,
    destination_lat,
    destination_lon,
    payload_kg=0.0
):
    """
    Best-effort telemetry adapter.

    Recognizes common battery fields such as:
    battery_percent, battery_percentage, battery_level,
    battery_remaining_percent.
    """

    if isinstance(telemetry, pd.DataFrame):
        row = (
            telemetry.iloc[-1].to_dict()
            if not telemetry.empty
            else {}
        )
    elif isinstance(telemetry, dict):
        row = telemetry
    else:
        row = {}

    battery_percent = None

    for column in [
        "battery_percent",
        "battery_percentage",
        "battery_level",
        "battery_remaining_percent"
    ]:
        if column in row and pd.notna(row[column]):
            battery_percent = float(row[column])
            break

    battery_capacity = None

    for column in [
        "battery_capacity_kwh",
        "battery_kwh",
        "battery_capacity"
    ]:
        if column in row and pd.notna(row[column]):
            battery_capacity = float(row[column])
            break

    if battery_capacity is None:
        battery_capacity = DEFAULT_BATTERY_KWH

    if battery_percent is None:
        available = battery_capacity
    else:
        available = (
            battery_capacity
            * max(0.0, min(battery_percent, 100.0))
            / 100.0
        )

    wind_speed = float(
        row.get(
            "wind_speed_kmph",
            0.0
        )
        or 0.0
    )

    return evaluate_drone_assignment(
        origin_lat,
        origin_lon,
        destination_lat,
        destination_lon,
        payload_kg=payload_kg,
        available_battery_kwh=available,
        wind_speed_kmph=wind_speed
    )


def main():
    result = evaluate_drone_assignment(
        12.9716,
        77.5946,
        12.9950,
        77.6200,
        payload_kg=2.0,
        available_battery_kwh=1.5
    )

    print(result)


if __name__ == "__main__":
    main()
