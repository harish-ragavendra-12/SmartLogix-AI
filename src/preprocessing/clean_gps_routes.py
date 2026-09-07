import json
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Load raw GPS routes JSON
# --------------------------------------------------

project_root = Path(__file__).resolve().parents[2]

input_path = project_root / "data" / "raw" / "gps_routes.json"

with open(input_path, "r", encoding="utf-8") as file:
    data = json.load(file)

print("Original route records:", len(data))


# --------------------------------------------------
# 2. Create route-level records
# --------------------------------------------------

route_records = []

for route in data:

    fuel_value = route["fuel_or_energy_used"]

    if isinstance(fuel_value, str):

        if "L" in fuel_value:
            fuel_energy_unit = "L"
            fuel_energy_value = fuel_value.replace("L", "").strip()

        elif "Wh" in fuel_value:
            fuel_energy_unit = "Wh"
            fuel_energy_value = fuel_value.replace("Wh", "").strip()

        else:
            fuel_energy_unit = "unknown"
            fuel_energy_value = fuel_value

    else:
        fuel_energy_unit = "unknown"
        fuel_energy_value = fuel_value

    route_records.append({
        "route_id": route["route_id"],
        "vehicle_id": route["vehicle_id"],
        "vehicle_type": route["vehicle_type"],
        "route_date": route["route_date"],
        "order_ids": ",".join(route["order_ids"]),
        "planned_distance_km": route["planned_distance_km"],
        "actual_distance_km": route["actual_distance_km"],
        "planned_duration_min": route["planned_duration_min"],
        "actual_duration_min": route["actual_duration_min"],
        "stops_planned": route["stops_planned"],
        "stops_completed": route["stops_completed"],
        "fuel_energy_value": fuel_energy_value,
        "fuel_energy_unit": fuel_energy_unit,
        "driver_id": route["driver_id"]
    })


routes_df = pd.DataFrame(route_records)


# --------------------------------------------------
# 3. Remove duplicate routes
# --------------------------------------------------

duplicates = routes_df.duplicated(subset="route_id").sum()

routes_df = routes_df.drop_duplicates(subset="route_id")

print("Duplicate routes removed:", duplicates)


# --------------------------------------------------
# 4. Clean route_date
# --------------------------------------------------

routes_df["route_date"] = pd.to_datetime(
    routes_df["route_date"],
    errors="coerce"
)


# --------------------------------------------------
# 5. Handle missing actual distance
# --------------------------------------------------

missing_distance = routes_df["actual_distance_km"].isna().sum()

routes_df["actual_distance_km"] = routes_df[
    "actual_distance_km"
].fillna(routes_df["planned_distance_km"])

print("Missing actual distances filled:", missing_distance)


# --------------------------------------------------
# 6. Clean numeric columns
# --------------------------------------------------

numeric_columns = [
    "planned_distance_km",
    "actual_distance_km",
    "planned_duration_min",
    "actual_duration_min",
    "stops_planned",
    "stops_completed",
    "fuel_energy_value"
]

for column in numeric_columns:
    routes_df[column] = pd.to_numeric(
        routes_df[column],
        errors="coerce"
    )


# --------------------------------------------------
# 7. Handle missing driver IDs
# --------------------------------------------------

routes_df["driver_id"] = routes_df["driver_id"].fillna("unknown")


# --------------------------------------------------
# 8. Create waypoint records
# --------------------------------------------------

waypoint_records = []

for route in data:

    for waypoint in route["waypoints"]:

        waypoint_records.append({
            "route_id": route["route_id"],
            "vehicle_id": route["vehicle_id"],
            "seq": waypoint.get("seq"),
            "lat": waypoint.get("lat"),
            "lon": waypoint.get("lon"),
            "ts": waypoint.get("ts"),
            "speed_kmph": waypoint.get("speed_kmph")
        })


waypoints_df = pd.DataFrame(waypoint_records)


# --------------------------------------------------
# 9. Clean waypoint timestamps
# --------------------------------------------------

def parse_timestamp(value):
    if pd.isna(value):
        return pd.NaT

    value = str(value).strip()

    # Handle Unix timestamp
    if value.isdigit() and len(value) == 10:
        return pd.to_datetime(
            int(value),
            unit="s",
            utc=True
        )

    # Handle normal date/time formats
    return pd.to_datetime(
        value,
        errors="coerce",
        format="mixed",
        utc=True
    )


waypoints_df["ts"] = waypoints_df["ts"].apply(parse_timestamp)


# --------------------------------------------------
# 10. Clean waypoint numeric columns
# --------------------------------------------------

waypoint_numeric_columns = [
    "seq",
    "lat",
    "lon",
    "speed_kmph"
]

for column in waypoint_numeric_columns:
    waypoints_df[column] = pd.to_numeric(
        waypoints_df[column],
        errors="coerce"
    )


# --------------------------------------------------
# 11. Create processed directory
# --------------------------------------------------

output_directory = project_root / "data" / "processed"

output_directory.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# 12. Save cleaned route data
# --------------------------------------------------

routes_output = output_directory / "gps_routes_clean.csv"

routes_df.to_csv(
    routes_output,
    index=False
)


# --------------------------------------------------
# 13. Save cleaned waypoint data
# --------------------------------------------------

waypoints_output = output_directory / "gps_waypoints_clean.csv"

waypoints_df.to_csv(
    waypoints_output,
    index=False
)


# --------------------------------------------------
# 14. Final validation
# --------------------------------------------------

print("\nCleaning completed successfully!")

print("\nRoutes dataset:")
print("Shape:", routes_df.shape)

print("\nWaypoints dataset:")
print("Shape:", waypoints_df.shape)

print("\nRoute missing values:")
print(routes_df.isnull().sum())

print("\nWaypoint missing values:")
print(waypoints_df.isnull().sum())

print("\nSaved files:")
print(routes_output)
print(waypoints_output)