import pandas as pd

from src.config.config import PROCESSED_DATA_DIR


INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "maintenance_temporal_features.csv"
)


def main():

    print("\n" + "=" * 70)
    print("MAINTENANCE TEMPORAL FEATURE COVERAGE")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["service_date"]
    )

    print("\nDataset shape:")
    print(df.shape)

    print("\nUnique vehicles:")
    print(df["vehicle_id"].nunique())

    print("\n" + "=" * 70)
    print("VEHICLE RECORD DISTRIBUTION")
    print("=" * 70)

    vehicle_counts = (
        df.groupby("vehicle_id")
        .size()
        .sort_values()
    )

    print("\nMinimum records for a vehicle:")
    print(vehicle_counts.min())

    print("\nMaximum records for a vehicle:")
    print(vehicle_counts.max())

    print("\nRecord-count distribution:")
    print(
        vehicle_counts
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\n" + "=" * 70)
    print("SINGLE-RECORD VEHICLES")
    print("=" * 70)

    single_record_vehicles = vehicle_counts[
        vehicle_counts == 1
    ]

    print(
        f"\nVehicles with exactly 1 temporal record: "
        f"{len(single_record_vehicles)}"
    )

    if not single_record_vehicles.empty:

        print("\nSample vehicle IDs:")

        print(
            single_record_vehicles
            .index
            .tolist()[:30]
        )

    print("\n" + "=" * 70)
    print("CHECKING VEH-0650")
    print("=" * 70)

    target_vehicle = "VEH-0650"

    vehicle_data = df[
        df["vehicle_id"].astype(str)
        == target_vehicle
    ]

    print(
        f"\n{target_vehicle} records: "
        f"{len(vehicle_data)}"
    )

    if vehicle_data.empty:

        print(
            f"{target_vehicle} is NOT present "
            "in maintenance_temporal_features.csv"
        )

    else:

        print(
            f"\n{target_vehicle} IS present "
            "in maintenance_temporal_features.csv"
        )

        print("\nRecords:")

        print(
            vehicle_data.to_string(
                index=False
            )
        )

    print("\n" + "=" * 70)
    print("MISSING VALUE CHECK")
    print("=" * 70)

    important_columns = [
        "previous_average_cost",
        "previous_max_cost",
        "previous_average_downtime",
        "previous_max_downtime",
        "previous_average_labour_hours",
        "previous_max_labour_hours",
        "previous_failure_rate",
        "previous_failure_count",
        "days_since_previous_maintenance"
    ]

    existing_columns = [
        column
        for column in important_columns
        if column in df.columns
    ]

    print("\nMissing values in temporal features:")

    print(
        df[existing_columns]
        .isna()
        .sum()
        .to_string()
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()