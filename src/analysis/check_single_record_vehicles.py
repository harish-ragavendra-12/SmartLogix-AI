import pandas as pd

from src.config.config import PROCESSED_DATA_DIR


INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "maintenance_history_clean.csv"
)


def main():

    print("\n" + "=" * 70)
    print("MAINTENANCE HISTORY RECORD DISTRIBUTION")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    print(f"\nTotal records   : {len(df)}")
    print(f"Unique vehicles : {df['vehicle_id'].nunique()}")

    # --------------------------------------------------------
    # Count maintenance records per vehicle
    # --------------------------------------------------------

    vehicle_counts = (
        df.groupby("vehicle_id")
        .size()
        .sort_values()
    )

    print("\n" + "=" * 70)
    print("RECORD COUNT PER VEHICLE")
    print("=" * 70)

    print(
        vehicle_counts
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Single-record vehicles
    # --------------------------------------------------------

    single_record_vehicles = vehicle_counts[
        vehicle_counts == 1
    ]

    print("\n" + "=" * 70)
    print("SINGLE-RECORD VEHICLES")
    print("=" * 70)

    print(
        f"\nVehicles with exactly one "
        f"maintenance record: "
        f"{len(single_record_vehicles)}"
    )

    if not single_record_vehicles.empty:

        print("\nSample vehicle IDs:")

        print(
            single_record_vehicles
            .index
            .tolist()[:30]
        )

    # --------------------------------------------------------
    # Specifically check VEH-0650
    # --------------------------------------------------------

    target_vehicle = "VEH-0650"

    target_count = (
        vehicle_counts
        .get(target_vehicle, 0)
    )

    print("\n" + "=" * 70)
    print(f"CHECKING {target_vehicle}")
    print("=" * 70)

    print(
        f"\nMaintenance records for "
        f"{target_vehicle}: {target_count}"
    )

    if target_count == 1:

        print(
            "\nThis vehicle is a single-record "
            "maintenance vehicle."
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"\nTotal vehicles with maintenance data : "
        f"{len(vehicle_counts)}"
    )

    print(
        f"Vehicles with exactly 1 record       : "
        f"{len(single_record_vehicles)}"
    )

    print(
        f"Vehicles with 2+ records              : "
        f"{(vehicle_counts >= 2).sum()}"
    )

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()