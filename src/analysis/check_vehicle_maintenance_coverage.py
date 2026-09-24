import pandas as pd
from pathlib import Path


# ============================================================
# VEHICLE ↔ MAINTENANCE COVERAGE CHECK
# ============================================================

# Project root:
# D:\Data Science\SmartLogix_AI

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FLEET_PATH = PROJECT_ROOT / "data" / "processed" / "fleet_vehicles_cleaned.csv"
MAINTENANCE_PATH = PROJECT_ROOT / "data" / "processed" / "maintenance_history_clean.csv"


def main():

    print("=" * 70)
    print("VEHICLE - MAINTENANCE COVERAGE ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load datasets
    # --------------------------------------------------------

    fleet_df = pd.read_csv(FLEET_PATH)
    maintenance_df = pd.read_csv(MAINTENANCE_PATH)

    print("\nDatasets loaded successfully.")

    # --------------------------------------------------------
    # 2. Basic dataset information
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    print("\nFleet dataset shape:")
    print(fleet_df.shape)

    print("\nMaintenance dataset shape:")
    print(maintenance_df.shape)

    # --------------------------------------------------------
    # 3. Check vehicle_id column
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VEHICLE ID CHECK")
    print("=" * 70)

    print("\nFleet columns:")
    print(fleet_df.columns.tolist())

    print("\nMaintenance columns:")
    print(maintenance_df.columns.tolist())

    # --------------------------------------------------------
    # 4. Unique vehicle counts
    # --------------------------------------------------------

    fleet_vehicle_ids = set(
        fleet_df["vehicle_id"].dropna().astype(str).unique()
    )

    maintenance_vehicle_ids = set(
        maintenance_df["vehicle_id"].dropna().astype(str).unique()
    )

    print("\n" + "=" * 70)
    print("UNIQUE VEHICLE COUNTS")
    print("=" * 70)

    print(f"\nUnique fleet vehicles       : {len(fleet_vehicle_ids)}")
    print(f"Unique maintenance vehicles: {len(maintenance_vehicle_ids)}")

    # --------------------------------------------------------
    # 5. Find overlapping vehicles
    # --------------------------------------------------------

    common_vehicle_ids = fleet_vehicle_ids.intersection(
        maintenance_vehicle_ids
    )

    print("\n" + "=" * 70)
    print("VEHICLE ID OVERLAP")
    print("=" * 70)

    print(f"\nVehicles present in BOTH datasets: {len(common_vehicle_ids)}")

    # --------------------------------------------------------
    # 6. Fleet vehicles without maintenance records
    # --------------------------------------------------------

    fleet_without_maintenance = fleet_vehicle_ids.difference(
        maintenance_vehicle_ids
    )

    print("\n" + "=" * 70)
    print("FLEET VEHICLES WITHOUT MAINTENANCE RECORDS")
    print("=" * 70)

    print(
        f"\nFleet vehicles without maintenance records: "
        f"{len(fleet_without_maintenance)}"
    )

    if fleet_without_maintenance:
        print("\nSample vehicle IDs:")

        for vehicle_id in sorted(fleet_without_maintenance)[:20]:
            print(vehicle_id)

    # --------------------------------------------------------
    # 7. Maintenance vehicles not present in fleet
    # --------------------------------------------------------

    maintenance_without_fleet = maintenance_vehicle_ids.difference(
        fleet_vehicle_ids
    )

    print("\n" + "=" * 70)
    print("MAINTENANCE VEHICLES NOT PRESENT IN FLEET")
    print("=" * 70)

    print(
        f"\nMaintenance vehicles not present in fleet: "
        f"{len(maintenance_without_fleet)}"
    )

    if maintenance_without_fleet:
        print("\nSample vehicle IDs:")

        for vehicle_id in sorted(maintenance_without_fleet)[:20]:
            print(vehicle_id)

    # --------------------------------------------------------
    # 8. Specifically check VEH-0650
    # --------------------------------------------------------

    target_vehicle = "VEH-0650"

    print("\n" + "=" * 70)
    print(f"SPECIFIC VEHICLE CHECK: {target_vehicle}")
    print("=" * 70)

    print(
        f"\n{target_vehicle} in fleet dataset       : "
        f"{target_vehicle in fleet_vehicle_ids}"
    )

    print(
        f"{target_vehicle} in maintenance dataset: "
        f"{target_vehicle in maintenance_vehicle_ids}"
    )

    # --------------------------------------------------------
    # 9. Maintenance records for VEH-0650
    # --------------------------------------------------------

    vehicle_maintenance = maintenance_df[
        maintenance_df["vehicle_id"].astype(str) == target_vehicle
    ]

    print(f"\nMaintenance records for {target_vehicle}:")
    print(len(vehicle_maintenance))

    if not vehicle_maintenance.empty:
        print("\nMaintenance records:")
        print(vehicle_maintenance.to_string(index=False))

    # --------------------------------------------------------
    # 10. Coverage percentage
    # --------------------------------------------------------

    if len(fleet_vehicle_ids) > 0:

        maintenance_coverage = (
            len(common_vehicle_ids)
            / len(fleet_vehicle_ids)
        ) * 100

        print("\n" + "=" * 70)
        print("MAINTENANCE COVERAGE")
        print("=" * 70)

        print(
            f"\nMaintenance coverage of fleet: "
            f"{maintenance_coverage:.2f}%"
        )

    # --------------------------------------------------------
    # 11. Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\nTotal fleet vehicles             : {len(fleet_vehicle_ids)}")
    print(
        f"Vehicles with maintenance data   : "
        f"{len(common_vehicle_ids)}"
    )
    print(
        f"Vehicles without maintenance data: "
        f"{len(fleet_without_maintenance)}"
    )
    print(
        f"Maintenance vehicles not in fleet: "
        f"{len(maintenance_without_fleet)}"
    )

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()