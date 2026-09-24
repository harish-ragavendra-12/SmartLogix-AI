import pandas as pd

from src.config.config import PROCESSED_DATA_DIR


def main():

    file_path = PROCESSED_DATA_DIR / "maintenance_recommendations.csv"

    df = pd.read_csv(file_path)

    print("=" * 70)
    print("MAINTENANCE RECOMMENDATION COVERAGE")
    print("=" * 70)

    print(f"\nDataset shape: {df.shape}")
    print(f"Unique vehicles: {df['vehicle_id'].nunique()}")

    target_vehicle = "VEH-0650"

    print("\n" + "=" * 70)
    print(f"CHECKING {target_vehicle}")
    print("=" * 70)

    vehicle_data = df[df["vehicle_id"].astype(str) == target_vehicle]

    if vehicle_data.empty:

        print(f"\n{target_vehicle} NOT FOUND in maintenance_recommendations.csv")

    else:

        print(f"\n{target_vehicle} FOUND in maintenance_recommendations.csv")

        print("\nRecommendation record:")
        print(vehicle_data.to_string(index=False))


if __name__ == "__main__":
    main()