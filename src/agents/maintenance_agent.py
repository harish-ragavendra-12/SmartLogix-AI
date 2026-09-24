import pandas as pd

from src.config.config import PROCESSED_DATA_DIR


# ============================================================
# Configuration
# ============================================================

OPERATIONAL_PROFILE_FILE = (
    PROCESSED_DATA_DIR
    / "operational_maintenance_profile.csv"
)

PREDICTIVE_RECOMMENDATIONS_FILE = (
    PROCESSED_DATA_DIR
    / "maintenance_recommendations.csv"
)


# ============================================================
# Load Operational Maintenance Profile
# ============================================================

def load_operational_maintenance_profile():

    file_path = OPERATIONAL_PROFILE_FILE

    df = pd.read_csv(
        file_path
    )

    return df


# ============================================================
# Load Predictive Maintenance Recommendations
# ============================================================

def load_predictive_recommendations():

    file_path = PREDICTIVE_RECOMMENDATIONS_FILE

    df = pd.read_csv(
        file_path
    )

    return df


# ============================================================
# Find Operational Maintenance Information
# ============================================================

def get_operational_maintenance_info(
    vehicle_id,
    df
):

    vehicle_data = (
        df[
            df["vehicle_id"] == vehicle_id
        ]
    )

    if vehicle_data.empty:

        return None

    return vehicle_data.iloc[0].to_dict()


# ============================================================
# Find Predictive Maintenance Information
# ============================================================

def get_predictive_maintenance_info(
    vehicle_id,
    df
):

    vehicle_data = (
        df[
            df["vehicle_id"] == vehicle_id
        ]
    )

    if vehicle_data.empty:

        return None

    return vehicle_data.iloc[0].to_dict()


# ============================================================
# Calculate Days Since Last Maintenance
# ============================================================

def calculate_days_since_last_maintenance(
    last_service_date
):

    last_service_date = pd.to_datetime(
        last_service_date,
        errors="coerce"
    )

    if pd.isna(last_service_date):

        return None

    current_date = pd.Timestamp.now().normalize()

    days_since = (
        current_date
        - last_service_date.normalize()
    ).days

    return max(
        days_since,
        0
    )


# ============================================================
# Build Operational Maintenance History
# ============================================================

def build_operational_history(
    operational_data
):

    days_since_last_maintenance = (
        calculate_days_since_last_maintenance(
            operational_data[
                "last_service_date"
            ]
        )
    )

    return {

        "maintenance_count": int(
            operational_data[
                "maintenance_count"
            ]
        ),

        "failure_count": int(
            operational_data[
                "failure_count"
            ]
        ),

        "known_failure_records": int(
            operational_data[
                "known_failure_records"
            ]
        ),

        "unknown_failure_count": int(
            operational_data[
                "unknown_failure_count"
            ]
        ),

        "failure_rate": round(
            float(
                operational_data[
                    "failure_rate"
                ]
            ) * 100,
            2
        ),

        "average_maintenance_cost": round(
            float(
                operational_data[
                    "average_maintenance_cost"
                ]
            ),
            2
        ),

        "average_downtime_hours": round(
            float(
                operational_data[
                    "average_downtime_hours"
                ]
            ),
            2
        ),

        "average_labour_hours": round(
            float(
                operational_data[
                    "average_labour_hours"
                ]
            ),
            2
        ),

        "first_service_date": (
            operational_data[
                "first_service_date"
            ]
        ),

        "last_service_date": (
            operational_data[
                "last_service_date"
            ]
        ),

        "last_service_type": (
            operational_data[
                "last_service_type"
            ]
        ),

        "last_failure_reported": (
            operational_data[
                "last_failure_reported"
            ]
        ),

        "days_since_last_maintenance": (
            days_since_last_maintenance
        )
    }


# ============================================================
# Build Operational Recommendation
# ============================================================

def build_operational_recommendation(
    operational_data
):

    history_level = (
        operational_data[
            "history_level"
        ]
    )

    evidence_level = (
        operational_data[
            "evidence_level"
        ]
    )

    maintenance_count = int(
        operational_data[
            "maintenance_count"
        ]
    )

    unknown_failure_count = int(
        operational_data[
            "unknown_failure_count"
        ]
    )

    # --------------------------------------------------------
    # Limited history
    # --------------------------------------------------------

    if history_level == "LIMITED":

        maintenance_category = (
            "LIMITED_HISTORY"
        )

        primary_action = (
            "Continue maintenance monitoring; "
            "insufficient historical records for "
            "strong predictive assessment."
        )

    # --------------------------------------------------------
    # Partial evidence
    # --------------------------------------------------------

    elif evidence_level == "PARTIAL":

        maintenance_category = (
            "PARTIAL_HISTORY"
        )

        primary_action = (
            "Review maintenance history and "
            "continue monitoring because some "
            "failure information is unknown."
        )

    # --------------------------------------------------------
    # Strong evidence
    # --------------------------------------------------------

    else:

        maintenance_category = (
            "ROUTINE_MONITORING"
        )

        primary_action = (
            "Continue routine maintenance monitoring "
            "based on available maintenance history."
        )

    # --------------------------------------------------------
    # Failure action
    # --------------------------------------------------------

    failure_count = int(
        operational_data[
            "failure_count"
        ]
    )

    if failure_count > 0:

        failure_action = (
            "Review previous failure incidents "
            "before assigning the vehicle."
        )

    elif unknown_failure_count > 0:

        failure_action = (
            "Failure history contains unknown records; "
            "verify maintenance records."
        )

    else:

        failure_action = (
            "No confirmed maintenance failures "
            "recorded in the available history."
        )

    # --------------------------------------------------------
    # Downtime action
    # --------------------------------------------------------

    average_downtime = float(
        operational_data[
            "average_downtime_hours"
        ]
    )

    if average_downtime >= 48:

        downtime_action = (
            "High historical downtime observed; "
            "review vehicle availability before assignment."
        )

    else:

        downtime_action = (
            "Historical downtime is available for "
            "operational monitoring."
        )

    # --------------------------------------------------------
    # Cost action
    # --------------------------------------------------------

    average_cost = float(
        operational_data[
            "average_maintenance_cost"
        ]
    )

    if average_cost >= 30000:

        cost_action = (
            "Maintenance cost is relatively high; "
            "monitor future maintenance expenditure."
        )

    else:

        cost_action = (
            "Continue monitoring maintenance expenditure."
        )

    # --------------------------------------------------------
    # Recency action
    # --------------------------------------------------------

    last_service_date = pd.to_datetime(
        operational_data[
            "last_service_date"
        ],
        errors="coerce"
    )

    if pd.isna(last_service_date):

        recency_action = (
            "Last maintenance date is unavailable."
        )

    else:

        days_since = (
            calculate_days_since_last_maintenance(
                last_service_date
            )
        )

        if days_since > 180:

            recency_action = (
                "Vehicle has not been serviced recently; "
                "verify maintenance status before assignment."
            )

        else:

            recency_action = (
                "Recent maintenance history is available."
            )

    # --------------------------------------------------------
    # Overall recommendation
    # --------------------------------------------------------

    if history_level == "LIMITED":

        overall_recommendation = (
            "Vehicle has limited maintenance history. "
            "Use available records for operational monitoring "
            "but avoid treating the vehicle as having a strong "
            "predictive maintenance history."
        )

    elif evidence_level == "PARTIAL":

        overall_recommendation = (
            "Vehicle has partial maintenance evidence because "
            "some failure information is unknown. Verify the "
            "maintenance records before making a high-confidence "
            "maintenance decision."
        )

    else:

        overall_recommendation = (
            "Vehicle has sufficient historical maintenance "
            "records for operational monitoring. Continue "
            "routine maintenance assessment."
        )

    return {

        "maintenance_category":
            maintenance_category,

        "primary_action":
            primary_action,

        "failure_action":
            failure_action,

        "downtime_action":
            downtime_action,

        "cost_action":
            cost_action,

        "recency_action":
            recency_action,

        "overall_recommendation":
            overall_recommendation
    }


# ============================================================
# Build Maintenance Agent Response
# ============================================================

def build_maintenance_response(
    operational_data,
    predictive_data
):

    # --------------------------------------------------------
    # Vehicle not found in operational history
    # --------------------------------------------------------

    if operational_data is None:

        return {

            "status": "not_found",

            "message": (
                "No maintenance history found "
                "for the given vehicle."
            )
        }

    # --------------------------------------------------------
    # Operational maintenance history
    # --------------------------------------------------------

    maintenance_history = (
        build_operational_history(
            operational_data
        )
    )

    # --------------------------------------------------------
    # Operational recommendation
    # --------------------------------------------------------

    operational_recommendation = (
        build_operational_recommendation(
            operational_data
        )
    )

    # --------------------------------------------------------
    # Base response
    # --------------------------------------------------------

    response = {

        "status": "success",

        "vehicle_id": (
            operational_data[
                "vehicle_id"
            ]
        ),

        "evidence": {

            "history_level": (
                operational_data[
                    "history_level"
                ]
            ),

            "evidence_level": (
                operational_data[
                    "evidence_level"
                ]
            )
        },

        "maintenance_history":
            maintenance_history,

        "recommendation":
            operational_recommendation
    }

    # --------------------------------------------------------
    # Predictive risk available
    # --------------------------------------------------------

    if predictive_data is not None:

        response["predictive_maintenance"] = {

            "available": True,

            "risk_score": round(
                float(
                    predictive_data[
                        "risk_score"
                    ]
                ),
                2
            ),

            "risk_level": (
                predictive_data[
                    "risk_level"
                ]
            ),

            "priority": (
                predictive_data[
                    "priority"
                ]
            )
        }

        # ----------------------------------------------------
        # Use predictive recommendation where available
        # ----------------------------------------------------

        response[
            "predictive_recommendation"
        ] = {

            "maintenance_category":
                predictive_data[
                    "maintenance_category"
                ],

            "primary_action":
                predictive_data[
                    "primary_action"
                ],

            "failure_action":
                predictive_data[
                    "failure_action"
                ],

            "downtime_action":
                predictive_data[
                    "downtime_action"
                ],

            "cost_action":
                predictive_data[
                    "cost_action"
                ],

            "recency_action":
                predictive_data[
                    "recency_action"
                ],

            "overall_recommendation":
                predictive_data[
                    "overall_recommendation"
                ]
        }

    # --------------------------------------------------------
    # Predictive risk unavailable
    # --------------------------------------------------------

    else:

        response["predictive_maintenance"] = {

            "available": False,

            "message": (
                "Predictive maintenance assessment "
                "is unavailable for this vehicle because "
                "sufficient temporal training history "
                "was not available."
            )
        }

    return response


# ============================================================
# Maintenance Agent
# ============================================================

def maintenance_agent(vehicle_id):

    # --------------------------------------------------------
    # Load operational profile
    # --------------------------------------------------------

    operational_df = (
        load_operational_maintenance_profile()
    )

    # --------------------------------------------------------
    # Load predictive recommendations
    # --------------------------------------------------------

    predictive_df = (
        load_predictive_recommendations()
    )

    # --------------------------------------------------------
    # Find operational information
    # --------------------------------------------------------

    operational_data = (
        get_operational_maintenance_info(
            vehicle_id,
            operational_df
        )
    )

    # --------------------------------------------------------
    # Find predictive information
    # --------------------------------------------------------

    predictive_data = (
        get_predictive_maintenance_info(
            vehicle_id,
            predictive_df
        )
    )

    # --------------------------------------------------------
    # Build final response
    # --------------------------------------------------------

    response = build_maintenance_response(
        operational_data,
        predictive_data
    )

    return response


# ============================================================
# Display Maintenance Agent Result
# ============================================================

def display_maintenance_result(response):

    print("\n" + "=" * 70)
    print("SMARTLOGIX AI - MAINTENANCE AGENT")
    print("=" * 70)

    print(
        f"\nStatus: {response['status']}"
    )

    if response["status"] != "success":

        print(
            response["message"]
        )

        return

    # --------------------------------------------------------
    # Vehicle
    # --------------------------------------------------------

    print(
        f"\nVehicle ID: "
        f"{response['vehicle_id']}"
    )

    # --------------------------------------------------------
    # Evidence
    # --------------------------------------------------------

    print("\n--- Maintenance Evidence ---")

    print(
        f"History Level: "
        f"{response['evidence']['history_level']}"
    )

    print(
        f"Evidence Level: "
        f"{response['evidence']['evidence_level']}"
    )

    # --------------------------------------------------------
    # Predictive Maintenance
    # --------------------------------------------------------

    print(
        "\n--- Predictive Maintenance ---"
    )

    predictive = (
        response[
            "predictive_maintenance"
        ]
    )

    if predictive["available"]:

        print(
            f"Risk Score: "
            f"{predictive['risk_score']}"
        )

        print(
            f"Risk Level: "
            f"{predictive['risk_level']}"
        )

        print(
            f"Priority: "
            f"{predictive['priority']}"
        )

    else:

        print(
            "Predictive assessment: "
            "Unavailable"
        )

        print(
            predictive["message"]
        )

    # --------------------------------------------------------
    # Maintenance History
    # --------------------------------------------------------

    history = (
        response[
            "maintenance_history"
        ]
    )

    print("\n--- Maintenance History ---")

    print(
        f"Maintenance Count: "
        f"{history['maintenance_count']}"
    )

    print(
        f"Failure Count: "
        f"{history['failure_count']}"
    )

    print(
        f"Known Failure Records: "
        f"{history['known_failure_records']}"
    )

    print(
        f"Unknown Failure Records: "
        f"{history['unknown_failure_count']}"
    )

    print(
        f"Failure Rate: "
        f"{history['failure_rate']}%"
    )

    print(
        f"Average Maintenance Cost: ₹"
        f"{history['average_maintenance_cost']}"
    )

    print(
        f"Average Downtime: "
        f"{history['average_downtime_hours']} hours"
    )

    print(
        f"Average Labour Hours: "
        f"{history['average_labour_hours']} hours"
    )

    print(
        f"First Service Date: "
        f"{history['first_service_date']}"
    )

    print(
        f"Last Service Date: "
        f"{history['last_service_date']}"
    )

    print(
        f"Last Service Type: "
        f"{history['last_service_type']}"
    )

    print(
        f"Last Failure Reported: "
        f"{history['last_failure_reported']}"
    )

    print(
        f"Days Since Last Maintenance: "
        f"{history['days_since_last_maintenance']}"
    )

    # --------------------------------------------------------
    # Operational Recommendation
    # --------------------------------------------------------

    recommendation = (
        response[
            "recommendation"
        ]
    )

    print(
        "\n--- Operational Recommendation ---"
    )

    print(
        f"Category: "
        f"{recommendation['maintenance_category']}"
    )

    print(
        f"Primary Action: "
        f"{recommendation['primary_action']}"
    )

    print(
        f"Failure Action: "
        f"{recommendation['failure_action']}"
    )

    print(
        f"Downtime Action: "
        f"{recommendation['downtime_action']}"
    )

    print(
        f"Cost Action: "
        f"{recommendation['cost_action']}"
    )

    print(
        f"Recency Action: "
        f"{recommendation['recency_action']}"
    )

    print(
        f"\nOverall Recommendation:\n"
        f"{recommendation['overall_recommendation']}"
    )

    # --------------------------------------------------------
    # Predictive Recommendation
    # --------------------------------------------------------

    if "predictive_recommendation" in response:

        predictive_recommendation = (
            response[
                "predictive_recommendation"
            ]
        )

        print(
            "\n--- Predictive Recommendation ---"
        )

        print(
            f"Category: "
            f"{predictive_recommendation['maintenance_category']}"
        )

        print(
            f"Primary Action: "
            f"{predictive_recommendation['primary_action']}"
        )

        print(
            f"Overall Recommendation:\n"
            f"{predictive_recommendation['overall_recommendation']}"
        )

    print("\n" + "=" * 70)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    sample_vehicle_id = "VEH-0274"

    result = maintenance_agent(
        sample_vehicle_id
    )

    display_maintenance_result(
        result
    )