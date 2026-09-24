# ============================================================
# SMARTLOGIX AI - VEHICLE ASSIGNMENT AGENT
# ============================================================
"""
Select a fleet vehicle for a delivery when historical route matching
has not already produced a trustworthy vehicle assignment.

Design principles:
1. Historical route matching and vehicle assignment are separate concerns.
2. UNASSIGNED is treated as a valid intermediate state, not a vehicle ID.
3. Fleet capacity and transport mode are hard constraints when available.
4. Maintenance intelligence is used as an assignment signal, not as a
   replacement for fleet availability/capacity checks.
5. Unknown/missing fleet metadata is reported explicitly instead of being
   silently assumed.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config.config import PROCESSED_DATA_DIR


# ============================================================
# CONFIGURATION
# ============================================================

FLEET_FILE = PROCESSED_DATA_DIR / "fleet_vehicles_cleaned.csv"
MAINTENANCE_PROFILE_FILE = (
    PROCESSED_DATA_DIR / "operational_maintenance_profile.csv"
)
MAINTENANCE_RISK_FILE = (
    PROCESSED_DATA_DIR / "vehicle_maintenance_risk.csv"
)

TOP_CANDIDATE_COUNT = 10

# Maintenance should influence ranking, but should not override hard
# fleet constraints.
MAINTENANCE_SCORE_WEIGHT = 0.35
CAPACITY_SCORE_WEIGHT = 0.25
AVAILABILITY_SCORE_WEIGHT = 0.25
HISTORY_SCORE_WEIGHT = 0.15


# ============================================================
# COLUMN RESOLUTION
# ============================================================

def _normalize_column_name(column):
    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def _build_column_map(df):
    return {
        _normalize_column_name(column): column
        for column in df.columns
    }


def _find_column(df, candidates):
    column_map = _build_column_map(df)

    for candidate in candidates:
        normalized = _normalize_column_name(candidate)
        if normalized in column_map:
            return column_map[normalized]

    return None


# ============================================================
# LOAD DATA
# ============================================================

def load_fleet_data():
    if not FLEET_FILE.exists():
        raise FileNotFoundError(
            f"Fleet dataset not found: {FLEET_FILE}"
        )

    df = pd.read_csv(FLEET_FILE)

    if df.empty:
        raise ValueError("Fleet dataset is empty.")

    return df


def load_maintenance_intelligence():
    profile = pd.DataFrame()
    risk = pd.DataFrame()

    if MAINTENANCE_PROFILE_FILE.exists():
        profile = pd.read_csv(MAINTENANCE_PROFILE_FILE)

    if MAINTENANCE_RISK_FILE.exists():
        risk = pd.read_csv(MAINTENANCE_RISK_FILE)

    return profile, risk


# ============================================================
# FLEET STANDARDIZATION
# ============================================================

def standardize_fleet_data(df):
    """Map common fleet-schema variants to a stable internal schema."""

    vehicle_id_col = _find_column(
        df,
        ["vehicle_id", "vehicleid", "id"]
    )

    if vehicle_id_col is None:
        raise ValueError(
            "Fleet dataset does not contain a vehicle ID column."
        )

    transport_col = _find_column(
        df,
        [
            "transport_mode",
            "transportation_mode",
            "mode",
            "vehicle_mode",
            "vehicle_type"
        ]
    )

    capacity_col = _find_column(
        df,
        [
            "payload_capacity_kg",
            "payload_capacity",
            "capacity_kg",
            "max_payload_kg",
            "max_payload",
            "load_capacity_kg",
            "capacity"
        ]
    )

    status_col = _find_column(
        df,
        [
            "availability_status",
            "availability",
            "vehicle_status",
            "status",
            "operational_status",
            "fleet_status"
        ]
    )

    active_col = _find_column(
        df,
        [
            "is_available",
            "available",
            "active",
            "is_active"
        ]
    )

    fragile_col = _find_column(
        df,
        [
            "supports_fragile",
            "fragile_supported",
            "fragile_capable"
        ]
    )

    hazmat_col = _find_column(
        df,
        [
            "supports_hazmat",
            "hazmat_supported",
            "hazmat_capable"
        ]
    )

    cold_chain_col = _find_column(
        df,
        [
            "supports_cold_chain",
            "cold_chain_supported",
            "cold_chain_capable"
        ]
    )

    result = pd.DataFrame(index=df.index)

    result["vehicle_id"] = (
        df[vehicle_id_col]
        .astype(str)
        .str.strip()
    )

    if transport_col is not None:
        result["transport_mode"] = (
            df[transport_col]
            .astype(str)
            .str.strip()
        )
    else:
        result["transport_mode"] = "UNKNOWN"

    if capacity_col is not None:
        result["capacity_kg"] = pd.to_numeric(
            df[capacity_col],
            errors="coerce"
        )
    else:
        result["capacity_kg"] = np.nan

    if status_col is not None:
        result["availability_status"] = (
            df[status_col]
            .astype(str)
            .str.strip()
        )
    else:
        result["availability_status"] = "UNKNOWN"

    if active_col is not None:
        result["is_available"] = (
            df[active_col]
            .apply(_parse_boolean)
        )
    else:
        result["is_available"] = pd.NA

    if fragile_col is not None:
        result["supports_fragile"] = (
            df[fragile_col]
            .apply(_parse_boolean)
        )
    else:
        result["supports_fragile"] = pd.NA

    if hazmat_col is not None:
        result["supports_hazmat"] = (
            df[hazmat_col]
            .apply(_parse_boolean)
        )
    else:
        result["supports_hazmat"] = pd.NA

    if cold_chain_col is not None:
        result["supports_cold_chain"] = (
            df[cold_chain_col]
            .apply(_parse_boolean)
        )
    else:
        result["supports_cold_chain"] = pd.NA

    return result.drop_duplicates(
        subset=["vehicle_id"]
    ).reset_index(drop=True)


def _parse_boolean(value):
    if pd.isna(value):
        return pd.NA

    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()

    if normalized in {"1", "true", "yes", "y", "available", "active"}:
        return True

    if normalized in {"0", "false", "no", "n", "unavailable", "inactive"}:
        return False

    return pd.NA


def _normalize_transport_mode(value):
    value = str(value).strip().lower()

    aliases = {
        "truck": "truck",
        "trucks": "truck",
        "lorry": "truck",
        "van": "van",
        "bike": "bike",
        "motorbike": "bike",
        "motorcycle": "bike",
        "drone": "drone",
        "ship": "ship",
        "air cargo": "air cargo",
        "air_cargo": "air cargo",
        "air": "air cargo"
    }

    return aliases.get(value, value)


# ============================================================
# MAINTENANCE STANDARDIZATION
# ============================================================

def build_maintenance_index(profile, risk):
    """Combine operational and predictive maintenance intelligence."""

    if profile.empty and risk.empty:
        return pd.DataFrame(
            columns=[
                "vehicle_id",
                "maintenance_score",
                "history_score",
                "risk_score",
                "risk_level",
                "priority",
                "history_level",
                "evidence_level"
            ]
        )

    if not profile.empty:
        profile = profile.copy()
        profile["vehicle_id"] = (
            profile["vehicle_id"]
            .astype(str)
            .str.strip()
        )

    if not risk.empty:
        risk = risk.copy()
        risk["vehicle_id"] = (
            risk["vehicle_id"]
            .astype(str)
            .str.strip()
        )

    if profile.empty:
        merged = risk.copy()
    elif risk.empty:
        merged = profile.copy()
    else:
        risk_columns = [
            column
            for column in [
                "vehicle_id",
                "risk_score",
                "risk_level",
                "priority"
            ]
            if column in risk.columns
        ]

        merged = profile.merge(
            risk[risk_columns],
            on="vehicle_id",
            how="outer"
        )

    if "risk_score" not in merged.columns:
        merged["risk_score"] = np.nan

    if "risk_level" not in merged.columns:
        merged["risk_level"] = "UNKNOWN"

    if "priority" not in merged.columns:
        merged["priority"] = "UNKNOWN"

    if "history_level" not in merged.columns:
        merged["history_level"] = "UNKNOWN"

    if "evidence_level" not in merged.columns:
        merged["evidence_level"] = "UNKNOWN"

    # Lower predictive risk is better.
    risk_score = pd.to_numeric(
        merged["risk_score"],
        errors="coerce"
    )

    merged["maintenance_score"] = (
        1.0 - (risk_score.fillna(50.0).clip(0, 100) / 100.0)
    )

    history_map = {
        "STRONG": 1.0,
        "MODERATE": 0.65,
        "LIMITED": 0.35
    }

    merged["history_score"] = (
        merged["history_level"]
        .astype(str)
        .str.upper()
        .map(history_map)
        .fillna(0.50)
    )

    return merged[
        [
            "vehicle_id",
            "maintenance_score",
            "history_score",
            "risk_score",
            "risk_level",
            "priority",
            "history_level",
            "evidence_level"
        ]
    ].drop_duplicates(
        subset=["vehicle_id"]
    )


# ============================================================
# HARD CONSTRAINTS
# ============================================================

def _availability_mask(df):
    """Apply availability only when the dataset actually exposes it."""

    if df.empty:
        return pd.Series(dtype=bool)

    status = (
        df["availability_status"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    known_unavailable = {
        "unavailable",
        "inactive",
        "maintenance",
        "under maintenance",
        "out_of_service",
        "out of service",
        "retired",
        "offline",
        "busy"
    }

    status_available = ~status.isin(known_unavailable)

    explicit_available = df["is_available"].isna()

    explicit_available = explicit_available | (
        df["is_available"] == True
    )

    return status_available & explicit_available


def _transport_mask(df, requested_mode):
    if requested_mode is None:
        return pd.Series(True, index=df.index)

    requested = _normalize_transport_mode(requested_mode)

    modes = (
        df["transport_mode"]
        .map(_normalize_transport_mode)
    )

    # If fleet transport metadata is absent/unknown, do not silently
    # reject every vehicle. The assignment result will report that the
    # transport-mode constraint could not be verified.
    known_mode = modes != "unknown"

    return (~known_mode) | (modes == requested)


def _payload_mask(df, package_weight):
    if package_weight is None:
        return pd.Series(True, index=df.index)

    weight = float(package_weight)
    capacity = pd.to_numeric(
        df["capacity_kg"],
        errors="coerce"
    )

    # Missing capacity is not treated as infinite capacity. It is kept as
    # a candidate only so the result can explicitly report missing metadata.
    return capacity.isna() | (capacity >= weight)


def _special_constraint_mask(df, order_data):
    mask = pd.Series(True, index=df.index)

    constraints = [
        ("is_fragile", "supports_fragile"),
        ("is_hazmat", "supports_hazmat"),
        ("cold_chain_required", "supports_cold_chain")
    ]

    for order_column, fleet_column in constraints:
        required = _parse_boolean(
            order_data.get(order_column, False)
        )

        if required is not True:
            continue

        supported = df[fleet_column]

        # If support metadata exists, enforce it. If missing, keep the
        # candidate but flag verification as unresolved later.
        mask &= supported.isna() | (supported == True)

    return mask


# ============================================================
# CANDIDATE SCORING
# ============================================================

def _calculate_capacity_score(capacity, package_weight):
    if pd.isna(capacity):
        return 0.50

    if package_weight is None:
        return 0.50

    capacity = float(capacity)
    weight = float(package_weight)

    if capacity <= 0:
        return 0.0

    utilization = weight / capacity

    if utilization > 1:
        return 0.0

    # Prefer vehicles with enough spare capacity without requiring
    # unnecessary over-capacity.
    return float(max(0.0, 1.0 - utilization))


def _calculate_availability_score(status, is_available):
    if is_available is True:
        return 1.0

    if is_available is False:
        return 0.0

    status = str(status).strip().lower()

    if status in {"available", "ready", "active", "idle", "operational"}:
        return 1.0

    if status in {"busy", "assigned"}:
        return 0.25

    if status == "unknown":
        return 0.50

    return 0.50


def score_candidates(candidates, package_weight):
    candidates = candidates.copy()

    candidates["capacity_score"] = candidates.apply(
        lambda row: _calculate_capacity_score(
            row["capacity_kg"],
            package_weight
        ),
        axis=1
    )

    candidates["availability_score"] = candidates.apply(
        lambda row: _calculate_availability_score(
            row["availability_status"],
            row["is_available"]
        ),
        axis=1
    )

    candidates["assignment_score"] = (
        MAINTENANCE_SCORE_WEIGHT * candidates["maintenance_score"]
        + CAPACITY_SCORE_WEIGHT * candidates["capacity_score"]
        + AVAILABILITY_SCORE_WEIGHT * candidates["availability_score"]
        + HISTORY_SCORE_WEIGHT * candidates["history_score"]
    )

    return candidates.sort_values(
        ["assignment_score", "capacity_score", "maintenance_score"],
        ascending=[False, False, False]
    ).reset_index(drop=True)


# ============================================================
# ASSIGNMENT
# ============================================================

def assign_vehicle(order_data, preferred_vehicle_id=None):
    """
    Return a fleet assignment decision.

    preferred_vehicle_id is used when a trustworthy historical route has
    already selected a vehicle. This function validates that vehicle instead
    of blindly replacing it.
    """

    print("\n" + "=" * 70)
    print("VEHICLE ASSIGNMENT")
    print("=" * 70)

    order_data = dict(order_data)

    requested_mode = order_data.get("transport_mode")
    package_weight = order_data.get("package_weight_kg")

    fleet_raw = load_fleet_data()
    fleet = standardize_fleet_data(fleet_raw)

    profile, risk = load_maintenance_intelligence()
    maintenance = build_maintenance_index(profile, risk)

    fleet = fleet.merge(
        maintenance,
        on="vehicle_id",
        how="left"
    )

    fleet["maintenance_score"] = fleet[
        "maintenance_score"
    ].fillna(0.50)

    fleet["history_score"] = fleet[
        "history_score"
    ].fillna(0.50)

    fleet["risk_level"] = fleet[
        "risk_level"
    ].fillna("UNKNOWN")

    fleet["priority"] = fleet[
        "priority"
    ].fillna("UNKNOWN")

    fleet["history_level"] = fleet[
        "history_level"
    ].fillna("UNKNOWN")

    fleet["evidence_level"] = fleet[
        "evidence_level"
    ].fillna("UNKNOWN")

    # --------------------------------------------------------
    # PREFERRED VEHICLE VALIDATION
    # --------------------------------------------------------

    if (
        preferred_vehicle_id
        and str(preferred_vehicle_id).upper() != "UNASSIGNED"
    ):
        preferred = fleet[
            fleet["vehicle_id"].astype(str) == str(preferred_vehicle_id)
        ].copy()

        if not preferred.empty:
            preferred = preferred[
                _availability_mask(preferred)
            ]
            preferred = preferred[
                _transport_mask(preferred, requested_mode)
            ]
            preferred = preferred[
                _payload_mask(preferred, package_weight)
            ]
            preferred = preferred[
                _special_constraint_mask(preferred, order_data)
            ]

            if not preferred.empty:
                preferred = score_candidates(
                    preferred,
                    package_weight
                )
                selected = preferred.iloc[0]
                return _build_assignment_result(
                    selected,
                    requested_mode,
                    package_weight,
                    source="HISTORICAL_ROUTE"
                )

    # --------------------------------------------------------
    # HARD-CONSTRAINT FILTERING
    # --------------------------------------------------------

    candidates = fleet.copy()
    initial_count = len(candidates)

    candidates = candidates[
        _availability_mask(candidates)
    ]
    after_availability = len(candidates)

    candidates = candidates[
        _transport_mask(candidates, requested_mode)
    ]
    after_transport = len(candidates)

    candidates = candidates[
        _payload_mask(candidates, package_weight)
    ]
    after_payload = len(candidates)

    candidates = candidates[
        _special_constraint_mask(candidates, order_data)
    ]
    after_special = len(candidates)

    if candidates.empty:
        return {
            "status": "not_assigned",
            "vehicle_id": None,
            "reason": "No fleet vehicle satisfied the available assignment constraints.",
            "requested_transport_mode": requested_mode,
            "package_weight_kg": package_weight,
            "fleet_count": initial_count,
            "filter_counts": {
                "after_availability": after_availability,
                "after_transport_mode": after_transport,
                "after_payload": after_payload,
                "after_special_constraints": after_special
            }
        }

    candidates = score_candidates(
        candidates,
        package_weight
    )

    top_candidates = candidates.head(
        TOP_CANDIDATE_COUNT
    ).copy()

    selected = top_candidates.iloc[0]

    result = _build_assignment_result(
        selected,
        requested_mode,
        package_weight,
        source="FLEET_ASSIGNMENT"
    )

    result["candidate_count"] = int(len(candidates))
    result["top_candidates"] = (
        top_candidates[
            [
                "vehicle_id",
                "transport_mode",
                "capacity_kg",
                "availability_status",
                "assignment_score",
                "maintenance_score",
                "risk_level",
                "priority",
                "history_level",
                "evidence_level"
            ]
        ]
        .round(4)
        .to_dict(orient="records")
    )

    return result


def _build_assignment_result(
    selected,
    requested_mode,
    package_weight,
    source
):
    capacity = selected["capacity_kg"]

    if pd.isna(capacity):
        capacity_utilization = None
        capacity_status = "NOT_VERIFIED"
    else:
        capacity = float(capacity)
        weight = float(package_weight) if package_weight is not None else None

        if weight is None or capacity <= 0:
            capacity_utilization = None
            capacity_status = "NOT_VERIFIED"
        else:
            capacity_utilization = round(
                (weight / capacity) * 100,
                2
            )
            capacity_status = "PASS"

    transport_verified = (
        _normalize_transport_mode(selected["transport_mode"])
        == _normalize_transport_mode(requested_mode)
        if requested_mode is not None
        and str(selected["transport_mode"]).upper() != "UNKNOWN"
        else None
    )

    special_metadata = {
        "fragile": _metadata_state(
            selected["supports_fragile"]
        ),
        "hazmat": _metadata_state(
            selected["supports_hazmat"]
        ),
        "cold_chain": _metadata_state(
            selected["supports_cold_chain"]
        )
    }

    return {
        "status": "assigned",
        "vehicle_id": str(selected["vehicle_id"]),
        "assignment_source": source,
        "transport_mode": str(selected["transport_mode"]),
        "requested_transport_mode": requested_mode,
        "availability_status": str(selected["availability_status"]),
        "capacity_kg": (
            None
            if pd.isna(selected["capacity_kg"])
            else round(float(selected["capacity_kg"]), 3)
        ),
        "package_weight_kg": (
            None
            if package_weight is None
            else float(package_weight)
        ),
        "capacity_utilization_pct": capacity_utilization,
        "capacity_status": capacity_status,
        "transport_mode_verified": transport_verified,
        "special_constraints": special_metadata,
        "assignment_score": round(
            float(selected["assignment_score"]),
            4
        ),
        "maintenance": {
            "risk_score": _safe_float(selected["risk_score"]),
            "risk_level": str(selected["risk_level"]),
            "priority": str(selected["priority"]),
            "history_level": str(selected["history_level"]),
            "evidence_level": str(selected["evidence_level"])
        }
    }


def _metadata_state(value):
    if pd.isna(value):
        return "UNKNOWN"

    return "SUPPORTED" if bool(value) else "NOT_SUPPORTED"


def _safe_float(value):
    if pd.isna(value):
        return None

    return round(float(value), 2)


# ============================================================
# STANDALONE TEST
# ============================================================

def main():
    sample_order = {
        "transport_mode": "Truck",
        "package_weight_kg": 3.5,
        "is_fragile": 0,
        "is_hazmat": 0,
        "cold_chain_required": 0
    }

    result = assign_vehicle(
        sample_order
    )

    print("\nAssignment result:")
    print(result)


if __name__ == "__main__":
    main()
