
"""
SmartLogix AI - Historical Route Optimizer

Provides:
- GPS cleaning
- Historical route summarisation
- Strict route matching
- Relaxed historical-route matching
- Direct geospatial fallback
- Traffic adjustment
- Route scoring/ranking
- Standardized route result

Route matching policy:

1. STRICT_MATCH
   Historical route endpoint mismatch <= 10 km

2. RELAXED_MATCH
   Historical route endpoint mismatch > 10 km
   but <= 50 km

3. DIRECT_GEODESIC_FALLBACK
   No acceptable historical route within 50 km.
   The optimizer does NOT select an unrelated historical route.
   Instead, it creates a direct origin-to-destination route estimate.

This prevents geographically unrelated historical routes from being
incorrectly used as the requested delivery route.
"""

from pathlib import Path
import math

import pandas as pd
import numpy as np

from src.config.config import PROCESSED_DATA_DIR


# ============================================================
# FILE PATHS
# ============================================================

GPS_FILE = (
    PROCESSED_DATA_DIR
    / "gps_waypoints_clean.csv"
)

TRAFFIC_FILE = (
    PROCESSED_DATA_DIR
    / "traffic_data_clean.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "route_optimization_results.csv"
)


# ============================================================
# ROUTE OPTIMIZATION CONFIGURATION
# ============================================================

ROUTE_MATCH_WEIGHT = 0.50

TRAVEL_TIME_WEIGHT = 0.50

CANDIDATE_ROUTE_COUNT = 20

EARTH_RADIUS_KM = 6371.0


# ============================================================
# ROUTE MATCHING THRESHOLDS
# ============================================================

# Historical route is considered a strict match when
# average origin/destination mismatch is <= 10 km.
MAX_ROUTE_MATCH_DISTANCE_KM = 10.0


# Historical route can be used as a relaxed match when
# mismatch is > 10 km but <= 50 km.
ALLOW_RELAXED_ROUTE_FALLBACK = True

MAX_RELAXED_ROUTE_MATCH_DISTANCE_KM = 50.0


# No historical route beyond this distance is considered
# suitable for the requested delivery.
MAX_HISTORICAL_ROUTE_FALLBACK_DISTANCE_KM = 50.0


# Speed used for the direct geospatial fallback.
# This is a baseline estimate, not a learned route speed.
DIRECT_ROUTE_BASELINE_SPEED_KMPH = 40.0


# ============================================================
# TRAFFIC CONFIGURATION
# ============================================================

BASELINE_TRAFFIC_SPEED_KMPH = 40.0


# ============================================================
# COLUMN FINDER
# ============================================================

def _find_column(
    df,
    candidates,
    required=True
):
    """
    Find a dataframe column using case-insensitive matching.
    """

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        key = (
            candidate
            .strip()
            .lower()
        )

        if key in lookup:

            return lookup[key]

    if required:

        raise KeyError(
            f"None of the expected columns were found: "
            f"{candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    return None


# ============================================================
# LOAD ROUTE DATA
# ============================================================

def load_route_data():

    print(
        "\n"
        + "=" * 60
    )

    print(
        "LOADING ROUTE DATA"
    )

    print(
        "=" * 60
    )

    gps = pd.read_csv(
        GPS_FILE
    )

    traffic = pd.read_csv(
        TRAFFIC_FILE
    )

    print(
        f"GPS data shape      : "
        f"{gps.shape}"
    )

    print(
        f"Traffic data shape  : "
        f"{traffic.shape}"
    )

    return (
        gps,
        traffic
    )


# ============================================================
# CLEAN GPS DATA
# ============================================================

def clean_gps_data(
    gps
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        "CLEANING GPS DATA"
    )

    print(
        "=" * 60
    )

    df = gps.copy()

    lat_col = _find_column(
        df,
        [
            "latitude",
            "lat"
        ]
    )

    lon_col = _find_column(
        df,
        [
            "longitude",
            "lon",
            "lng"
        ]
    )

    df[lat_col] = pd.to_numeric(
        df[lat_col],
        errors="coerce"
    )

    df[lon_col] = pd.to_numeric(
        df[lon_col],
        errors="coerce"
    )

    before = len(df)

    df = df.dropna(
        subset=[
            lat_col,
            lon_col
        ]
    ).copy()

    df = df[
        df[lat_col].between(
            -90,
            90
        )
        &
        df[lon_col].between(
            -180,
            180
        )
    ].copy()

    df = df.drop_duplicates().copy()

    print(
        f"Rows before cleaning : "
        f"{before}"
    )

    print(
        f"Rows after cleaning  : "
        f"{len(df)}"
    )

    print(
        f"Rows removed         : "
        f"{before - len(df)}"
    )

    return df


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def calculate_haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):
    """
    Calculate great-circle distance in kilometres.
    """

    lat1 = np.radians(
        lat1
    )

    lon1 = np.radians(
        lon1
    )

    lat2 = np.radians(
        lat2
    )

    lon2 = np.radians(
        lon2
    )

    dlat = (
        lat2 - lat1
    )

    dlon = (
        lon2 - lon1
    )

    a = (
        np.sin(
            dlat / 2.0
        ) ** 2
        +
        np.cos(lat1)
        *
        np.cos(lat2)
        *
        np.sin(
            dlon / 2.0
        ) ** 2
    )

    return (
        EARTH_RADIUS_KM
        *
        2.0
        *
        np.arcsin(
            np.sqrt(a)
        )
    )


# ============================================================
# CREATE ROUTE SUMMARY
# ============================================================

def create_route_summary(
    gps
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        "CREATING ROUTE SUMMARIES"
    )

    print(
        "=" * 60
    )

    df = gps.copy()

    route_col = _find_column(
        df,
        [
            "route_id",
            "route",
            "trip_id"
        ]
    )

    vehicle_col = _find_column(
        df,
        [
            "vehicle_id",
            "vehicle"
        ]
    )

    lat_col = _find_column(
        df,
        [
            "latitude",
            "lat"
        ]
    )

    lon_col = _find_column(
        df,
        [
            "longitude",
            "lon",
            "lng"
        ]
    )

    speed_col = _find_column(
        df,
        [
            "speed_kmph",
            "speed",
            "average_speed_kmph"
        ],
        required=False
    )

    summaries = []

    for route_id, group in df.groupby(
        route_col
    ):

        group = group.copy()

        group = group.sort_index()

        if len(group) < 2:

            continue

        start = group.iloc[0]

        end = group.iloc[-1]

        distance = float(
            calculate_haversine_distance(
                start[lat_col],
                start[lon_col],
                end[lat_col],
                end[lon_col]
            )
        )

        if speed_col:

            speed = pd.to_numeric(
                group[speed_col],
                errors="coerce"
            ).dropna()

            if not speed.empty:

                average_speed = float(
                    speed.mean()
                )

            else:

                average_speed = np.nan

        else:

            average_speed = np.nan

        if (
            not np.isfinite(
                average_speed
            )
            or
            average_speed <= 0
        ):

            average_speed = 30.0

        estimated_time = (
            distance
            /
            average_speed
        )

        summaries.append(
            {
                "route_id": route_id,

                "vehicle_id": (
                    group[
                        vehicle_col
                    ].iloc[0]
                ),

                "origin_lat": float(
                    start[lat_col]
                ),

                "origin_lon": float(
                    start[lon_col]
                ),

                "destination_lat": float(
                    end[lat_col]
                ),

                "destination_lon": float(
                    end[lon_col]
                ),

                "distance_km": distance,

                "average_speed_kmph": (
                    average_speed
                ),

                "estimated_time_hours": (
                    estimated_time
                ),

                "waypoint_count": (
                    len(group)
                )
            }
        )

    result = pd.DataFrame(
        summaries
    )

    print(
        f"Valid routes created : "
        f"{len(result)}"
    )

    return result


# ============================================================
# CALCULATE ROUTE MATCH DISTANCE
# ============================================================

def calculate_route_match_distance(
    route_summary,
    origin_lat,
    origin_lon,
    destination_lat,
    destination_lon
):

    df = route_summary.copy()

    origin_distance = (
        calculate_haversine_distance(
            df["origin_lat"],
            df["origin_lon"],
            origin_lat,
            origin_lon
        )
    )

    destination_distance = (
        calculate_haversine_distance(
            df["destination_lat"],
            df["destination_lon"],
            destination_lat,
            destination_lon
        )
    )

    df[
        "origin_match_distance_km"
    ] = origin_distance

    df[
        "destination_match_distance_km"
    ] = destination_distance

    # Balanced endpoint mismatch.
    df[
        "route_match_distance_km"
    ] = (
        origin_distance
        +
        destination_distance
    ) / 2.0

    return df


# ============================================================
# DIRECT GEOSPATIAL FALLBACK
# ============================================================

def create_direct_route_fallback(
    origin_lat,
    origin_lon,
    destination_lat,
    destination_lon,
    nearest_historical_distance
):
    """
    Create a route estimate directly from the requested
    origin and destination.

    This is used when no historical route is sufficiently
    close to the requested delivery.

    Important:
    vehicle_id is intentionally UNASSIGNED because a historical
    vehicle from an unrelated route must not be selected merely
    because it happens to be geographically nearest.
    """

    direct_distance = float(
        calculate_haversine_distance(
            origin_lat,
            origin_lon,
            destination_lat,
            destination_lon
        )
    )

    direct_speed = (
        DIRECT_ROUTE_BASELINE_SPEED_KMPH
    )

    direct_time = (
        direct_distance
        /
        direct_speed
    )

    return pd.DataFrame(
        [
            {
                "route_id": (
                    "DIRECT-ROUTE"
                ),

                "vehicle_id": (
                    "UNASSIGNED"
                ),

                "origin_lat": (
                    float(origin_lat)
                ),

                "origin_lon": (
                    float(origin_lon)
                ),

                "destination_lat": (
                    float(destination_lat)
                ),

                "destination_lon": (
                    float(destination_lon)
                ),

                "distance_km": (
                    direct_distance
                ),

                "average_speed_kmph": (
                    direct_speed
                ),

                "estimated_time_hours": (
                    direct_time
                ),

                "waypoint_count": 0,

                "origin_match_distance_km": 0.0,

                "destination_match_distance_km": 0.0,

                "route_match_distance_km": (
                    nearest_historical_distance
                ),

                "route_match_status": (
                    "DIRECT_GEODESIC_FALLBACK"
                )
            }
        ]
    )


# ============================================================
# SELECT CANDIDATE ROUTES
# ============================================================

def select_candidate_routes(
    route_summary,
    origin_lat,
    origin_lon,
    destination_lat,
    destination_lon,
    candidate_count=CANDIDATE_ROUTE_COUNT
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        "SELECTING CANDIDATE ROUTES"
    )

    print(
        "=" * 60
    )

    df = calculate_route_match_distance(
        route_summary,
        origin_lat,
        origin_lon,
        destination_lat,
        destination_lon
    )

    # --------------------------------------------------------
    # STRICT MATCH
    # --------------------------------------------------------

    strict = df[
        df[
            "route_match_distance_km"
        ]
        <= MAX_ROUTE_MATCH_DISTANCE_KM
    ].copy()

    print(
        f"Maximum route mismatch allowed : "
        f"{MAX_ROUTE_MATCH_DISTANCE_KM:.1f} km"
    )

    print(
        f"Strictly feasible routes       : "
        f"{len(strict)}"
    )

    if not strict.empty:

        candidates = (
            strict
            .sort_values(
                [
                    "route_match_distance_km",
                    "estimated_time_hours"
                ]
            )
            .head(
                candidate_count
            )
            .copy()
        )

        candidates[
            "route_match_status"
        ] = "STRICT_MATCH"

    # --------------------------------------------------------
    # RELAXED MATCH
    # --------------------------------------------------------

    elif ALLOW_RELAXED_ROUTE_FALLBACK:

        print(
            f"\nNo routes were found within "
            f"{MAX_ROUTE_MATCH_DISTANCE_KM:.1f} km."
        )

        print(
            "Checking relaxed historical-route range."
        )

        relaxed = df[
            df[
                "route_match_distance_km"
            ]
            <= MAX_RELAXED_ROUTE_MATCH_DISTANCE_KM
        ].copy()

        if not relaxed.empty:

            nearest = float(
                relaxed[
                    "route_match_distance_km"
                ].min()
            )

            print(
                f"Nearest historical route mismatch : "
                f"{nearest:.3f} km"
            )

            print(
                "WARNING: Using a historical route "
                "outside the strict threshold."
            )

            candidates = (
                relaxed
                .sort_values(
                    [
                        "route_match_distance_km",
                        "estimated_time_hours"
                    ]
                )
                .head(
                    candidate_count
                )
                .copy()
            )

            candidates[
                "route_match_status"
            ] = "RELAXED_MATCH"

        # ----------------------------------------------------
        # DIRECT GEOSPATIAL FALLBACK
        # ----------------------------------------------------

        else:

            nearest = float(
                df[
                    "route_match_distance_km"
                ].min()
            )

            print(
                f"Nearest historical route mismatch : "
                f"{nearest:.3f} km"
            )

            print(
                "No acceptable historical route found."
            )

            print(
                "Using direct geospatial fallback."
            )

            candidates = (
                create_direct_route_fallback(
                    origin_lat,
                    origin_lon,
                    destination_lat,
                    destination_lon,
                    nearest
                )
            )

    # --------------------------------------------------------
    # NO FALLBACK
    # --------------------------------------------------------

    else:

        raise ValueError(
            "No feasible route candidates found."
        )

    print(
        f"\nCandidate routes selected : "
        f"{len(candidates)}"
    )

    print(
        candidates[
            [
                "route_id",
                "vehicle_id",
                "route_match_distance_km",
                "route_match_status",
                "distance_km",
                "average_speed_kmph",
                "estimated_time_hours"
            ]
        ].to_string(
            index=False
        )
    )

    return candidates


# ============================================================
# PREPARE TRAFFIC CONTEXT
# ============================================================

def prepare_traffic_context(
    traffic
):

    speed_col = _find_column(
        traffic,
        [
            "speed_kmph",
            "average_speed_kmph",
            "speed"
        ],
        required=False
    )

    congestion_col = _find_column(
        traffic,
        [
            "congestion_index",
            "traffic_level",
            "congestion"
        ],
        required=False
    )

    if speed_col:

        average_speed = (
            pd.to_numeric(
                traffic[speed_col],
                errors="coerce"
            ).mean()
        )

    else:

        average_speed = (
            BASELINE_TRAFFIC_SPEED_KMPH
        )

    if congestion_col:

        average_congestion = (
            pd.to_numeric(
                traffic[congestion_col],
                errors="coerce"
            ).mean()
        )

    else:

        average_congestion = 0.0

    if (
        not np.isfinite(
            average_speed
        )
        or
        average_speed <= 0
    ):

        average_speed = (
            BASELINE_TRAFFIC_SPEED_KMPH
        )

    factor = (
        BASELINE_TRAFFIC_SPEED_KMPH
        /
        average_speed
    )

    return {
        "average_traffic_speed_kmph": (
            float(average_speed)
        ),

        "average_congestion_index": (
            float(average_congestion)
        ),

        "baseline_speed_kmph": (
            BASELINE_TRAFFIC_SPEED_KMPH
        ),

        "traffic_adjustment_factor": (
            float(factor)
        )
    }


# ============================================================
# APPLY TRAFFIC ADJUSTMENT
# ============================================================

def apply_traffic_adjustment(
    candidates,
    traffic_context
):

    df = candidates.copy()

    factor = traffic_context[
        "traffic_adjustment_factor"
    ]

    df[
        "traffic_adjusted_time_hours"
    ] = (
        df[
            "estimated_time_hours"
        ]
        *
        factor
    )

    return df


# ============================================================
# NORMALIZE COLUMN
# ============================================================

def normalize_column(
    series,
    reverse=False
):

    values = pd.to_numeric(
        series,
        errors="coerce"
    ).astype(float)

    minimum = values.min()

    maximum = values.max()

    if (
        pd.isna(minimum)
        or
        pd.isna(maximum)
    ):

        return pd.Series(
            np.zeros(
                len(series)
            ),
            index=series.index
        )

    if np.isclose(
        minimum,
        maximum
    ):

        normalized = pd.Series(
            np.zeros(
                len(series)
            ),
            index=series.index
        )

    else:

        normalized = (
            (
                values
                -
                minimum
            )
            /
            (
                maximum
                -
                minimum
            )
        )

    if reverse:

        normalized = (
            1.0
            -
            normalized
        )

    return normalized


# ============================================================
# CALCULATE ROUTE OPTIMIZATION SCORE
# ============================================================

def calculate_route_optimization_score(
    candidates
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        "CALCULATING ROUTE OPTIMIZATION SCORE"
    )

    print(
        "=" * 60
    )

    df = candidates.copy()

    match_score = normalize_column(
        df[
            "route_match_distance_km"
        ]
    )

    travel_score = normalize_column(
        df[
            "traffic_adjusted_time_hours"
        ]
    )

    df[
        "route_optimization_score"
    ] = (
        ROUTE_MATCH_WEIGHT
        *
        match_score
        +
        TRAVEL_TIME_WEIGHT
        *
        travel_score
    )

    print(
        f"Route match weight : "
        f"{ROUTE_MATCH_WEIGHT:.2f}"
    )

    print(
        f"Travel time weight : "
        f"{TRAVEL_TIME_WEIGHT:.2f}"
    )

    return df


# ============================================================
# RANK ROUTES
# ============================================================

def rank_routes(
    candidates
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        "RANKING ROUTES"
    )

    print(
        "=" * 60
    )

    df = (
        candidates
        .sort_values(
            "route_optimization_score"
        )
        .reset_index(
            drop=True
        )
    )

    df[
        "route_rank"
    ] = (
        df.index
        +
        1
    )

    columns = [
        "route_rank",
        "route_id",
        "vehicle_id",
        "route_match_distance_km",
        "route_match_status",
        "distance_km",
        "average_speed_kmph",
        "traffic_adjusted_time_hours",
        "route_optimization_score"
    ]

    print(
        df[
            columns
        ].to_string(
            index=False
        )
    )

    return df


# ============================================================
# ROUTE QUALITY METRICS
# ============================================================

def calculate_route_quality_metrics(
    ranked_routes
):

    if ranked_routes.empty:

        return {
            "candidate_count": 0,

            "strict_candidate_count": 0,

            "best_route_match_distance_km": None,

            "best_route_distance_km": None,

            "best_route_eta_hours": None
        }

    return {
        "candidate_count": int(
            len(ranked_routes)
        ),

        "strict_candidate_count": int(
            (
                ranked_routes[
                    "route_match_status"
                ]
                ==
                "STRICT_MATCH"
            ).sum()
        ),

        "best_route_match_distance_km": round(
            float(
                ranked_routes.iloc[0][
                    "route_match_distance_km"
                ]
            ),
            3
        ),

        "best_route_distance_km": round(
            float(
                ranked_routes.iloc[0][
                    "distance_km"
                ]
            ),
            3
        ),

        "best_route_eta_hours": round(
            float(
                ranked_routes.iloc[0][
                    "traffic_adjusted_time_hours"
                ]
            ),
            3
        )
    }


# ============================================================
# SELECT BEST ROUTE
# ============================================================

def select_best_route(
    ranked_routes
):

    if ranked_routes.empty:

        raise ValueError(
            "No ranked routes available."
        )

    return ranked_routes.iloc[
        0
    ].copy()


# ============================================================
# SAVE ROUTE RESULTS
# ============================================================

def save_route_results(
    ranked_routes
):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    ranked_routes.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nRoute results saved to:\n"
        f"{OUTPUT_FILE}"
    )


# ============================================================
# OPTIMIZE SINGLE ROUTE
# ============================================================

def optimize_single_route(
    origin_lat,
    origin_lon,
    destination_lat,
    destination_lon
):

    gps, traffic = (
        load_route_data()
    )

    gps = clean_gps_data(
        gps
    )

    route_summary = (
        create_route_summary(
            gps
        )
    )

    candidates = (
        select_candidate_routes(
            route_summary,
            origin_lat,
            origin_lon,
            destination_lat,
            destination_lon
        )
    )

    traffic_context = (
        prepare_traffic_context(
            traffic
        )
    )

    print(
        f"\nAverage traffic speed       : "
        f"{traffic_context['average_traffic_speed_kmph']:.2f} km/h"
    )

    print(
        f"Average congestion index    : "
        f"{traffic_context['average_congestion_index']:.2f}"
    )

    print(
        f"Baseline traffic speed      : "
        f"{traffic_context['baseline_speed_kmph']:.2f} km/h"
    )

    print(
        f"Traffic adjustment factor   : "
        f"{traffic_context['traffic_adjustment_factor']:.4f}"
    )

    candidates = (
        apply_traffic_adjustment(
            candidates,
            traffic_context
        )
    )

    candidates = (
        calculate_route_optimization_score(
            candidates
        )
    )

    ranked = (
        rank_routes(
            candidates
        )
    )

    save_route_results(
        ranked
    )

    return (
        ranked,
        traffic_context
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SMARTLOGIX AI - ROUTE OPTIMIZER"
    )

    print(
        "=" * 70
    )

    # Example request matching
    # the Master Agent.

    origin_lat = 13.0827

    origin_lon = 80.2707

    destination_lat = 12.9716

    destination_lon = 77.5946

    ranked, _ = (
        optimize_single_route(
            origin_lat,
            origin_lon,
            destination_lat,
            destination_lon
        )
    )

    best = (
        select_best_route(
            ranked
        )
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "SELECTED ROUTE"
    )

    print(
        "=" * 60
    )

    print(
        f"Route ID             : "
        f"{best['route_id']}"
    )

    print(
        f"Vehicle ID           : "
        f"{best['vehicle_id']}"
    )

    print(
        f"Match Status         : "
        f"{best['route_match_status']}"
    )

    print(
        f"Match Distance       : "
        f"{best['route_match_distance_km']:.3f} km"
    )

    print(
        f"Route Distance       : "
        f"{best['distance_km']:.3f} km"
    )


if __name__ == "__main__":
    main()
