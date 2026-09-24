# ============================================================
# SMARTLOGIX AI - DELIVERY PREDICTION AGENT
# ============================================================

from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

try:

    from src.config.config import MODELS_DIR

except ImportError:

    from config.config import MODELS_DIR


# ============================================================
# MODEL PATHS
# ============================================================

TRANSPORTATION_MODEL_PATH = (
    MODELS_DIR
    / "transportation"
    / "transportation_classifier.joblib"
)

ETA_MODEL_PATH = (
    MODELS_DIR
    / "eta"
    / "eta_regressor.joblib"
)


# ============================================================
# NORMALIZE ORDER DATA
# ============================================================

def normalize_order_data(
    order_data
):

    if isinstance(
        order_data,
        list
    ):

        if len(order_data) == 0:

            raise ValueError(
                "order_data list is empty."
            )

        if len(order_data) != 1:

            raise ValueError(
                "order_data list must contain exactly one order."
            )

        order_data = order_data[0]

    if not isinstance(
        order_data,
        dict
    ):

        raise TypeError(
            "order_data must be a dictionary "
            "or a list containing one dictionary."
        )

    return order_data


# ============================================================
# CREATE ORDER DATAFRAME
# ============================================================

def create_order_dataframe(
    order_data
):

    order_data = normalize_order_data(
        order_data
    )

    order_df = pd.DataFrame(
        [order_data]
    )

    return order_df


# ============================================================
# LOAD TRANSPORTATION MODEL
# ============================================================

def load_transportation_model():

    print(
        "\n" + "=" * 60
    )

    print(
        "LOADING TRANSPORTATION MODEL"
    )

    print(
        "=" * 60
    )

    print(
        f"\nLoading model from:\n"
        f"{TRANSPORTATION_MODEL_PATH}"
    )

    if not TRANSPORTATION_MODEL_PATH.exists():

        raise FileNotFoundError(
            "Transportation model not found:\n"
            f"{TRANSPORTATION_MODEL_PATH}"
        )

    model = joblib.load(
        TRANSPORTATION_MODEL_PATH
    )

    print(
        "\nTransportation model loaded successfully."
    )

    print(
        f"Model type: {type(model).__name__}"
    )

    return model


# ============================================================
# LOAD ETA MODEL
# ============================================================

def load_eta_model():

    print(
        "\n" + "=" * 60
    )

    print(
        "LOADING ETA MODEL"
    )

    print(
        "=" * 60
    )

    print(
        f"\nLoading model from:\n"
        f"{ETA_MODEL_PATH}"
    )

    if not ETA_MODEL_PATH.exists():

        raise FileNotFoundError(
            "ETA model not found:\n"
            f"{ETA_MODEL_PATH}"
        )

    model = joblib.load(
        ETA_MODEL_PATH
    )

    print(
        "\nETA model loaded successfully."
    )

    print(
        f"Model type: {type(model).__name__}"
    )

    return model


# ============================================================
# GET RAW INPUT FEATURES FROM PIPELINE
# ============================================================

def get_pipeline_input_features(
    pipeline
):

    # --------------------------------------------------------
    # CASE 1
    # --------------------------------------------------------
    # Pipeline itself contains feature_names_in_
    # --------------------------------------------------------

    feature_names = getattr(
        pipeline,
        "feature_names_in_",
        None
    )

    if feature_names is not None:

        return list(
            feature_names
        )

    # --------------------------------------------------------
    # CASE 2
    # --------------------------------------------------------
    # ColumnTransformer contains feature_names_in_
    # --------------------------------------------------------

    preprocessor = None

    if hasattr(
        pipeline,
        "named_steps"
    ):

        preprocessor = (
            pipeline
            .named_steps
            .get(
                "preprocessor"
            )
        )

    if preprocessor is None:

        raise ValueError(
            "Saved model does not contain a "
            "recognizable preprocessor."
        )

    feature_names = getattr(
        preprocessor,
        "feature_names_in_",
        None
    )

    if feature_names is not None:

        return list(
            feature_names
        )

    # --------------------------------------------------------
    # CASE 3
    # --------------------------------------------------------
    # Extract columns directly from fitted
    # ColumnTransformer
    # --------------------------------------------------------

    transformers = getattr(
        preprocessor,
        "transformers_",
        None
    )

    if transformers is None:

        raise ValueError(
            "Unable to determine the raw input "
            "features expected by the saved model."
        )

    feature_names = []

    for (
        transformer_name,
        transformer,
        columns
    ) in transformers:

        # Ignore remainder
        if transformer_name == "remainder":

            continue

        # Ignore dropped transformer
        if transformer == "drop":

            continue

        # Column names
        if isinstance(
            columns,
            (list, tuple)
        ):

            for column in columns:

                if isinstance(
                    column,
                    str
                ):

                    feature_names.append(
                        column
                    )

        # Pandas Index
        elif hasattr(
            columns,
            "tolist"
        ):

            values = columns.tolist()

            for column in values:

                if isinstance(
                    column,
                    str
                ):

                    feature_names.append(
                        column
                    )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    feature_names = list(
        dict.fromkeys(
            feature_names
        )
    )

    if not feature_names:

        raise ValueError(
            "Unable to determine the raw input "
            "features expected by the saved model."
        )

    return feature_names


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def prepare_model_input(
    order_df,
    pipeline,
    target_column=None,
    predicted_transport_mode=None
):

    print(
        "\n" + "-" * 60
    )

    print(
        "PREPARING MODEL INPUT"
    )

    print(
        "-" * 60
    )

    expected_features = (
        get_pipeline_input_features(
            pipeline
        )
    )

    print(
        "\nExpected raw features:"
    )

    print(
        expected_features
    )

    # --------------------------------------------------------
    # COPY ORDER DATA
    # --------------------------------------------------------

    model_df = (
        order_df
        .copy()
    )

    # --------------------------------------------------------
    # ADD PREDICTED TRANSPORT MODE
    # --------------------------------------------------------
    # ETA model may require transport_mode.
    # --------------------------------------------------------

    if (
        predicted_transport_mode is not None
    ):

        model_df[
            "transport_mode"
        ] = predicted_transport_mode

    # --------------------------------------------------------
    # REMOVE TARGET FOR TRANSPORTATION MODEL
    # --------------------------------------------------------

    if (
        target_column is not None
        and
        target_column in model_df.columns
    ):

        model_df = model_df.drop(
            columns=[
                target_column
            ]
        )

    # --------------------------------------------------------
    # CREATE MISSING EXPECTED FEATURES
    # --------------------------------------------------------
    # Missing values can be handled by the
    # saved preprocessing pipeline.
    # --------------------------------------------------------

    for feature in expected_features:

        if feature not in model_df.columns:

            model_df[
                feature
            ] = pd.NA

    # --------------------------------------------------------
    # KEEP ONLY EXPECTED FEATURES
    # --------------------------------------------------------

    model_df = model_df[
        expected_features
    ].copy()

    print(
        "\nPrepared input shape:"
    )

    print(
        model_df.shape
    )

    print(
        "\nPrepared input columns:"
    )

    print(
        model_df.columns.tolist()
    )

    return model_df


# ============================================================
# TRANSPORTATION PREDICTION
# ============================================================

def predict_transportation(
    order_df,
    transportation_model
):

    print(
        "\n" + "=" * 60
    )

    print(
        "TRANSPORTATION PREDICTION"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # PREPARE MODEL INPUT
    # --------------------------------------------------------

    model_input = (
        prepare_model_input(
            order_df,
            transportation_model,
            target_column="transport_mode"
        )
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    prediction = (
        transportation_model
        .predict(
            model_input
        )
    )

    if len(
        prediction
    ) == 0:

        raise ValueError(
            "Transportation model returned "
            "no prediction."
        )

    predicted_transport_mode = str(
        prediction[0]
    )

    print(
        "\nPredicted Transportation Mode:"
    )

    print(
        f">>> {predicted_transport_mode}"
    )

    # --------------------------------------------------------
    # PREDICT PROBABILITIES
    # --------------------------------------------------------

    if hasattr(
        transportation_model,
        "predict_proba"
    ):

        try:

            probabilities = (
                transportation_model
                .predict_proba(
                    model_input
                )
            )

            classes = getattr(
                transportation_model
                .named_steps
                .get("model"),
                "classes_",
                None
            )

            if classes is not None:

                print(
                    "\nTransportation Probabilities:"
                )

                for (
                    class_name,
                    probability
                ) in zip(
                    classes,
                    probabilities[0]
                ):

                    print(
                        f"{class_name:<12}: "
                        f"{probability * 100:.2f}%"
                    )

        except Exception as error:

            print(
                "\nProbability display skipped:"
            )

            print(
                error
            )

    return predicted_transport_mode


# ============================================================
# ETA PREDICTION
# ============================================================

def predict_delivery_eta(
    order_df,
    predicted_transport_mode,
    eta_model
):

    print(
        "\n" + "=" * 60
    )

    print(
        "ETA PREDICTION"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # PREPARE ETA INPUT
    # --------------------------------------------------------

    model_input = (
        prepare_model_input(
            order_df,
            eta_model,
            target_column=None,
            predicted_transport_mode=(
                predicted_transport_mode
            )
        )
    )

    # --------------------------------------------------------
    # PREDICT ETA
    # --------------------------------------------------------

    prediction = (
        eta_model
        .predict(
            model_input
        )
    )

    if len(
        prediction
    ) == 0:

        raise ValueError(
            "ETA model returned no prediction."
        )

    eta_hours = float(
        prediction[0]
    )

    # --------------------------------------------------------
    # VALIDATE ETA
    # --------------------------------------------------------

    if eta_hours < 0:

        raise ValueError(
            "ETA model returned a negative "
            "delivery time."
        )

    print(
        f"\nPredicted ETA: "
        f"{eta_hours:.2f} hours"
    )

    return eta_hours


# ============================================================
# CONVERT HOURS TO TIME
# ============================================================

def convert_hours_to_time(
    hours
):

    total_minutes = round(
        float(hours) * 60
    )

    hours_part = (
        total_minutes // 60
    )

    minutes_part = (
        total_minutes % 60
    )

    return (
        f"{hours_part} hours "
        f"{minutes_part} minutes"
    )


# ============================================================
# COMPLETE DELIVERY PREDICTION
# ============================================================

def run_delivery_prediction(
    order_data
):

    print(
        "\n" + "=" * 70
    )

    print(
        "DELIVERY PREDICTION"
    )

    print(
        "=" * 70
    )

    try:

        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        normalized_order = (
            normalize_order_data(
                order_data
            )
        )

        # ----------------------------------------------------
        # DATAFRAME
        # ----------------------------------------------------

        order_df = (
            create_order_dataframe(
                normalized_order
            )
        )

        # ----------------------------------------------------
        # LOAD MODELS
        # ----------------------------------------------------

        transportation_model = (
            load_transportation_model()
        )

        eta_model = (
            load_eta_model()
        )

        # ----------------------------------------------------
        # TRANSPORTATION
        # ----------------------------------------------------

        predicted_transport_mode = (
            predict_transportation(
                order_df,
                transportation_model
            )
        )

        # ----------------------------------------------------
        # ETA
        # ----------------------------------------------------

        eta_hours = (
            predict_delivery_eta(
                order_df,
                predicted_transport_mode,
                eta_model
            )
        )

        eta_hours = round(
            float(eta_hours),
            2
        )

        eta_minutes = round(
            eta_hours * 60
        )

        eta_duration = (
            convert_hours_to_time(
                eta_hours
            )
        )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        print(
            "\nPredicted Transport Mode : "
            f"{predicted_transport_mode}"
        )

        print(
            "Predicted ETA            : "
            f"{eta_hours:.2f} hours"
        )

        print(
            "ETA Duration             : "
            f"{eta_duration}"
        )

        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        return {

            "status": "success",

            "transport_mode":
                predicted_transport_mode,

            "predicted_eta_hours":
                eta_hours,

            "predicted_eta_minutes":
                eta_minutes,

            "eta_duration":
                eta_duration
        }

    except Exception as error:

        print(
            "\nDelivery prediction failed:"
        )

        print(
            str(error)
        )

        return {

            "status": "failed",

            "message": str(error)
        }


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\nSmartLogix AI - "
        "Delivery Prediction Agent"
    )

    print(
        "\nThis module provides reusable "
        "transportation and ETA prediction "
        "functions."
    )