# ============================================================
# SMARTLOGIX AI - ETA MODEL TRAINING
# ============================================================

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from src.config.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    FIGURES_DIR
)


# ============================================================
# LOAD ETA FEATURE DATASET
# ============================================================

def load_feature_dataset(file_name):

    print("\n" + "=" * 60)
    print("LOADING ETA FEATURE DATASET")
    print("=" * 60)

    file_path = PROCESSED_DATA_DIR / file_name

    print(f"\nLoading dataset from:")
    print(file_path)

    df = pd.read_csv(file_path)

    print(f"\nDataset shape: {df.shape}")

    return df


# ============================================================
# INSPECT DATASET
# ============================================================

def inspect_dataset(df, target_column):

    print("\n" + "=" * 60)
    print("ETA DATASET INSPECTION")
    print("=" * 60)

    print("\nShape:")
    print(df.shape)

    print("\nTarget:")
    print(target_column)

    print("\nMissing values:")

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values.")
    else:
        print(missing)

    print("\nTarget statistics:")
    print(df[target_column].describe())


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

def prepare_features_and_target(df, target_column):

    print("\n" + "=" * 60)
    print("PREPARING FEATURES AND TARGET")
    print("=" * 60)

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    print(f"\nFeature shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    return X, y


# ============================================================
# IDENTIFY FEATURE TYPES
# ============================================================

def identify_feature_types(X):

    numerical_columns = (
        X.select_dtypes(
            include=[
                "int64",
                "float64",
                "int32",
                "float32",
                "bool"
            ]
        )
        .columns
        .tolist()
    )

    categorical_columns = (
        X.select_dtypes(
            include=[
                "object",
                "string",
                "category"
            ]
        )
        .columns
        .tolist()
    )

    print("\n" + "=" * 60)
    print("FEATURE TYPES")
    print("=" * 60)

    print("\nNumerical features:")

    for column in numerical_columns:
        print(f"- {column}")

    print("\nCategorical features:")

    for column in categorical_columns:
        print(f"- {column}")

    print(
        f"\nTotal numerical features: "
        f"{len(numerical_columns)}"
    )

    print(
        f"Total categorical features: "
        f"{len(categorical_columns)}"
    )

    return numerical_columns, categorical_columns


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_dataset(X, y):

    print("\n" + "=" * 60)
    print("TRAIN / TEST SPLIT")
    print("=" * 60)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )
    )

    print(
        f"\nTraining samples: "
        f"{len(X_train)}"
    )

    print(
        f"Testing samples: "
        f"{len(X_test)}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# CREATE PREPROCESSOR
# ============================================================

def create_preprocessor(
    numerical_columns,
    categorical_columns
):

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )

    return preprocessor


# ============================================================
# CREATE REGRESSION MODELS
# ============================================================

def create_models():

    print("\n" + "=" * 60)
    print("CREATING ETA REGRESSION MODELS")
    print("=" * 60)

    models = {

        "Decision Tree": DecisionTreeRegressor(
            max_depth=15,
            min_samples_split=10,
            random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            n_jobs=-1,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=10,
            random_state=42
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            n_jobs=-1,
            random_state=42
        )
    }

    for model_name in models:
        print(f"- {model_name}")

    return models


# ============================================================
# BUILD PIPELINE
# ============================================================

def build_pipeline(preprocessor, model):

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    return pipeline


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model_name,
    pipeline,
    X_test,
    y_test
):

    print("\n" + "=" * 60)
    print(f"EVALUATING: {model_name}")
    print("=" * 60)

    y_pred = pipeline.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = mean_squared_error(
        y_test,
        y_pred
    ) ** 0.5

    r2 = r2_score(
        y_test,
        y_pred
    )

    print(
        f"\nMAE  : {mae:.4f} hours"
    )

    print(
        f"RMSE : {rmse:.4f} hours"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    metrics = {
        "model": model_name,
        "mae_hours": float(mae),
        "rmse_hours": float(rmse),
        "r2_score": float(r2)
    }

    return metrics, y_pred


# ============================================================
# TRAIN AND COMPARE MODELS
# ============================================================

def train_and_compare_models(
    preprocessor,
    models,
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n" + "=" * 60)
    print("ETA MODEL TRAINING AND COMPARISON")
    print("=" * 60)

    results = []

    trained_pipelines = {}

    predictions = {}

    for model_name, model in models.items():

        print("\n" + "-" * 60)
        print(f"Training: {model_name}")
        print("-" * 60)

        pipeline = build_pipeline(
            preprocessor,
            model
        )

        pipeline.fit(
            X_train,
            y_train
        )

        print("Training completed.")

        metrics, y_pred = evaluate_model(
            model_name,
            pipeline,
            X_test,
            y_test
        )

        results.append(metrics)

        trained_pipelines[
            model_name
        ] = pipeline

        predictions[
            model_name
        ] = y_pred

    results_df = pd.DataFrame(
        results
    )

    return (
        results_df,
        trained_pipelines,
        predictions
    )


# ============================================================
# DISPLAY MODEL COMPARISON
# ============================================================

def display_model_comparison(
    results_df
):

    print("\n" + "=" * 60)
    print("ETA MODEL COMPARISON")
    print("=" * 60)

    display_columns = [
        "model",
        "mae_hours",
        "rmse_hours",
        "r2_score"
    ]

    comparison_df = (
        results_df[
            display_columns
        ]
        .sort_values(
            by="rmse_hours"
        )
    )

    print(
        comparison_df.to_string(
            index=False
        )
    )


# ============================================================
# SELECT BEST MODEL
# ============================================================

def select_best_model(
    results_df,
    trained_pipelines
):

    print("\n" + "=" * 60)
    print("ETA MODEL SELECTION")
    print("=" * 60)

    best_index = (
        results_df[
            "rmse_hours"
        ]
        .idxmin()
    )

    best_model_name = (
        results_df
        .loc[
            best_index,
            "model"
        ]
    )

    best_pipeline = (
        trained_pipelines[
            best_model_name
        ]
    )

    print(
        f"\nSelected model: "
        f"{best_model_name}"
    )

    print(
        "Selection metric: "
        "RMSE"
    )

    return (
        best_model_name,
        best_pipeline
    )


# ============================================================
# CREATE ACTUAL VS PREDICTED PLOT
# ============================================================

def create_actual_vs_predicted_plot(
    model_name,
    pipeline,
    X_test,
    y_test
):

    print("\n" + "=" * 60)
    print("CREATING ACTUAL VS PREDICTED PLOT")
    print("=" * 60)

    y_pred = pipeline.predict(
        X_test
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.5
    )

    minimum = min(
        y_test.min(),
        y_pred.min()
    )

    maximum = max(
        y_test.max(),
        y_pred.max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.xlabel(
        "Actual Delivery Hours"
    )

    plt.ylabel(
        "Predicted Delivery Hours"
    )

    plt.title(
        "Actual vs Predicted ETA\n"
        f"{model_name}"
    )

    plt.tight_layout()

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        FIGURES_DIR
        / "eta_actual_vs_predicted.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nActual vs predicted plot saved to:"
    )

    print(output_path)


# ============================================================
# CREATE RESIDUAL PLOT
# ============================================================

def create_residual_plot(
    model_name,
    pipeline,
    X_test,
    y_test
):

    print("\n" + "=" * 60)
    print("CREATING RESIDUAL PLOT")
    print("=" * 60)

    y_pred = pipeline.predict(
        X_test
    )

    residuals = (
        y_test.values
        - y_pred
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        y_pred,
        residuals,
        alpha=0.5
    )

    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.xlabel(
        "Predicted Delivery Hours"
    )

    plt.ylabel(
        "Residual"
    )

    plt.title(
        "ETA Residual Plot\n"
        f"{model_name}"
    )

    plt.tight_layout()

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        FIGURES_DIR
        / "eta_residual_plot.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nResidual plot saved to:"
    )

    print(output_path)


# ============================================================
# SAVE BEST MODEL
# ============================================================

def save_model(
    best_model_name,
    best_pipeline
):

    print("\n" + "=" * 60)
    print("SAVING ETA MODEL")
    print("=" * 60)

    ETA_MODEL_DIR = (
        MODELS_DIR
        / "eta"
    )

    ETA_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        ETA_MODEL_DIR
        / "eta_regressor.joblib"
    )

    joblib.dump(
        best_pipeline,
        model_path
    )

    print(
        "\nETA model saved to:"
    )

    print(model_path)

    return model_path


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results_df,
    best_model_name
):

    print("\n" + "=" * 60)
    print("SAVING ETA MODEL RESULTS")
    print("=" * 60)

    ETA_MODEL_DIR = (
        MODELS_DIR
        / "eta"
    )

    ETA_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_path = (
        ETA_MODEL_DIR
        / "eta_model_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    metadata = {
        "selected_model": best_model_name,
        "selection_metric": "rmse",
        "target": "actual_delivery_hours"
    }

    metadata_path = (
        ETA_MODEL_DIR
        / "eta_model_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print(
        "\nResults saved to:"
    )

    print(results_path)

    print(
        "\nMetadata saved to:"
    )

    print(metadata_path)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("SMARTLOGIX AI - ETA MODEL TRAINING")
    print("=" * 60)

    DATASET_FILE = (
        "eta_features.csv"
    )

    TARGET_COLUMN = (
        "actual_delivery_hours"
    )

    # --------------------------------------------------------
    # Step 1: Load dataset
    # --------------------------------------------------------

    df = load_feature_dataset(
        DATASET_FILE
    )

    # --------------------------------------------------------
    # Step 2: Inspect dataset
    # --------------------------------------------------------

    inspect_dataset(
        df,
        TARGET_COLUMN
    )

    # --------------------------------------------------------
    # Step 3: Prepare X and y
    # --------------------------------------------------------

    X, y = prepare_features_and_target(
        df,
        TARGET_COLUMN
    )

    # --------------------------------------------------------
    # Step 4: Identify feature types
    # --------------------------------------------------------

    (
        numerical_columns,
        categorical_columns
    ) = identify_feature_types(X)

    # --------------------------------------------------------
    # Step 5: Train/test split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_dataset(
        X,
        y
    )

    # --------------------------------------------------------
    # Step 6: Create preprocessor
    # --------------------------------------------------------

    preprocessor = create_preprocessor(
        numerical_columns,
        categorical_columns
    )

    # --------------------------------------------------------
    # Step 7: Create models
    # --------------------------------------------------------

    models = create_models()

    # --------------------------------------------------------
    # Step 8: Train and compare
    # --------------------------------------------------------

    (
        results_df,
        trained_pipelines,
        predictions
    ) = train_and_compare_models(
        preprocessor,
        models,
        X_train,
        X_test,
        y_train,
        y_test
    )

    # --------------------------------------------------------
    # Step 9: Display comparison
    # --------------------------------------------------------

    display_model_comparison(
        results_df
    )

    # --------------------------------------------------------
    # Step 10: Select best model
    # --------------------------------------------------------

    (
        best_model_name,
        best_pipeline
    ) = select_best_model(
        results_df,
        trained_pipelines
    )

    # --------------------------------------------------------
    # Step 11: Actual vs predicted
    # --------------------------------------------------------

    create_actual_vs_predicted_plot(
        best_model_name,
        best_pipeline,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Step 12: Residual plot
    # --------------------------------------------------------

    create_residual_plot(
        best_model_name,
        best_pipeline,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Step 13: Save model
    # --------------------------------------------------------

    save_model(
        best_model_name,
        best_pipeline
    )

    # --------------------------------------------------------
    # Step 14: Save results
    # --------------------------------------------------------

    save_results(
        results_df,
        best_model_name
    )

    print("\n" + "=" * 60)
    print(
        "ETA MODEL TRAINING "
        "COMPLETED SUCCESSFULLY"
    )
    print("=" * 60)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()