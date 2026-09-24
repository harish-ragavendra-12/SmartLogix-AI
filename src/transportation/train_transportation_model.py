# ============================================================
# SMARTLOGIX AI - TRANSPORTATION MODEL TRAINING
# ============================================================

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config.config import PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR


# ============================================================
# 1. LOAD DATASET
# ============================================================

def load_feature_dataset(file_name):

    file_path = PROCESSED_DATA_DIR / file_name

    print("\n" + "=" * 60)
    print("LOADING TRANSPORTATION FEATURE DATASET")
    print("=" * 60)

    print(
        f"\nLoading dataset from:\n{file_path}"
    )

    df = pd.read_csv(file_path)

    print(
        f"\nDataset shape: {df.shape}"
    )

    return df


# ============================================================
# 2. BASIC DATA INSPECTION
# ============================================================

def inspect_dataset(df, target_column):

    print("\n" + "=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nTarget:")
    print(target_column)

    print("\nTarget distribution:")
    print(
        df[target_column]
        .value_counts()
    )

    print("\nMissing values:")

    missing = (
        df.isnull()
        .sum()
    )

    missing = missing[
        missing > 0
    ]

    if missing.empty:

        print(
            "No missing values."
        )

    else:

        print(
            missing
        )


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

def prepare_features_and_target(
    df,
    target_column
):

    print("\n" + "=" * 60)
    print("PREPARING FEATURES AND TARGET")
    print("=" * 60)

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    print(
        f"\nFeature shape: {X.shape}"
    )

    print(
        f"Target shape: {y.shape}"
    )

    print(
        "\nTarget classes:"
    )

    print(
        sorted(
            y.unique()
        )
    )

    return X, y


# ============================================================
# 4. IDENTIFY FEATURE TYPES
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

    print(
        "\nNumerical features:"
    )

    for column in numerical_columns:

        print(
            f"- {column}"
        )

    print(
        "\nCategorical features:"
    )

    for column in categorical_columns:

        print(
            f"- {column}"
        )

    print(
        f"\nTotal numerical features: "
        f"{len(numerical_columns)}"
    )

    print(
        f"Total categorical features: "
        f"{len(categorical_columns)}"
    )

    return (
        numerical_columns,
        categorical_columns
    )


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

def split_dataset(
    X,
    y
):

    print("\n" + "=" * 60)
    print("TRAIN / TEST SPLIT")
    print("=" * 60)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
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

    print(
        "\nTraining target distribution:"
    )

    print(
        y_train.value_counts()
    )

    print(
        "\nTesting target distribution:"
    )

    print(
        y_test.value_counts()
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# 6. CREATE PREPROCESSOR
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
            ),
            (
                "scaler",
                StandardScaler()
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
# 7. CREATE MODELS
# ============================================================

def create_models():

    print("\n" + "=" * 60)
    print("CREATING CLASSIFICATION MODELS")
    print("=" * 60)

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=15,
                min_samples_split=10,
                class_weight="balanced",
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42
            )
    }

    for model_name in models:

        print(
            f"- {model_name}"
        )

    return models


# ============================================================
# 8. BUILD MODEL PIPELINE
# ============================================================

def build_pipeline(
    preprocessor,
    model
):

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
# 9. CALCULATE ROC-AUC
# ============================================================

def calculate_roc_auc(
    pipeline,
    X_test,
    y_test
):

    try:

        y_probability = (
            pipeline.predict_proba(
                X_test
            )
        )

        classes = (
            pipeline
            .named_steps["model"]
            .classes_
        )

        # Multiclass ROC-AUC
        score = roc_auc_score(
            y_test,
            y_probability,
            multi_class="ovr",
            average="weighted",
            labels=classes
        )

        return score

    except Exception as error:

        print(
            f"ROC-AUC could not be calculated: "
            f"{error}"
        )

        return None


# ============================================================
# 10. EVALUATE MODEL
# ============================================================

def evaluate_model(
    model_name,
    pipeline,
    X_test,
    y_test
):

    print("\n" + "=" * 60)
    print(
        f"EVALUATING: {model_name}"
    )
    print("=" * 60)

    y_pred = (
        pipeline.predict(
            X_test
        )
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    roc_auc = calculate_roc_auc(
        pipeline,
        X_test,
        y_test
    )

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    if roc_auc is not None:

        print(
            f"ROC-AUC  : {roc_auc:.4f}"
        )

    else:

        print(
            "ROC-AUC  : Not available"
        )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    metrics = {

        "model": model_name,

        "accuracy": float(
            accuracy
        ),

        "precision_weighted": float(
            precision
        ),

        "recall_weighted": float(
            recall
        ),

        "f1_weighted": float(
            f1
        ),

        "roc_auc_weighted_ovr":
            (
                float(roc_auc)
                if roc_auc is not None
                else None
            )
    }

    return (
        metrics,
        y_pred
    )


# ============================================================
# 11. TRAIN AND COMPARE MODELS
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
    print("MODEL TRAINING AND COMPARISON")
    print("=" * 60)

    results = []

    trained_pipelines = {}

    predictions = {}

    for model_name, model in models.items():

        print("\n" + "-" * 60)

        print(
            f"Training: {model_name}"
        )

        print("-" * 60)

        pipeline = build_pipeline(
            preprocessor,
            model
        )

        pipeline.fit(
            X_train,
            y_train
        )

        print(
            "Training completed."
        )

        metrics, y_pred = (
            evaluate_model(
                model_name,
                pipeline,
                X_test,
                y_test
            )
        )

        results.append(
            metrics
        )

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
# 12. DISPLAY MODEL COMPARISON
# ============================================================

def display_model_comparison(
    results_df
):

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    display_columns = [
        "model",
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "roc_auc_weighted_ovr"
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# 13. SELECT MODEL
# ============================================================

def select_best_model(
    results_df,
    trained_pipelines
):

    print("\n" + "=" * 60)
    print("MODEL SELECTION")
    print("=" * 60)

    # Use weighted F1 as the primary selection
    # metric because the target classes are imbalanced.

    best_index = (
        results_df[
            "f1_weighted"
        ]
        .idxmax()
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
        "Weighted F1-score"
    )

    return (
        best_model_name,
        best_pipeline
    )


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

def create_confusion_matrix(
    model_name,
    pipeline,
    X_test,
    y_test
):

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    y_pred = (
        pipeline.predict(
            X_test
        )
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        cm
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    figure = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=sorted(
            y_test.unique()
        )
    )

    figure.plot(
        xticks_rotation=45
    )

    plt.title(
        f"Transportation Mode - "
        f"Confusion Matrix\n"
        f"{model_name}"
    )

    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / "transportation_confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nConfusion matrix saved to:"
    )

    print(
        output_path
    )


# ============================================================
# 15. SAVE MODEL
# ============================================================

def save_model(
    model_name,
    pipeline
):

    TRANSPORTATION_MODEL_DIR = (
        MODELS_DIR
        / "transportation"
    )

    TRANSPORTATION_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        TRANSPORTATION_MODEL_DIR
        / "transportation_classifier.joblib"
    )

    joblib.dump(
        pipeline,
        model_path
    )

    print(
        "\nModel saved to:"
    )

    print(
        model_path
    )

    return model_path


# ============================================================
# 16. SAVE MODEL RESULTS
# ============================================================

def save_results(
    results_df,
    best_model_name
):

    TRANSPORTATION_MODEL_DIR = (
        MODELS_DIR
        / "transportation"
    )

    TRANSPORTATION_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_path = (
        TRANSPORTATION_MODEL_DIR
        / "transportation_model_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    metadata = {

        "selected_model":
            best_model_name,

        "selection_metric":
            "weighted_f1"
    }

    metadata_path = (
        TRANSPORTATION_MODEL_DIR
        / "transportation_model_metadata.json"
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

    print(
        results_path
    )

    print(
        "\nMetadata saved to:"
    )

    print(
        metadata_path
    )


# ============================================================
# 17. MAIN
# ============================================================

def main():

    print(
        "\n" + "=" * 60
    )

    print(
        "SMARTLOGIX AI - "
        "TRANSPORTATION MODEL TRAINING"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    DATASET_FILE = (
        "transportation_features.csv"
    )

    TARGET_COLUMN = (
        "transport_mode"
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_feature_dataset(
        DATASET_FILE
    )

    # --------------------------------------------------------
    # Inspect
    # --------------------------------------------------------

    inspect_dataset(
        df,
        TARGET_COLUMN
    )

    # --------------------------------------------------------
    # Prepare X and y
    # --------------------------------------------------------

    X, y = (
        prepare_features_and_target(
            df,
            TARGET_COLUMN
        )
    )

    # --------------------------------------------------------
    # Identify feature types
    # --------------------------------------------------------

    (
        numerical_columns,
        categorical_columns
    ) = identify_feature_types(
        X
    )

    # --------------------------------------------------------
    # Train/test split
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
    # Preprocessor
    # --------------------------------------------------------

    preprocessor = (
        create_preprocessor(
            numerical_columns,
            categorical_columns
        )
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = create_models()

    # --------------------------------------------------------
    # Train + compare
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
    # Comparison
    # --------------------------------------------------------

    display_model_comparison(
        results_df
    )

    # --------------------------------------------------------
    # Select model
    # --------------------------------------------------------

    (
        best_model_name,
        best_pipeline
    ) = select_best_model(
        results_df,
        trained_pipelines
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    create_confusion_matrix(
        best_model_name,
        best_pipeline,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    save_model(
        best_model_name,
        best_pipeline
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    save_results(
        results_df,
        best_model_name
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "TRANSPORTATION MODEL TRAINING "
        "COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()