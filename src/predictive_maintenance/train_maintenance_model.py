import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.tree import DecisionTreeClassifier

from src.config.config import (
    PROCESSED_DATA_DIR,
    MAINTENANCE_MODEL_DIR,
    FIGURES_DIR,
    RANDOM_STATE
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "maintenance_features.csv"
)

MODEL_FILE = (
    MAINTENANCE_MODEL_DIR
    / "maintenance_classifier.joblib"
)

RESULTS_FILE = (
    MAINTENANCE_MODEL_DIR
    / "maintenance_model_results.csv"
)

METADATA_FILE = (
    MAINTENANCE_MODEL_DIR
    / "maintenance_model_metadata.json"
)

CONFUSION_MATRIX_FILE = (
    FIGURES_DIR
    / "maintenance_confusion_matrix.png"
)

TARGET_COLUMN = "failure_reported"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n" + "=" * 70)
    print("LOADING MAINTENANCE FEATURES")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Dataset shape : {df.shape}"
    )

    print("\nColumns:")

    print(
        df.columns.tolist()
    )

    return df


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

def prepare_features_and_target(df):

    print("\n" + "=" * 70)
    print("PREPARING FEATURES AND TARGET")
    print("=" * 70)

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN]

    print(
        f"Feature shape : {X.shape}"
    )

    print(
        f"Target shape  : {y.shape}"
    )

    print("\nTarget distribution:")

    print(
        y.value_counts()
        .sort_index()
    )

    return X, y


# ============================================================
# IDENTIFY FEATURE TYPES
# ============================================================

def identify_feature_types(X):

    print("\n" + "=" * 70)
    print("IDENTIFYING FEATURE TYPES")
    print("=" * 70)

    numerical_features = (
        X.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    categorical_features = (
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

    print(
        f"Numerical features   : "
        f"{len(numerical_features)}"
    )

    print(
        f"Categorical features : "
        f"{len(categorical_features)}"
    )

    print("\nNumerical features:")

    print(
        numerical_features
    )

    print("\nCategorical features:")

    print(
        categorical_features
    )

    return (
        numerical_features,
        categorical_features
    )


# ============================================================
# CREATE PREPROCESSOR
# ============================================================

def create_preprocessor(
    numerical_features,
    categorical_features
):

    print("\n" + "=" * 70)
    print("CREATING PREPROCESSING PIPELINE")
    print("=" * 70)

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
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    print(
        "Numerical preprocessing : "
        "Median Imputation + StandardScaler"
    )

    print(
        "Categorical preprocessing : "
        "Most Frequent Imputation + OneHotEncoder"
    )

    return preprocessor


# ============================================================
# CREATE MODELS
# ============================================================

def create_models():

    print("\n" + "=" * 70)
    print("CREATING CLASSIFICATION MODELS")
    print("=" * 70)

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=10,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),

        "Gradient Boosting":
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=RANDOM_STATE
            )
    }

    print(
        "Models created:"
    )

    for model_name in models:

        print(
            f"- {model_name}"
        )

    return models


# ============================================================
# TRAIN AND EVALUATE MODEL
# ============================================================

def train_and_evaluate_model(
    model_name,
    model,
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n" + "-" * 70)
    print(
        f"TRAINING: {model_name}"
    )
    print("-" * 70)

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

    pipeline.fit(
        X_train,
        y_train
    )

    y_pred = pipeline.predict(
        X_test
    )

    # --------------------------------------------------------
    # Probability prediction
    # --------------------------------------------------------

    y_probability = (
        pipeline.predict_proba(X_test)[:, 1]
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "No Failure",
                "Failure Reported"
            ],
            zero_division=0
        )
    )

    return {

        "model_name": model_name,

        "pipeline": pipeline,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "roc_auc": roc_auc,

        "y_test": y_test,

        "y_pred": y_pred

    }


# ============================================================
# COMPARE MODELS
# ============================================================

def compare_models(results):

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    comparison_records = []

    for result in results:

        comparison_records.append({

            "model": result["model_name"],

            "accuracy": result["accuracy"],

            "precision": result["precision"],

            "recall": result["recall"],

            "f1_score": result["f1_score"],

            "roc_auc": result["roc_auc"]

        })

    comparison_df = pd.DataFrame(
        comparison_records
    )

    comparison_df = comparison_df.sort_values(
        by="f1_score",
        ascending=False
    ).reset_index(drop=True)

    print(
        comparison_df.to_string(
            index=False
        )
    )

    return comparison_df


# ============================================================
# SELECT BEST MODEL
# ============================================================

def select_best_model(
    results,
    comparison_df
):

    print("\n" + "=" * 70)
    print("SELECTING BEST MODEL")
    print("=" * 70)

    best_model_name = (
        comparison_df.iloc[0]["model"]
    )

    best_result = next(
        result
        for result in results
        if result["model_name"]
        == best_model_name
    )

    print(
        f"Selected model : "
        f"{best_model_name}"
    )

    print(
        f"Weighted F1     : "
        f"{best_result['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC         : "
        f"{best_result['roc_auc']:.4f}"
    )

    return best_result


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    best_result
):

    print("\n" + "=" * 70)
    print("CREATING CONFUSION MATRIX")
    print("=" * 70)

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cm = confusion_matrix(
        best_result["y_test"],
        best_result["y_pred"]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "No Failure",
            "Failure Reported"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    display.plot(
        ax=ax
    )

    ax.set_title(
        "Predictive Maintenance - "
        "Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Confusion matrix saved to:\n"
        f"{CONFUSION_MATRIX_FILE}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    best_result
):

    print("\n" + "=" * 70)
    print("SAVING BEST MODEL")
    print("=" * 70)

    MAINTENANCE_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        best_result["pipeline"],
        MODEL_FILE
    )

    print(
        f"Model saved to:\n"
        f"{MODEL_FILE}"
    )


# ============================================================
# SAVE MODEL RESULTS
# ============================================================

def save_model_results(
    comparison_df
):

    print("\n" + "=" * 70)
    print("SAVING MODEL RESULTS")
    print("=" * 70)

    MAINTENANCE_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    print(
        f"Results saved to:\n"
        f"{RESULTS_FILE}"
    )


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(
    best_result,
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n" + "=" * 70)
    print("SAVING MODEL METADATA")
    print("=" * 70)

    metadata = {

        "project": "SmartLogix AI",

        "module": "Predictive Maintenance",

        "target_column": TARGET_COLUMN,

        "selected_model":
            best_result["model_name"],

        "metrics": {

            "accuracy":
                round(
                    best_result["accuracy"],
                    4
                ),

            "precision":
                round(
                    best_result["precision"],
                    4
                ),

            "recall":
                round(
                    best_result["recall"],
                    4
                ),

            "f1_score":
                round(
                    best_result["f1_score"],
                    4
                ),

            "roc_auc":
                round(
                    best_result["roc_auc"],
                    4
                )
        },

        "training_samples":
            len(X_train),

        "testing_samples":
            len(X_test),

        "failure_distribution": {

            "no_failure":
                int(
                    (y_train == 0).sum()
                    +
                    (y_test == 0).sum()
                ),

            "failure_reported":
                int(
                    (y_train == 1).sum()
                    +
                    (y_test == 1).sum()
                )
        },

        "random_state":
            RANDOM_STATE

    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print(
        f"Metadata saved to:\n"
        f"{METADATA_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("SMARTLOGIX AI - PREDICTIVE MAINTENANCE")
    print("MODEL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # 2. Prepare features and target
    # --------------------------------------------------------

    X, y = prepare_features_and_target(
        df
    )

    # --------------------------------------------------------
    # 3. Identify feature types
    # --------------------------------------------------------

    (
        numerical_features,
        categorical_features
    ) = identify_feature_types(
        X
    )

    # --------------------------------------------------------
    # 4. Train/test split
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CREATING TRAIN / TEST SPLIT")
    print("=" * 70)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    print(
        f"Training samples : "
        f"{len(X_train)}"
    )

    print(
        f"Testing samples  : "
        f"{len(X_test)}"
    )

    print(
        "\nTraining target distribution:"
    )

    print(
        y_train.value_counts()
        .sort_index()
    )

    print(
        "\nTesting target distribution:"
    )

    print(
        y_test.value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # 5. Create preprocessor
    # --------------------------------------------------------

    preprocessor = create_preprocessor(
        numerical_features,
        categorical_features
    )

    # --------------------------------------------------------
    # 6. Create models
    # --------------------------------------------------------

    models = create_models()

    # --------------------------------------------------------
    # 7. Train and evaluate
    # --------------------------------------------------------

    results = []

    for model_name, model in models.items():

        result = train_and_evaluate_model(
            model_name,
            model,
            preprocessor,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # 8. Compare models
    # --------------------------------------------------------

    comparison_df = compare_models(
        results
    )

    # --------------------------------------------------------
    # 9. Select best model
    # --------------------------------------------------------

    best_result = select_best_model(
        results,
        comparison_df
    )

    # --------------------------------------------------------
    # 10. Save confusion matrix
    # --------------------------------------------------------

    save_confusion_matrix(
        best_result
    )

    # --------------------------------------------------------
    # 11. Save model
    # --------------------------------------------------------

    save_model(
        best_result
    )

    # --------------------------------------------------------
    # 12. Save results
    # --------------------------------------------------------

    save_model_results(
        comparison_df
    )

    # --------------------------------------------------------
    # 13. Save metadata
    # --------------------------------------------------------

    save_metadata(
        best_result,
        X_train,
        X_test,
        y_train,
        y_test
    )

    # --------------------------------------------------------
    # 14. Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREDICTIVE MAINTENANCE MODEL TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Best model : "
        f"{best_result['model_name']}"
    )

    print(
        f"F1 Score   : "
        f"{best_result['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC    : "
        f"{best_result['roc_auc']:.4f}"
    )


if __name__ == "__main__":
    main()