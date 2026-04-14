import numpy as np
import pandas as pd
import logging
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.data.pandas_dataset import PandasDataset
import os
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import json
import joblib
from dotenv import load_dotenv
import dagshub

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_data(path: Path) -> pd.DataFrame:
    try:
        logging.info("Loading data")
        df = pd.read_csv(path)
        df = df.dropna().reset_index(drop=True)
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


def load_model(path: Path):
    try:
        logging.info("Loading model")
        return joblib.load(path)
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise


def setup_mlflow():
    try:
        dagshub.init(repo_owner='rahulpatel16092005', repo_name='youtube-comment-analysis', mlflow=True)
        mlflow.set_tracking_uri("https://dagshub.com/rahulpatel16092005/youtube-comment-analysis.mlflow")
        mlflow.set_experiment("dvc-pipeline-runs")

        logging.info("MLflow setup completed")
    except Exception as e:
        logging.error(f"Error setting up MLflow: {e}")
        raise


def create_features(df: pd.DataFrame):
    try:
        wc = df['comment'].apply(lambda x: len(x.split())).values.reshape(-1, 1)
        cc = df['comment'].apply(lambda x: len(x)).values.reshape(-1, 1)
        awl = df['comment'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0
        ).values.reshape(-1, 1)
        return wc, cc, awl
    except Exception as e:
        logging.error(f"Error creating features: {e}")
        raise


def prepare_data(df: pd.DataFrame, vectorizer):
    try:
        logging.info("Preparing test data")

        X_tfidf = vectorizer.transform(df['comment'])
        wc, cc, awl = create_features(df)

        X = np.hstack([X_tfidf.toarray(), wc, cc, awl])
        y = df['category'].values

        return X, y, X_tfidf
    except Exception as e:
        logging.error(f"Error preparing data: {e}")
        raise


def evaluate_model(model, X, y):
    try:
        logging.info("Evaluating model")

        y_pred = model.predict(X)

        report = classification_report(y, y_pred, output_dict=True)
        cm = confusion_matrix(y, y_pred)

        return report, cm, y_pred
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        raise


def log_confusion_matrix(cm: np.ndarray, path: Path):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.savefig(path)
        plt.clf()

        mlflow.log_artifact(path)

        logging.info("Confusion matrix logged")
    except Exception as e:
        logging.error(f"Error logging confusion matrix: {e}")
        raise


def log_metrics(report: dict):
    try:
        accuracy = report.get("accuracy", 0)
        mlflow.log_metric("accuracy", accuracy)

        for label, metrics in report.items():
            if isinstance(metrics, dict) and "precision" in metrics:
                mlflow.log_metrics({
                    f"{label}_precision": metrics["precision"],
                    f"{label}_recall": metrics["recall"],
                    f"{label}_f1": metrics["f1-score"]
                })

        with open("metrics.json", "w") as f:
            json.dump({"accuracy": accuracy}, f)

        logging.info("Metrics logged")
    except Exception as e:
        logging.error(f"Error logging metrics: {e}")
        raise


def log_models(model, vectorizer, X_sample, X_tfidf_sample):
    try:
        sig_model = infer_signature(X_sample, model.predict(X_sample))
        mlflow.sklearn.log_model(model, "lgbm_model", signature=sig_model)

        sig_vec = infer_signature(X_tfidf_sample, X_tfidf_sample)
        mlflow.sklearn.log_model(vectorizer, "tfidf_vectorizer", signature=sig_vec)

        logging.info("Models logged")
    except Exception as e:
        logging.error(f"Error logging models: {e}")
        raise


def save_model_info(run_id: str, path: Path):
    try:
        info = {
            "run_id": run_id,
            "model_path": "lgb_model",
            "vectorizer_path": "tfidf_vectorizer"
        }

        with open(path, "w") as f:
            json.dump(info, f, indent=4)

        logging.info("Model info saved")
    except Exception as e:
        logging.error(f"Error saving model info: {e}")
        raise


def main():
    try:
        setup_mlflow()

        base_path = Path(__file__).parent.parent

        test_path = base_path.parent / "data" / "interim" / "preprocessed_test.csv"
        model_path = base_path.parent / "models" / "lgb_model.pkl"
        vectorizer_path = base_path.parent / "models" / "tfidf_vectorizer.pkl"

        df = load_data(test_path)
        model = load_model(model_path)
        vectorizer = load_model(vectorizer_path)

        X, y, X_tfidf = prepare_data(df, vectorizer)

        with mlflow.start_run() as run:
            report, cm, _ = evaluate_model(model, X, y)

            log_metrics(report)

            cm_path = base_path / "visualizations" / "confusion_matrix.png"
            log_confusion_matrix(cm, cm_path)

            log_models(model, vectorizer, X[:10], X_tfidf[:10])
            mlflow.log_artifact(vectorizer_path, artifact_path="tfidf_vectorizer.pkl")
            mlflow.log_artifact(model_path, artifact_path="lgb_model.pkl")

            save_model_info(run.info.run_id, "experiment_info.json")

            mlflow.set_tag("model", "LightGBM")
            mlflow.set_tag("vectorizer", "TF-IDF")

        logging.info("Evaluation pipeline completed")

    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()