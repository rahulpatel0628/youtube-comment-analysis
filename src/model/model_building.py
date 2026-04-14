import pandas as pd
import yaml
import logging
import joblib
from pathlib import Path
from lightgbm import LGBMClassifier


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_data(path: str) -> pd.DataFrame:
    try:
        logging.info("Loading training data")
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


def load_params(path: str):
    try:
        logging.info("Loading parameters")
        with open(path, "r") as f:
            params = yaml.safe_load(f)
        return params
    except Exception as e:
        logging.error(f"Error loading params: {e}")
        raise


def train_model(X, y, params):
    try:
        logging.info("Training LightGBM model")

        model = LGBMClassifier(**params)
        model.fit(X, y)

        logging.info("Model training completed")
        return model

    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise


def save_model(model, path: Path):
    try:
        logging.info("Saving model")

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)

        logging.info("Model saved successfully")

    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise


def main():
    try:
        data_path = "data/processed/train_tfidf.csv"
        params_path = "params.yaml"
        model_path = Path("models/lgb_model.pkl")

        data = load_data(data_path)

        params = load_params(params_path)['model_building']

        X = data.drop('sentiment', axis=1)
        y = data['sentiment']

        model = train_model(X, y, params)

        save_model(model, model_path)

        logging.info("Model training pipeline completed")

    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()