import numpy as np
import pandas as pd
import yaml
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_data(url: str) -> pd.DataFrame:
    try:
        logging.info("Loading dataset")
        df = pd.read_csv(url)
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logging.info("Starting preprocessing")

        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df.rename(columns={'clean_comment': 'comment', 'category': 'category'}, inplace=True)
        df = df[df['comment'].str.strip() != ""]
        df['category'] = df['category'].map({'-1': 0, '0': 1, '1': 2})

        logging.info("Preprocessing completed")
        return df
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise


def load_params(params_path: str) -> dict:
    try:
        logging.info("Loading parameters")
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        return params
    except Exception as e:
        logging.error(f"Error loading params: {e}")
        raise


def split_data(df: pd.DataFrame, test_size: float, random_state: int):
    try:
        logging.info("Splitting data")

        x_train, x_test, y_train, y_test = train_test_split(
            df['comment'],
            df['category'],
            test_size=test_size,
            random_state=random_state
        )

        train_df = pd.DataFrame({'comment': x_train, 'category': y_train})
        test_df = pd.DataFrame({'comment': x_test, 'category': y_test})

        logging.info("Data split completed")
        return train_df, test_df
    except Exception as e:
        logging.error(f"Error during data split: {e}")
        raise


def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, output_path: Path):
    try:
        logging.info("Saving datasets")

        output_path.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(output_path / "train.csv", index=False)
        test_df.to_csv(output_path / "test.csv", index=False)

        logging.info("Datasets saved successfully")
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise


def main():
    try:
        data_url = "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
        params_path = "params.yaml"
        output_path = Path(__file__).parent.parent.parent / "data" / "raw"

        df = load_data(data_url)
        df = preprocess_data(df)

        params = load_params(params_path)
        test_size = params['data_ingestion']['test_size']
        random_state = params['data_ingestion']['random_state']

        train_df, test_df = split_data(df, test_size, random_state)
        save_data(train_df, test_df, output_path)

    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()