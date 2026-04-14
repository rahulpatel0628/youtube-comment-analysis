import numpy as np
import pandas as pd
import yaml
import logging
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_data(path: str) -> pd.DataFrame:
    try:
        logging.info("Loading processed data")
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


def load_params(params_path: str):
    try:
        logging.info("Loading parameters")
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        return params
    except Exception as e:
        logging.error(f"Error loading params: {e}")
        raise


def apply_tfidf(df: pd.DataFrame, max_features: int, n_grams: tuple):
    try:
        logging.info("Applying TF-IDF")

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=n_grams
        )

        X_train_tfidf = vectorizer.fit_transform(df['comment'])

        tfidf_df = pd.DataFrame(
            X_train_tfidf.toarray(),
            columns=vectorizer.get_feature_names_out()
        )

        tfidf_df['word_count'] = df['word_count'].values
        tfidf_df['char_count'] = df['char_count'].values
        tfidf_df['avg_word_length'] = df['avg_word_length'].values
        tfidf_df['sentiment'] = df['category'].values

        logging.info(f"TF-IDF completed. Shape: {tfidf_df.shape}")

        return tfidf_df, vectorizer

    except Exception as e:
        logging.error(f"Error in TF-IDF: {e}")
        raise


def save_data(df: pd.DataFrame, output_path: Path):
    try:
        logging.info("Saving TF-IDF data")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        logging.info("Data saved successfully")
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise


def save_vectorizer(vectorizer, path: Path):
    try:
        logging.info("Saving vectorizer")

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(vectorizer, path)

        logging.info("Vectorizer saved successfully")
    except Exception as e:
        logging.error(f"Error saving vectorizer: {e}")
        raise


def main():
    try:
        data_path = "data/interim/preprocessed_train.csv"
        params_path = "params.yaml"

        output_data_path = Path("data/processed/train_tfidf.csv")
        vectorizer_path = Path("models/tfidf_vectorizer.pkl")

        df = load_data(data_path)
        df.dropna(inplace=True)
        params = load_params(params_path)
        max_features = params['feature_engineering']['max_features']
        n_grams = tuple(params['feature_engineering']['n_grams'])

        tfidf_df, vectorizer = apply_tfidf(df, max_features, n_grams)

        save_data(tfidf_df, output_data_path)
        save_vectorizer(vectorizer, vectorizer_path)

        logging.info("Feature engineering pipeline completed")

    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()