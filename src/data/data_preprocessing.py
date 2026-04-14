import numpy as np
import pandas as pd
import nltk
import logging
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_data(train_path: str, test_path: str):
    try:
        logging.info("Loading train and test data")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        return train_df, test_df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


def download_nltk_resources():
    try:
        logging.info("Downloading NLTK resources")
        nltk.download('stopwords')
        nltk.download('wordnet')
    except Exception as e:
        logging.error(f"Error downloading NLTK data: {e}")
        raise


def get_filtered_stopwords():
    try:
        stop_words = set(stopwords.words('english'))
        important_stop_words = {
            'not', 'but', 'whenever', 'however',
            'although', 'though', 'yet', 'nevertheless'
        }
        return stop_words - important_stop_words
    except Exception as e:
        logging.error(f"Error preparing stopwords: {e}")
        raise


def clean_text_column(df: pd.DataFrame, stop_words, lemmatizer):
    try:
        df['comment'] = df['comment'].str.strip()
        df['comment'] = df['comment'].str.replace(r'http\S+|www\S+|https\S+', '', regex=True)
        df['comment'] = df['comment'].str.replace(r'[^\w\s]', '', regex=True)
        df['comment'] = df['comment'].str.replace(r'\d+', '', regex=True)
        df['comment'] = df['comment'].str.replace(r'\n', ' ', regex=True)
        df['comment'] = df['comment'].str.replace(r'\s+', ' ', regex=True).str.strip()

        df['comment'] = df['comment'].apply(
            lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words])
        )

        df['comment'] = df['comment'].apply(
            lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()])
        )

        return df
    except Exception as e:
        logging.error(f"Error cleaning text: {e}")
        raise


def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, output_path: Path):
    try:
        logging.info("Saving processed data")

        output_path.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(output_path / "preprocessed_train.csv", index=False)
        test_df.to_csv(output_path / "preprocessed_test.csv", index=False)

        logging.info("Processed data saved successfully")
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise


def main():
    try:
        train_path = "data/raw/train.csv"
        test_path = "data/raw/test.csv"
        output_path = Path("data/interim")

        train_df, test_df = load_data(train_path, test_path)

        download_nltk_resources()
        stop_words = get_filtered_stopwords()
        lemmatizer = WordNetLemmatizer()

        train_df = clean_text_column(train_df, stop_words, lemmatizer)
        test_df = clean_text_column(test_df, stop_words, lemmatizer)

        save_data(train_df, test_df, output_path)

    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()