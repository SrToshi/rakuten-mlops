import pandas as pd
import nltk
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import json
import math


def build_text_input(df):
    """Assemble the text the model sees, from designation + description.

    This exists as a shared function because training and inference used to
    build that text differently: training concatenated designation with
    description, while predict.py passed description alone and never looked
    at designation. Rows with an empty description therefore reached the
    model as an empty string, and every one of them got the same prediction
    with identical confidence.

    It also replaces the original `designation + str(description)`, where
    str() on a pandas Series stringified the ENTIRE column and appended that
    same blob to every row.
    """
    designation = (
        df["designation"].fillna("") if "designation" in df.columns else ""
    )
    description = (
        df["description"].fillna("") if "description" in df.columns else ""
    )
    return (designation + " " + description).str.strip()


class DataImporter:
    def __init__(self, filepath="data/preprocessed"):
        self.filepath = filepath

    def load_data(self):
        data = pd.read_csv(f"{self.filepath}/X_train_update.csv")
        data["description"] = build_text_input(data)
        data = data.drop(["Unnamed: 0", "designation"], axis=1)

        target = pd.read_csv(f"{self.filepath}/Y_train_CVw08PX.csv")
        target = target.drop(["Unnamed: 0"], axis=1)
        modalite_mapping = {
            modalite: i for i, modalite in enumerate(target["prdtypecode"].unique())
        }
        target["prdtypecode"] = target["prdtypecode"].replace(modalite_mapping)

        with open("models/mapper.pkl", "wb") as fichier:
            pickle.dump(modalite_mapping, fichier)

        # predict.py and api.py read mapper.json (index -> prdtypecode), which
        # nothing ever regenerated: a retrained model would therefore decode its
        # class indices with a stale mapping. Write it right here, beside the
        # pickle, so the two can never drift apart.
        with open("models/mapper.json", "w", encoding="utf-8") as fichier:
            json.dump(
                {str(index): str(code) for code, index in modalite_mapping.items()},
                fichier,
                indent=2,
            )

        df = pd.concat([data, target], axis=1)

        return df

    def split_train_test(
        self,
        df,
        samples_per_class=600,
        val_samples_per_class=50,
        test_samples_per_class=None,
    ):
        """Split the dataset into three DISJOINT, class-balanced sets.

        The original implementation drew the validation rows out of the test
        set and never removed them, so validation was a subset of test. Any
        metric reported on "test" was therefore reported on rows that had
        already been used to fit the networks (both branches train with
        validation_data=X_val and an EarlyStopping that watches it).

        That matters here because the three sets have three different jobs:

          train : fit the text and image branches
          val   : early stopping, and the search for the ensemble blend
                  weights — both are model selection
          test  : the single number that decides champion versus challenger,
                  computed on rows no fitting decision has ever seen

        test_samples_per_class caps the test set. Each test row costs a VGG16
        forward pass, so without a cap a small training run would spend most
        of its time scoring the leftovers.
        """
        train_parts, val_parts, test_parts = [], [], []

        for _, group in df.groupby("prdtypecode"):
            train = group.sample(n=samples_per_class, random_state=42)
            remaining = group.drop(train.index)

            n_val = min(val_samples_per_class, len(remaining))
            val = remaining.sample(n=n_val, random_state=42)
            remaining = remaining.drop(val.index)

            if test_samples_per_class is not None:
                n_test = min(test_samples_per_class, len(remaining))
                remaining = remaining.sample(n=n_test, random_state=42)

            train_parts.append(train)
            val_parts.append(val)
            test_parts.append(remaining)

        def _shuffle_and_split(parts):
            # Shuffled as one frame, then separated. Shuffling X and y
            # independently happens to line up today because both calls draw
            # the same permutation, but it only takes one differing row count
            # for the labels to silently stop matching their features.
            frame = pd.concat(parts).sample(frac=1, random_state=42)
            frame = frame.reset_index(drop=True)
            labels = frame["prdtypecode"]
            features = frame.drop(["prdtypecode"], axis=1)
            return features, labels

        X_train, y_train = _shuffle_and_split(train_parts)
        X_val, y_val = _shuffle_and_split(val_parts)
        X_test, y_test = _shuffle_and_split(test_parts)

        return X_train, X_val, X_test, y_train, y_val, y_test


class ImagePreprocessor:
    def __init__(self, filepath="data/preprocessed/image_train"):
        self.filepath = filepath

    def preprocess_images_in_df(self, df):
        df["image_path"] = (
            f"{self.filepath}/image_"
            + df["imageid"].astype(str)
            + "_product_"
            + df["productid"].astype(str)
            + ".jpg"
        )


class TextPreprocessor:
    def __init__(self):
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("wordnet")
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(
            stopwords.words("french")
        )  # Vous pouvez choisir une autre langue si nécessaire

    def preprocess_text(self, text):

        if isinstance(text, float) and math.isnan(text):
            return ""
        # Supprimer les balises HTML
        text = BeautifulSoup(text, "html.parser").get_text()

        # Supprimer les caractères non alphabétiques
        text = re.sub(r"[^a-zA-Z]", " ", text)

        # Tokenization
        words = word_tokenize(text.lower())

        # Suppression des stopwords et lemmatisation
        filtered_words = [
            self.lemmatizer.lemmatize(word)
            for word in words
            if word not in self.stop_words
        ]

        return " ".join(filtered_words[:10])

    def preprocess_text_in_df(self, df, columns):
        for column in columns:
            df[column] = df[column].apply(self.preprocess_text)
