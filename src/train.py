"""Script to train model."""
import argparse
import json
from time import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def _get_args(argv=None):
    """Get arguments from user input."""
    parser = argparse.ArgumentParser(
        description="Train model."
    )
    parser.add_argument(
        "-d", "--data",
        type=str,
        help="Filepath to load training data. Example: data/train.csv"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        help="Filepath to save the trained model. Example: model.joblib"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        help="Filepath to save the model performance. Example: metrics.json"
    )
    args, _ = parser.parse_known_args(argv)
    return args


def _check_dir(directory: Path):
    if not directory.exists():
        print(f"{directory} doesn't exist. Creating one...")
        directory.mkdir(parents=True)

    print(f"{directory} exist!")


def create_model():
    model = Pipeline(
        [
            (
                "transformer", ColumnTransformer(
                    [("encoder", OneHotEncoder(), ["ext"])],
                    remainder="passthrough"
                )
            ),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression())
        ]
    )
    return model


def main(argv=None):
    """Main function to train model."""
    args = _get_args(argv)

    print("Start training model")
    start = time()

    data_filepath = Path(args.data)
    _check_dir(data_filepath.parent)

    model_filepath = Path(args.model)
    _check_dir(model_filepath.parent)

    metrics_filepath = Path(args.metrics)
    _check_dir(metrics_filepath.parent)

    data = pd.read_csv(data_filepath)
    print("Loaded training data with shape:", data.shape)

    features, target = data.drop(columns=["target"]), data["target"]
    model = create_model()
    model.fit(features, target)

    joblib.dump(model, model_filepath)

    predictions = model.predict(features)
    report = classification_report(target, predictions, output_dict=True)
    with open(metrics_filepath, "w") as f:
        json.dump(report, f)

    print(f"Done training in {time()-start:3f}s")


if __name__ == "__main__":
    main()
