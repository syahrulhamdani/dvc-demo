import argparse
from time import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DATA_SPLIT_DIR = Path("data/split")


def _get_args(argv=None):
    """Get arguments from user input."""
    parser = argparse.ArgumentParser(
        description="Prepare dataset for training."
    )
    parser.add_argument(
        "-d", "--data",
        type=str,
        help="Filepath to raw data. Example: data/data.csv"
    )
    args, _ = parser.parse_known_args(argv)
    return args


def _check_dir(directory: Path):
    if not directory.exists():
        print(f"{directory} doesn't exist. Creating one...")
        directory.mkdir(parents=True)

    print(f"{directory} exist!")


def main(argv=None):
    """Main function to prepare data."""
    args = _get_args(argv)

    print("Start preparing data")
    start = time()

    data_filepath = Path(args.data)
    if not data_filepath.exists():
        raise f"{data_filepath} not found."

    data = pd.read_csv(data_filepath)
    print("Loaded data with shape:", data.shape)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns=["target", "scrape_date"]),
        data["target"],
        test_size=.2,
        random_state=111
    )
    print(
        f"Got {X_train.shape[0]} for training and {X_test.shape[0]} for test"
    )

    phishing_train = pd.concat([X_train, y_train], axis=1)
    phishing_test = pd.concat([X_test, y_test], axis=1)

    _check_dir(DATA_SPLIT_DIR)
    phishing_train.to_csv(DATA_SPLIT_DIR / "phishing_train.csv", index=False)
    phishing_test.to_csv(DATA_SPLIT_DIR / "phishing_test.csv", index=False)
    print("Done splitting data")


if __name__ == "__main__":
    main()
