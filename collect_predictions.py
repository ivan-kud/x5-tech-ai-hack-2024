from pathlib import Path

import pandas as pd

from utils import preprocess_data


DATA_PATH = Path("./data/")
MODELS_OUTPUT_PATH = Path("./models_output/")

USE_LENGTH_FEATURES = True


def main():
    print('\nCollecting predictions:')

    # Load input data
    data_file = DATA_PATH / "test.csv"
    X = pd.read_csv(data_file, index_col="line_id")
    preprocess_data(X)
    label_exists = "is_hallucination" in X
    if label_exists:
        y = X["is_hallucination"]
        X.drop(columns="is_hallucination", inplace=True)
    print(data_file)

    # Add length features from input data
    if USE_LENGTH_FEATURES:
        X["s_len"] = X["summary"].str.len()
        X["q_len"] = X["question"].str.len()
        X["a_len"] = X["answer"].str.len()
    X.drop(columns=["summary", "question", "answer"], inplace=True)

    # Load features from transformer-based models
    for model_path in [path for path in MODELS_OUTPUT_PATH.iterdir() if path.is_dir()]:
        for model_file in [path for path in model_path.iterdir() if path.is_file()]:
            df = pd.read_csv(model_file, index_col="line_id")
            for column in df:
                X[f"{column}_{model_path.parts[-1]}{model_file.stem[5:]}"] = df[column]
            print(model_file)

    # Write dataframes to files
    X_file = MODELS_OUTPUT_PATH / "data_X.csv"
    X.to_csv(X_file, index=True, float_format="%.8f")
    print("Done!", X_file)
    if label_exists:
        y_file = MODELS_OUTPUT_PATH / "data_y.csv"
        y.to_csv(y_file, index=True, float_format="%.8f")
        print("Done!", y_file)


if __name__ == "__main__":
    main()
