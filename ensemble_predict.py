from pathlib import Path

import catboost as cb
import pandas as pd


DATA_PATH = Path('./data/')
MODELS_OUTPUT_PATH = Path('./models_output/')
METAMODEL_PATH = Path('./metamodel/')


def main():
    print('\nPredicting by metamodel:')

    # Load data
    X = pd.read_csv(MODELS_OUTPUT_PATH / 'data_X.csv', index_col='line_id')

    # Load metamodel
    model = cb.CatBoostClassifier()
    model.load_model(METAMODEL_PATH / 'metamodel.cbm')

    # Predict
    X['is_hallucination'] = model.predict(X).tolist()
    
    # Write submission file
    submission_file = DATA_PATH / "submission.csv"
    X["is_hallucination"].to_csv(submission_file, index=True)
    print("Done!", submission_file)


if __name__ == "__main__":
    main()
