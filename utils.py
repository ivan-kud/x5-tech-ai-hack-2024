import pandas as pd


def preprocess_data(df: pd.DataFrame):
    """Inplace preprocessing of input data"""
    df['answer'] = df['answer'].str.slice(stop=500)


def norm_probs(a: float, b: float) -> tuple[float, float]:
    """Normalization of two numbers to 1"""
    if (sum_ := a + b) > 0:
        a, b = min(a / sum_, 1.0), min(b / sum_, 1.0)
    else:
        a, b = 0.5, 0.5
    return a, b
