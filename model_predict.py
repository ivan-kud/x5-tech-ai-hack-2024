import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from utils import preprocess_data, norm_probs


DATA_PATH = Path("./data/")
OUTPUT_PATH = Path("./models_output/")

DEVICES = ["cpu", "cuda", "auto"]

MODELS = {
    "ivankud/DeBERTa-v3-large-mnli-fever-anli-ling-wanli": {
        "predict_group": "moritz",
        "threshold": 0.50,
    },
    "ivankud/deberta-v3-large-tasksource-nli": {
        "predict_group": "sileod",
        "threshold": 0.50,
    },
    "ivankud/deberta-v2-xlarge-mnli": {
        "predict_group": "microsoft",
        "threshold": 0.27,
    },
}


class Model:
    def __init__(self, model_name: str, device: str | None):
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            device_map=device,
            token=os.environ.get("HF_READ_TOKEN"),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=os.environ.get("HF_READ_TOKEN"),
        )

    def predict(
        self, summary: str, question: str, answer: str
    ) -> tuple[float, float | None]:
        # Define premise and hypothesis
        premise = f"{summary}"
        hypothesis = f"Вопрос: {question}\n\nОтвет: {answer}"

        # Predict
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.no_grad():
            logits = self.model(**inputs.to(self.model.device)).logits
        probabilities = F.softmax(logits, dim=1)

        # Extract contradiction and entailment probs
        entailment_proba = None
        if MODELS[self.model_name]["predict_group"] == "microsoft":
            contradiction_proba = probabilities[0][0].item()
            entailment_proba = probabilities[0][2].item()
        elif MODELS[self.model_name]["predict_group"] in ["sileod", "moritz"]:
            contradiction_proba = probabilities[0][2].item()
            entailment_proba = probabilities[0][0].item()

        return contradiction_proba, entailment_proba

    def predict_df(self, data: pd.DataFrame) -> dict[str, list]:
        # Predict probabilities row by row
        probs = {"p(Contr)": [], "p(Entl)": []}
        for _, row in tqdm(data.iterrows(), total=len(data)):
            contradiction_proba, entailment_proba = self.predict(
                row["summary"],
                row["question"],
                row["answer"],
            )

            # Add predicted probas to lists
            probs["p(Contr)"].append(contradiction_proba)
            if entailment_proba is not None:
                probs["p(Entl)"].append(entailment_proba)

        # Remove entailment probas if they are not returned by model
        if len(probs["p(Entl)"]) == 0:
            del probs["p(Entl)"]

        return probs


def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=MODELS)
    parser.add_argument("--device", type=str, choices=DEVICES, default=None)
    args = parser.parse_args()
    print(f"\n{args=}")

    # Read input df
    input_df = pd.read_csv(DATA_PATH / "test.csv", index_col="line_id")
    preprocess_data(input_df)

    # Create model
    model = Model(args.model_name, args.device)

    # Predict contradiction and entailment probs
    print(f"\nPredicting by {args.model_name}:")
    probs = model.predict_df(input_df)

    # Create output df with predicted probs
    output_df = pd.DataFrame(index=input_df.index)
    for key in probs:
        output_df[key] = probs[key]

    # Calculate hallucination probs and labels
    if "p(Entl)" in output_df:
        output_df["p(Halluc)"] = output_df.apply(
            lambda x: norm_probs(x["p(Contr)"], x["p(Entl)"])[0], axis=1
        )
    else:
        output_df["p(Halluc)"] = output_df["p(Contr)"]
    halluc_threshold = MODELS[args.model_name].get("threshold", 0.5)
    output_df["is_hallucination"] = (output_df["p(Halluc)"] > halluc_threshold).astype("int")

    # Write probs to file
    probs_dir = OUTPUT_PATH / args.model_name.replace("/", "-")
    probs_dir.mkdir(parents=True, exist_ok=True)
    probs_file = probs_dir / "probs.csv"
    if "p(Entl)" in output_df:
        columns = ["p(Contr)", "p(Entl)"]
    else:
        columns = ["p(Contr)"]
    output_df[columns].to_csv(probs_file, index=True, float_format="%.8f")
    print("Done!", probs_file)

    # Write submission file
    submission_file = DATA_PATH / "submission.csv"
    output_df["is_hallucination"].to_csv(submission_file, index=True)
    print("Done!", submission_file)


if __name__ == "__main__":
    main()
