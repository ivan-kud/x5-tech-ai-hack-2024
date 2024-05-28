import argparse
import os
from pathlib import Path

from datasets import Dataset, DatasetDict
import evaluate
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from transformers import TrainingArguments, Trainer
import wandb

from model_predict import Model
from utils import preprocess_data


DATA_PATH = Path("./data/")
MODELS_OUTPUT_PATH = Path("./saved_models/")
MODELS_SAVE_PATH = Path("./models_output/")
MODELS = [
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    "sileod/deberta-v3-large-tasksource-nli",
    "microsoft/deberta-v2-xlarge-mnli",
]
DEVICES = ["cpu", "cuda", "auto"]

OPTIMIZER = "adamw_torch"
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 3e-5

SCHEDULER = "cosine"
WARMUP_RATIO = 0.00
WARMUP_STEPS = 0

TRAIN_EPOCHS = 2
TRAIN_STEPS = 90
LOGGING_STEPS = 10

USE_CPU = False  # Used 2 GPUs
AUTO_FIND_BATCH_SIZE = False
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2

LOGGER_PROJECT_NAME = "X5-HACK"
SEED = 42


def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=MODELS)
    parser.add_argument("--device", type=str, choices=DEVICES, default=None)
    parser.add_argument("--cv_splits", type=int, default=None)
    args = parser.parse_args()
    print(f"\n{args=}")

    # Load dataframe
    df = pd.read_csv(DATA_PATH / "share" / "train.csv", index_col="line_id")
    preprocess_data(df)

    # Prepare dataframe
    df["premise"] = df["summary"]
    df["hypothesis"] = "Вопрос: " + df["question"] + "\n\nОтвет: " + df["answer"]
    df.rename_axis("idx", inplace=True)
    df["label"] = df["is_hallucination"]
    if args.model_name in [
        "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "sileod/deberta-v3-large-tasksource-nli",
    ]:
        df["label"] = df["label"].map({0: 0, 1: 2})
    elif args.model_name in [
        "microsoft/deberta-v2-xlarge-mnli",
    ]:
        df["label"] = df["label"].map({0: 2, 1: 0})

    # Define metric
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Split dataframe for training and validation parts
    if args.cv_splits is None:
        splits = zip([df.index], [df.index])
    else:
        cv = StratifiedKFold(n_splits=args.cv_splits)
        splits = cv.split(df, df["label"])

    # Fine-tune model for each split
    output_dfs = []
    for i, (train_index, test_index) in enumerate(splits):
        # Initialize logger
        group_name = (f"{args.model_name.replace('/', '-')}"
            + f"_lr{LEARNING_RATE:.0e}_epochs{TRAIN_EPOCHS}")
        run_name = "main-2" if args.cv_splits is None else f"split-{i}"
        wandb.init(project=LOGGER_PROJECT_NAME, name=run_name, group=group_name)
        print(f"\nTraining of {args.model_name}, {run_name}:")

        # Initialize model and tokenizer
        model = Model(args.model_name, args.device)
        if args.model_name == "sileod/deberta-v3-large-tasksource-nli":
            tokenizer_args = dict(padding="max_length", max_length=1158)
        else:
            tokenizer_args = dict(padding=True)

        def tokenize_function(examples):
            return model.tokenizer(
                examples["premise"],
                examples["hypothesis"],
                truncation=True,
                **tokenizer_args,
            )

        # Form dataset
        train_df = df.loc[train_index]
        valid_df = df.loc[test_index]
        dataset = DatasetDict()
        dataset["train"] = Dataset.from_pandas(
            train_df[["premise", "hypothesis", "label"]]
        )
        dataset["validation"] = Dataset.from_pandas(
            valid_df[["premise", "hypothesis", "label"]]
        )
        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # Define directory to save model
        model_output_dir = (MODELS_OUTPUT_PATH
                            / args.model_name.replace("/", "-")
                            / run_name)

        # Train
        training_args = TrainingArguments(
            output_dir=model_output_dir,
            overwrite_output_dir=True,
            use_cpu=USE_CPU,
            auto_find_batch_size=AUTO_FIND_BATCH_SIZE,
            per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
            per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            num_train_epochs=TRAIN_EPOCHS,
            max_steps=TRAIN_STEPS,
            optim=OPTIMIZER,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_type=SCHEDULER,
            warmup_ratio=WARMUP_RATIO,
            warmup_steps=WARMUP_STEPS,
            eval_strategy="no" if args.cv_splits is None else "steps",
            eval_steps=3,
            log_level="passive",
            logging_strategy="steps",
            logging_steps=LOGGING_STEPS,
            save_strategy="epoch",
            save_total_limit=1,
            save_only_model=True,
            load_best_model_at_end=False,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to="wandb",
            run_name=run_name,
            seed=SEED,
            hub_model_id=f"{args.model_name.split('/')[1]}-{run_name}",
            push_to_hub=False,
        )
        trainer = Trainer(
            model=model.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            compute_metrics=compute_metrics,
            tokenizer=model.tokenizer,
        )
        if args.cv_splits is not None:
            trainer.evaluate()
        trainer.train()
        if args.cv_splits is None:
            trainer.push_to_hub(token=os.environ.get("HF_WRITE_TOKEN"))
        wandb.finish()

        # Predict validation split
        print(f"\nPredicting by {args.model_name}:")
        model.model_name = f"ivankud/{args.model_name.split('/')[1]}"
        probs = model.predict_df(valid_df)
        output_valid_df = pd.DataFrame(index=valid_df.index)
        for key in probs:
            output_valid_df[key] = probs[key]
        output_dfs.append(output_valid_df)

    # Define path to save model prediction
    if args.cv_splits is None:
        output_dir = MODELS_OUTPUT_PATH / f"{args.model_name.replace('/', '-')}"
        output_file = output_dir / "biased_probs.csv"
    else:
        output_dir = MODELS_SAVE_PATH / f"{args.model_name.replace('/', '-')}-ft"
        output_file = output_dir / "probs.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write output df to file
    output_df = pd.concat(output_dfs)
    output_df.rename_axis("line_id", inplace=True)
    output_df.sort_index(inplace=True)
    output_df.to_csv(output_file, index=True, float_format="%.8f")
    print("Done!", output_file)


if __name__ == "__main__":
    main()
