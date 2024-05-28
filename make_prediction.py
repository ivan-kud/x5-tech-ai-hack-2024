import os


USE_ENSEMBLE = False

if USE_ENSEMBLE:
    # Predict by ensemble
    os.system("python ./model_predict.py --model_name ivankud/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    os.system("python ./model_predict.py --model_name ivankud/deberta-v3-large-tasksource-nli")
    os.system("python ./model_predict.py --model_name ivankud/deberta-v2-xlarge-mnli")
    os.system("python ./collect_predictions.py")
    os.system("python ./ensemble_predict.py")
else:
    # Predict by one model: choose one
    os.system("python ./model_predict.py --model_name ivankud/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    # os.system("python ./model_predict.py --model_name ivankud/deberta-v3-large-tasksource-nli")
    # os.system("python ./model_predict.py --model_name ivankud/deberta-v2-xlarge-mnli")
