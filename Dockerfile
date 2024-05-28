FROM python:3.10-slim

WORKDIR /app

# Download models from hub
RUN pip install --no-cache-dir huggingface-hub
RUN huggingface-cli download ivankud/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
# RUN huggingface-cli download ivankud/deberta-v3-large-tasksource-nli
# RUN huggingface-cli download ivankud/deberta-v2-xlarge-mnli

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

VOLUME /app/data

RUN chmod +x model_predict.py && chmod +x make_prediction.py

CMD ["python", "make_prediction.py"]

# CMD ["uvicorn", "backend:app", "--host", "127.0.0.1", "--port", "8000"]