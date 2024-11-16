FROM python:3.10-slim
WORKDIR /BIV_HACK

COPY requirements.txt requirements.txt
COPY data/payments_main.tsv data/payments_main.tsv
COPY data/payments_training.tsv data/payments_training.tsv
ADD folder ru_core_news_sm-3.8.0
COPY main.py main.py

RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install openpyxl


CMD ["python", "./main.py"]