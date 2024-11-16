FROM python:3.8-slim-buster
WORKDIR /BIV_HACK

COPY requirements.txt requirements.txt
COPY main.py main.py

RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install openpyxl


CMD ["python", "./main.py"]