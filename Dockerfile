FROM python:3.8-slim-buster
WORKDIR /BIV_HACK

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY main.py main.py



CMD ['python', '-u', 'main.py']