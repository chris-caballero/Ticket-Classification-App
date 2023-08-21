FROM python:3.10-slim

WORKDIR /App
COPY requirements.txt /App/
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY . /App/