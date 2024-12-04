# syntax=docker/dockerfile:1
FROM python:3.12-alpine
WORKDIR /code
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD python opc_server.py