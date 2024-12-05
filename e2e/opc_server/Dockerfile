# syntax=docker/dockerfile:1
FROM python:3.12-alpine
WORKDIR /code
RUN pip install asyncua
COPY . .
CMD python opc_server.py
