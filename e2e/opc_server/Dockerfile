# syntax=docker/dockerfile:1
FROM python:3.13-alpine
WORKDIR /code
RUN pip install asyncua
CMD ["uaserver", "--populate", "--url", "opc.tcp://opc-server:4840/rlcore/server/"]
