# Use Python 3.13 slim image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install uv
RUN pip install uv

# Copy source code
COPY ./corerl /app/corerl
COPY ./libs /app/libs
COPY ./test /app/test
COPY ./coreio /app/coreio

# Set working directory to test
WORKDIR /app/corerl

# needed for health check within compose.yaml
RUN apt-get update && apt-get install curl -y

# Install dependencies using uv
RUN uv sync && uv pip install .
ENV PATH="/app/corerl/.venv/bin:$PATH"

ENTRYPOINT ["corerl_main"]
