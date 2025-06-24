# !!!! ATTENTION: BROKEN !!!!
# The image build successfully, but it doesn't install the python packages correctly

# syntax=docker/dockerfile:1

# Ensure that an SSH agent that is able to access our private dependencies is forwarded:
#
#  docker build -t rlcoretech/corerl --ssh default=$SSH_AUTH_SOCK .

# Stage: compile dependencies with SSH key forwarding to pull dependencies from private Github repository
# Bookworm image is needed due to dependency on git cli
FROM ghcr.io/astral-sh/uv:python3.13-bookworm AS base

# Setup and download public key for Github
RUN mkdir -p -m 0600 ~/.ssh &&\
  ssh-keyscan github.com >> ~/.ssh/known_hosts

WORKDIR /app

COPY ./coreio /app/coreio
COPY ./libs /app/libs
COPY ./test /app/test
COPY ./corerl /app/corerl

WORKDIR /app/corerl

# Install the corerl dependencies
RUN --mount=type=ssh \
  uv sync --no-dev

# Build corerl package which emits built .whl into /app/dist
RUN uv build --wheel

# See also: https://github.com/rlcoretech/core-rl/pull/347#discussion_r1906215954
# Convert our wheel such that we only include .pyc files
RUN uv pip install --system "pyc_wheel==1.3.0" &&\
  whl_file_name=$(ls /app/corerl/dist/corerl-*.whl) &&\
  python -m pyc_wheel "$whl_file_name"

# Stage: install corerl to minimal Python 3 image
FROM python:3.13-slim AS corerl

# needed for health check within compose.yaml
RUN apt-get update && apt-get install curl -y

COPY --from=base /app/corerl/dist /app/dist
WORKDIR /app

# Set up the entrypoint to reference our corerl main script, dynamically pass arguments on run
ENTRYPOINT ["corerl_main"]

# To open an interactive shell with the local configurations mounted:
#
#  docker run --volume $(pwd)/config:/app/config:ro --entrypoint=/bin/sh -it rlcoretech/corerl
#
# To run the `corerl_main` script using a configuration, e.g. mountain_car_continuous
# and have any emitted outputs written to a local machine accessible output directory:
#
#  docker run \
#    --volume $(pwd)/config:/app/config:ro \
#    --volume $(pwd)/output:/app/output \
#    rlcoretech/corerl \
#    --config-name mountain_car_continuous
#
# After this runs, the experiment output should exist within your $(pwd)/output folder.
# This may require permissions to be adjusted after the container runs:
#
#   sudo chown -R $(id -u):$(id -g) $(pwd)/output
