# syntax=docker/dockerfile:1

# Ensure that an SSH agent that is able to access our private dependencies is forwarded:
#
#  docker build -t rlcoretech/corerl --ssh default=$SSH_AUTH_SOCK .

# Stage: compile dependencies with SSH key forwarding to pull dependencies from private Github repository
# Bookworm image is needed due to dependency on git cli
FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

# Setup and download public key for Github
RUN mkdir -p -m 0600 ~/.ssh &&\
  ssh-keyscan github.com >> ~/.ssh/known_hosts

WORKDIR /app

# Copy minimal pyproject.toml for dependencies
COPY ./pyproject.toml /app/pyproject.toml

# Install the corerl dependencies
RUN --mount=type=ssh \
  uv pip compile --extra=coreenv_gh pyproject.toml -o deps.txt && \
  # This step ensures that our dependencies exist in a folder called 'vendor'
  # which can be referenced within setuptools and added to our generated corerl wheel
  uv pip install --system --target /app/vendor -r deps.txt

# copy source code for building corerl package
COPY ./corerl /app/corerl

# Build corerl package which emits built .whl into /app/dist
RUN uv build --wheel

# See also: https://github.com/rlcoretech/core-rl/pull/347#discussion_r1906215954
# Convert our wheel such that we only include .pyc files
RUN uv pip install --system pyc_wheel &&\
  whl_file_name=$(ls /app/dist/corerl-*.whl) &&\
  python -m pyc_wheel "$whl_file_name"

# Stage: install corerl to minimal Python 3 image
FROM python:3.12-slim AS corerl

COPY --from=base /app/dist /app/dist
WORKDIR /app

# Our corerl image is quite large with default cuda dependencies.
# RUN pip install /app/dist/corerl-*.whl
# Minimal CPU supported installation is used instead.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu &&\
  pip --no-cache-dir install --no-compile /app/dist/corerl-*cp*.whl

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
