# syntax=docker/dockerfile:1

# Ensure that an SSH agent that is able to access our private linesearchopt dependency is forwarded:
#
#  docker build -t rlcoretech/corerl --ssh default=$SSH_AUTH_SOCK .

# Stage 1, compile dependencies with SSH key forwarding to pull linesearchopt from private Github repository
# Bookworm image is needed due to dependency on git cli
FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

# Setup and download public key for Github
RUN mkdir -p -m 0600 ~/.ssh &&\
  ssh-keyscan github.com >> ~/.ssh/known_hosts

WORKDIR /app

# Copy minimal source code for building corerl package
COPY ./corerl /app/corerl
COPY ./pyproject.toml /app/pyproject.toml

# Install the corerl dependencies
RUN --mount=type=ssh \
  uv pip compile --extra=linesearchopt_gh pyproject.toml -o deps.txt && \
  # This step ensures that our dependencies exist in a folder called 'vendor'
  # which can be referenced within setuptools and added to our generated corerl wheel
  uv pip install --system --target /app/vendor -r deps.txt

# Build corerl package which emits built .whl into /app/dist
RUN uv build --wheel

# See also: https://github.com/rlcoretech/core-rl/pull/347#discussion_r1906215954
# Convert our wheel such that we only include .pyc files
RUN uv pip install --system pyc_wheel &&\
  whl_file_name=$(ls /app/dist/corerl-*.whl) &&\
  python -m pyc_wheel "$whl_file_name"

# Stage 2, install corerl to minimal Python 3 image
FROM python:3.12-slim AS corerl

# This label is used by Github to link the package to our repository,
# otherwise it will be scoped to our rlcoretech organization
LABEL org.opencontainers.image.source=https://github.com/rlcoretech/core-rl

COPY --from=base /app/dist /app/dist
WORKDIR /app

# RUN pip install /app/dist/corerl-*.whl
# Our corerl image is quite large, mostly due to our python dependencies. The following flags are added:
# --no-cache-dir (reduces our image size by ~3.3G)
# --no-compile (reduces our image size by ~0.2G)
RUN pip --no-cache-dir install --no-compile /app/dist/corerl-*.whl

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
