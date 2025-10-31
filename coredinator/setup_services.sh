#!/bin/bash
# This script is designed to be run in Git Bash on Windows.

# 1. Check for a command-line argument for the volume (e.g., 'c', 'd').
#    If the first argument ($1) is non-empty, use it.
#    Otherwise, default to 'c'.
VOLUME_LETTER="${1:-c}"

# 2. Convert the letter to the Git Bash path convention (/c, /d, etc.)
#    Ensure it is lowercase for consistency.
GIT_BASH_VOLUME_ROOT="/$(echo "$VOLUME_LETTER" | tr '[:upper:]' '[:lower:]')"

# 3. Define the base path using the dynamically set volume root
BASE_PATH="$GIT_BASH_VOLUME_ROOT/RLCore/services"

echo "Creating directory structure at $BASE_PATH..."

# Create all directories in one command.
# The -p flag ensures that all parent directories are created.
# Brace expansion {dir1,dir2} creates multiple subdirectories.
mkdir -p "$BASE_PATH"/{agent/{bin,certs,config,logs},grafana,postgresql/data}

# Create the empty files
echo "Creating configuration files..."
touch "$BASE_PATH/grafana/config.txt"
touch "$BASE_PATH/postgresql/setup.txt"

echo "Done. Directory structure created successfully."
