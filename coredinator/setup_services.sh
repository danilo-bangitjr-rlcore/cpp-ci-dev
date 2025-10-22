#!/bin/bash
# This script is designed to be run in Git Bash on Windows.

# Define the base path using Git Bash's path convention (/c/ instead of C:\)
BASE_PATH="/c/RLCore/services"

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
