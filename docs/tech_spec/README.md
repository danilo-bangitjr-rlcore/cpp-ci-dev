# How to Generate the CoreRL Technical Specification DOCX

This document outlines the steps to regenerate the `Complete_Tech_Spec.docx` file from the source Markdown files.

## 1. Prerequisites & Setup

These steps only need to be performed once. They install all the necessary tools and configure them to work together.

*Note: The `sudo` commands require administrative privileges to install system-wide packages.*

### Step 1.1: Install Core Packages
Install `pandoc`, `nodejs`, `npm`, and `pipx` using Fedora's `dnf` package manager.

```bash
sudo dnf install -y pandoc nodejs npm pipx
```

### Step 1.2: Install Mermaid CLI
Use `npm` to install the Mermaid command-line interface, which renders the diagrams.

```bash
npm install -g @mermaid-js/mermaid-cli
```

### Step 1.3: Install the Pandoc Mermaid Filter
Use `pipx` to install the pandoc filter that enables Mermaid diagram support.

```bash
pipx install pandoc-mermaid-filter
```

### Step 1.4: Create a Wrapper for the Mermaid CLI
The pandoc filter expects an executable named `mermaid`, but the CLI tool is named `mmdc`. We also need to pass a `--quiet` flag to it. The following steps create a script and a symbolic link to bridge this gap.

1.  **Create the wrapper script:** (adjust the Node version path if different)
    ```bash
    echo '#!/usr/bin/env bash\n$(command -v mmdc) "$@" --quiet' > ~/.local/bin/mermaid-wrapper.sh
    ```

2.  **Make the script executable:**
    ```bash
    chmod +x ~/.local/bin/mermaid-wrapper.sh
    ```

3.  **Create a symbolic link named `mermaid` that points to the wrapper:**
    ```bash
    ln -sf ~/.local/bin/mermaid-wrapper.sh ~/.local/bin/mermaid
    ```

## 2. Regeneration Command

Once the setup is complete, you can regenerate the document at any time by running the following `pandoc` command from your home directory. It combines all the specified markdown files in order and converts them into a single DOCX file.

```bash
# Run from the repository root (the directory that contains docs/)
cd /path/to/monorepo

pandoc \
    docs/tech_spec.md \
    docs/tech_spec/coregateway.md \
    docs/tech_spec/corerl.md \
    docs/tech_spec/coreio.md \
    docs/tech_spec/coredinator.md \
    docs/tech_spec/coretelemetry.md \
    docs/tech_spec/libraries.md \
    docs/tech_spec/research.md \
    docs/tech_spec/internal_practices.md \
    docs/tech_spec/config_schemas.md \
    --filter pandoc-mermaid \
    --embed-resources --standalone \
    --toc \
    -o ~/Downloads/Complete_Tech_Spec.docx
```
