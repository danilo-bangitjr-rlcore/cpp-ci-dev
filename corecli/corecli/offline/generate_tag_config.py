import sys

import click


@click.command(name="generate-tag-config")
@click.option(
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to the configuration YAML file",
)
@click.argument("overrides", nargs=-1)
def generate_tag_config(config: str, overrides: tuple[str, ...]) -> None:
    """Generate tag configurations from database statistics.

    This command reads tag statistics from TimescaleDB and generates
    a YAML configuration file with operating and expected ranges based
    on percentiles.

    Additional config values can be overridden using key=value syntax:

        corecli offline generate-tag-config --config cfg.yaml csv_path=/path/to/data.csv
    """
    # Set up sys.argv for lib_config to parse
    # lib_config expects: [script_name, --config-name=path, key=value, key=value, ...]
    sys.argv = ["generate_tag_config", f"--config-name={config}", *overrides]

    # Import and call the original script's main function
    # The @load_config decorator will handle config loading from sys.argv
    from coreoffline.scripts.generate_tag_configs import main

    main()
