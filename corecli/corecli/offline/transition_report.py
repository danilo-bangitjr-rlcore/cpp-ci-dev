import sys

import click


@click.command(name="transition-report")
@click.option(
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to the configuration YAML file",
)
@click.argument("overrides", nargs=-1)
def transition_report(config: str, overrides: tuple[str, ...]) -> None:
    """Generate transition analysis report from offline data.

    This command runs the data pipeline on offline data from TimescaleDB,
    collects transitions, and generates a report with
    statistics about state transitions, rewards, and goal violations.

    Additional config values can be overridden using key=value syntax:

        corecli offline transition-report --config cfg.yaml report.output_dir=./results
    """
    # Set up sys.argv for lib_config to parse
    # lib_config expects: [script_name, --config-name=path, key=value, key=value, ...]
    sys.argv = ["transition_report", f"--config-name={config}", *overrides]

    # Import and call the original script's main function
    # The @load_config decorator will handle config loading from sys.argv
    from coreoffline.scripts.create_transition_report import main

    main()
