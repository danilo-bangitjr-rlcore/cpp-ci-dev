import sys

import click


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to the configuration YAML file",
)
@click.argument("overrides", nargs=-1)
def train(config: str, overrides: tuple[str, ...]) -> None:
    """Run offline RL training from data in TimescaleDB.

    This command loads offline transitions from the database and trains
    a reinforcement learning agent using the configured algorithm.

    Additional config values can be overridden using key=value syntax:

        corecli offline train --config cfg.yaml offline_training.offline_steps=50000
    """
    # Set up sys.argv for lib_config to parse
    # lib_config expects: [script_name, --config-name=path, key=value, key=value, ...]
    sys.argv = ["train", f"--config-name={config}", *overrides]

    # Import and call the original script's main function
    # The @load_config decorator will handle config loading from sys.argv
    from coreoffline.scripts.run_offline_training import main

    main()
