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
def bclone(config: str, overrides: tuple[str, ...]) -> None:
    """Run behaviour cloning training on offline data.

    This command trains linear and deep learning models to clone the
    behavior present in offline trajectories. Models are evaluated using
    cross-validation and scatter plots are generated comparing predictions.

    Additional config values can be overridden using key=value syntax:

        corecli offline bclone --config cfg.yaml behaviour_clone.k_folds=10
    """
    # Set up sys.argv for lib_config to parse
    # lib_config expects: [script_name, --config-name=path, key=value, key=value, ...]
    sys.argv = ["bclone", f"--config-name={config}", *overrides]

    # Import and call the original script's main function
    # The @load_config decorator will handle config loading from sys.argv
    from coreoffline.scripts.behaviour_clone import main

    main()
