import hydra
import asyncio
import datetime as dt

from omegaconf import DictConfig
from corerl.utils.processes import keep_alive


async def async_main(cfg: DictConfig):
    exec = str(cfg.deployment.python_executable).split(' ')
    python_entrypoint = cfg.deployment.python_entrypoint
    config_name = cfg.deployment.config_name
    options = cfg.deployment.options

    cmd = [*exec, python_entrypoint, '--config-name', config_name, *options]

    await keep_alive(
        cmd,
        base_backoff=2,
        max_backoff=dt.timedelta(hours=1),
    )


@hydra.main(version_base=None, config_name='config', config_path="config/")
def main(cfg: DictConfig):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_main(cfg))


if __name__ == '__main__':
    main()
