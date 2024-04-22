import hydra
from omegaconf import OmegaConf, DictConfig


@hydra.main(version_base=None, config_name='config', config_path="../config")
def main(cfg: DictConfig) -> None:
    OmegaConf.save(cfg, "temp-config.yaml")


if __name__ == "__main__":
    main()
