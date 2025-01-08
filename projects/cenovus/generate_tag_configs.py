from pathlib import Path
from corerl.configs.config import config
from corerl.configs.loader import load_config
from cenovus.utils.data import get_file_paths
from corerl.data_pipeline.db.data_reader import DataReader, TagDBConfig, TagStats


@config(allow_extra=True)
class Config:
    db: TagDBConfig


def build_tag_config(tag: TagStats):
    return f"""
- name: {tag.tag}
  is_action: false
  imputer:
    name: copy
    imputation_horizon: 60
  state_constructor:
    - name: normalize
      min: {tag.min}
      max: {tag.max}
"""


@load_config(Config, base='projects/cenovus/configs', config_name='generate_tag_configs')
def generate(cfg: Config):
    reader = DataReader(cfg.db)
    tags = get_file_paths()

    stats: list[TagStats] = []
    for tag_file in tags:
        tag_name = tag_file.name.removesuffix('.json')

        tag_stats = reader.get_tag_stats(tag_name)
        stats.append(tag_stats)


    tab = '  '
    with Path('projects/cenovus/configs/tags.yaml').open('w') as f:
        f.write('tags:')
        for tag in stats:
            # in exceptional cases, there is only np.nan data recorded for a tag
            # so both tag.min and tag.max are None. We should just skip these tags
            if tag.min is None:
                continue

            tag_cfg = build_tag_config(tag)
            for line in tag_cfg.splitlines():
                f.write(tab + line + '\n')

            f.write('\n')


if __name__ == '__main__':
    generate()
