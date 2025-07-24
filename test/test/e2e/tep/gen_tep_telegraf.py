import logging
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from test.e2e.tep.opc_tep import ALL_NODE_NAMES

logger = logging.getLogger(__name__)

@dataclass
class TagData:
    id_name: str
    name: str
    ns: int
    id_type: str = "s"

def generate_telegraf_conf(tag_data: Sequence[TagData]):
    source_path = Path("../telegraf/base_telegraf.conf")
    target_path = Path("./telegraf/tennessee_telegraf.conf")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target_path)

    block = ""
    with open(target_path, "a") as f:
        for row in tag_data:
            block += "[[inputs.opcua.nodes]]\n"
            block += " " * 2 + f'namespace = "{row.ns}"\n'
            block += " " * 2 + f'identifier_type = "{row.id_type}"\n'
            block += " " * 2 + f'identifier = "{row.id_name}"\n'
            block += " " * 2 + 'name = "val"\n'
            block += " " * 2 + f'default_tags = {{ name = "{row.name}" }}\n'
            block += "\n"
        f.write(block)

    logger.info(f"Generated {target_path}")

def main():
    all_tag_names = ALL_NODE_NAMES
    tag_data = [TagData(id_name=name, name=name, ns=2) for name in all_tag_names]
    generate_telegraf_conf(tag_data)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        encoding="utf-8",
        level=logging.DEBUG,
    )

    main()

