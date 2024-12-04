# Create the certificates
import shutil
from pathlib import Path
import argparse
import tempfile

from generate_keycert.keycert import (
    gen_cert_key_pair
)


def setup(cert_path: Path, env_cfg_path: Path) -> None:
    # e2e_path = Path(__file__).parent.absolute()

    # server cert
    trusted_path = cert_path / "certificates" / "trusted" / "certs"
    # trusted_path = e2e_path / "opc_server/certificates/trusted/certs"
    server_path = cert_path / "certificates" / "server"
    # server_path = e2e_path / "opc_server/certificates/server"
    trusted_path.mkdir(parents=True, exist_ok=True)
    server_path.mkdir(parents=True, exist_ok=True)
    gen_cert_key_pair(target_path=server_path, server=True)

    # telegraf cert
    telegraf_path = cert_path / "telegraf" / "certificates"
    # telegraf_path = e2e_path / "telegraf/certificates"
    telegraf_path.mkdir(parents=True, exist_ok=True)
    gen_cert_key_pair(target_path=telegraf_path, name="telegraf")
    shutil.copy(src=telegraf_path / "cert.pem", dst=trusted_path / "telegraf.pem")

    # agent cert
    agent_path = cert_path / "agent" / "certificates"
    # agent_path = e2e_path / "agent/certificates"
    agent_path.mkdir(parents=True, exist_ok=True)
    gen_cert_key_pair(target_path=agent_path, name="agent")
    shutil.copy(src=agent_path / "cert.pem", dst=trusted_path / "agent.pem")

    # # copy agent config, used in docker build
    # shutil.copy(src=env_cfg_path, dst=e2e_path / "env.yaml")
    # Watch out, we still havent used the following!
    # shutil.copy(src=env_cfg_path, dst=e2e_path / "telegraf/env.yaml")

    # # # clean up docker containers
    # # cleanup_script_path = e2e_path / "cleanup_docker_images.sh"
    # # subprocess.run(["sudo", cleanup_script_path.absolute()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        help="env config file, usually in config/env/[your_env].yaml",
        default="config/env/scrubber_online_test.yaml"
    )
    args = parser.parse_args()
    env_cfg_path = Path(args.env)
    file_path = Path(__file__).parent

    with tempfile.TemporaryDirectory(dir=file_path) as tmpdirname:
        cert_path = file_path / tmpdirname
        setup(cert_path, env_cfg_path=env_cfg_path)
        print(f"Certificates created in temporary directroy: {cert_path.absolute()}\n"
              "Press ENTER to exit")
        input()
