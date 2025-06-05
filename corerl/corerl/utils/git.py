from pathlib import Path


def get_active_branch(base: Path = Path('.')) -> str:
    refs = (base / '.git/HEAD').read_text()

    for ref in refs.splitlines():
        if ref.startswith('ref: '):
            return ref.partition('refs/heads/')[2]

    raise Exception('Was unable to determine active branch')
