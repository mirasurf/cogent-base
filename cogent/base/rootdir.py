from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def get_absolute_path(path_from_root: str) -> str:
    absolute_path = ROOT_DIR / path_from_root
    return str(absolute_path.resolve())
