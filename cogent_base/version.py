import importlib.metadata
import os
from contextlib import suppress
from pathlib import Path


def get_cogent_version() -> str:
    with suppress(FileNotFoundError, StopIteration):
        with open(os.path.join(Path(__file__).parent.parent, "pyproject.toml"), encoding="utf-8") as pyproject_toml:
            version = next(line for line in pyproject_toml if line.startswith("version")).split("=")[1].strip("'\"\n ")
            return f"{version}-trunk"
    try:
        return importlib.metadata.version("cogent-base")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
