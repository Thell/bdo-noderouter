# data_store.py

import importlib.resources
import json
from pathlib import Path
import tomllib


def path() -> Path:
    with importlib.resources.as_file(importlib.resources.files().joinpath("data")) as path:
        return path


def is_file(filename: str) -> bool:
    return path().joinpath(filename).is_file()


def read_text(filename: str) -> str:
    return path().joinpath(filename).read_text(encoding="utf-8")


def read_strings_csv(filename: str) -> dict:
    import csv

    filepath = path().joinpath(filename)
    with open(filepath, "r", encoding="UTF-8") as f:
        records = {int(entry["Param0"]): entry["String"] for entry in csv.DictReader(f)}
    return records


def read_json(filename: str) -> dict:
    content = read_text(filename)
    return json.loads(content)


def write_json(filename: str, data: dict | str) -> None:
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            raise ValueError("Data is a string but not valid JSON")
    with path().joinpath(filename).open("w", encoding="utf-8") as data_file:
        json.dump(data, data_file, indent=4)


def read_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def initialized(last_sha: str, filenames: list[str]) -> bool:
    filename = "git_commit.txt"
    current_sha = read_text(filename) if is_file(filename) else None
    return last_sha == current_sha and all(is_file(f) for f in filenames)


def get_config(filename: str | Path) -> dict:
    if isinstance(filename, str) and not filename.endswith(".toml"):
        project_path = path().parent
        filename = Path(filename).with_suffix(".toml")
        filename = project_path / filename
    else:
        filename = Path(filename)
    if not filename.is_file():
        raise FileNotFoundError(f"Config file '{filename.as_posix()}' not found!")
    return read_toml(filename)
