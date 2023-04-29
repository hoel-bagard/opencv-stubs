"""Script to set every undefined type as an alias of Any."""
import argparse
import subprocess
from collections import defaultdict
from pathlib import Path


def main() -> None:
    argparse.ArgumentParser(description="Script to set every undefined type as an alias of Any.",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    print("Running pyright")
    pyright_result = subprocess.run(["pyright", "."], stdout=subprocess.PIPE).stdout.decode().splitlines()

    aliases_to_add: dict[Path, set[str]] = defaultdict(set)
    for line in pyright_result:
        if "is not defined (reportUndefinedVariable)" not in line:
            continue
        file_path = Path(line.split(":", maxsplit=1)[0].lstrip())
        alias_to_add = line.split('"')[1]
        aliases_to_add[file_path].add(alias_to_add)

    print("Adding the following aliases:")
    for path, names in aliases_to_add.items():
        print(f"{path}: {names}")

    for path, names in aliases_to_add.items():
        print(f"Adding aliases to file {path.name}")
        with path.open("r", encoding="utf-8") as stub_file:
            write_line = 0
            while "from" in (line := stub_file.readline().strip()) or "import" in line or line == "":
                write_line += 1
                continue

        write_line = max(1, write_line)
        for name in names:
            subprocess.run(["sed", "-i", f"{write_line}i{name}: TypeAlias = Any\n", path], stdout=subprocess.PIPE)
        subprocess.run(["sed", "-i", f"{write_line}i\n", path], stdout=subprocess.PIPE)

    print("Finished")


if __name__ == "__main__":
    main()
