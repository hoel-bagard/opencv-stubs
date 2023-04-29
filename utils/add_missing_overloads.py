"""Script to add the overload decorator to any method/function missing it."""
import argparse
import subprocess
from pathlib import Path


def get_line() -> str | None:
    pyright_result = subprocess.run(["pyright", "."], stdout=subprocess.PIPE).stdout.decode().splitlines()
    for line in pyright_result:
        if "is obscured by a declaration of the same name" not in line:
            continue
        return line
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=("Script to add the overload decorator to any method/function "
                                                  "missing it."),
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()  # pyright: ignore[reportUnusedVariable]

    print("Running pyright...")
    nb_to_fix = len([1
                     for line in subprocess.run(["pyright", "."], stdout=subprocess.PIPE).stdout.decode().splitlines()
                     if "is obscured by a declaration of the same name" in line])
    print(f"Estimated number of overloads to add: {nb_to_fix}")
    nb_fixed = 1
    while (line := get_line()):
        print(f"Overloads added: {nb_fixed}/{nb_to_fix}")

        file_path = Path(line.split(":", maxsplit=1)[0].lstrip())
        write_line = line.split(":")[1]
        subprocess.run(["sed", "-i", f"{write_line}i\    @overload", file_path],
                       stdout=subprocess.PIPE)

        nb_fixed += 1

    print("Finished")


if __name__ == "__main__":
    main()
