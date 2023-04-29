import argparse
import inspect
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Script to generate the stubs for the opencv classes.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-path", "-o", type=Path, default=Path("cv2_constants.pyi"), help="Output path.")
    parser.add_argument("--merge-with", "-m", type=Path, default=None,
                        help="Path to a file with some constants already defined.")
    args = parser.parse_args()

    output_path: Path = args.output_path
    merge_with_path: Path | None = args.merge_with

    existing_constants: set[str] = set()
    if merge_with_path is not None:
        with merge_with_path.open("r", encoding="utf-8") as constants_file:
            for line in constants_file:
                if ": int" in line:
                    existing_constants.add(line.split(":")[0])

    stubs: list[str] = ["\n\n\n"]
    for (name, member) in inspect.getmembers(cv2):
        if (not inspect.isfunction(member)
            and not inspect.isclass(member)
            and not inspect.isbuiltin(member)
            and not inspect.ismodule(member)
            and not name.startswith("_")
            and name not in existing_constants):
            stubs.append(f"{name}: int\n")

    with output_path.open("a", encoding="utf-8") as stub_file:
        stub_file.writelines(stubs)
    print("Finished")


if __name__ == "__main__":
    main()
