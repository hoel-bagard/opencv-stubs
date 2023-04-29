import argparse
import inspect
import re
from pathlib import Path

import cv2

from .processing_utils import process_class


def main() -> None:
    parser = argparse.ArgumentParser(description="Script to generate the stubs for the opencv classes.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-path", "-o", type=Path, default=Path("cv2_classes.pyi"), help="Output path.")
    # parser.add_argument("--one-line-docstrings", "-one", action="store_true", help="Makes docstrings fit on one line.")
    args = parser.parse_args()

    output_path: Path = args.output_path
    # one_line_docstrings: bool = args.one_line_docstrings

    stubs: list[str] = []
    for (name, member) in inspect.getmembers(cv2):
        if inspect.isclass(member):
            process_class(name, stubs)

    # Cleanup
    stubs_str = "\n".join(stubs)
    stubs_str = re.sub(r'"""\n        \n        """', '""""""', stubs_str)
    stubs_str = re.sub(r"\n        \n", "\n\n", stubs_str)

    with output_path.open("w", encoding="utf-8") as stub_file:
        stub_file.write(stubs_str)
    print("Finished")



if __name__ == "__main__":
    main()
