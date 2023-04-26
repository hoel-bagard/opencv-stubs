import argparse
import inspect
from pathlib import Path
import re

import cv2

from .processing_utils import process_class, process_function

def main() -> None:
    parser = argparse.ArgumentParser(description="Script to generate the stubs for the opencv classes.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-path", "-o", type=Path, default=Path("src/cv2-stubs"), help="Output path.")
    args = parser.parse_args()

    output_path: Path = args.output_path

    modules = [(name, member)
               for (name, member) in inspect.getmembers(cv2)
               if inspect.ismodule(member)
               and name != "Error"]

    for (module_name, module) in modules:
        print(f"Processing {module_name}")
        stubs: list[str] = []

        # Add classes
        for (name, member) in inspect.getmembers(module):
            if inspect.isclass(member):
                full_name = f"{module_name}.{name}"
                process_class(full_name, stubs)

        # Add functions
        for (name, member) in inspect.getmembers(module):
            if inspect.isfunction(member):
                full_name = f"{module_name}.{name}"
                process_function(full_name, stubs)

        stubs.append("")
        # Add constants
        for (name, member) in inspect.getmembers(module):
            if (not inspect.isfunction(member)
                and not inspect.isclass(member)
                and not inspect.isbuiltin(member)
                and not inspect.ismodule(member)
                and not name.startswith("_")):
                stubs.append(f"{name}: int")

        # Cleanup
        stubs_str = "\n".join(stubs)
        stubs_str = re.sub(r'"""\n        \n        """', '""""""', stubs_str)
        stubs_str = re.sub(r"\n        \n", "\n\n", stubs_str)

        with (output_path / "modules" / f"{module_name}.pyi").open("w", encoding="utf-8") as stub_file:
            stub_file.write(stubs_str)
    print("Finished")



if __name__ == "__main__":
    main()
