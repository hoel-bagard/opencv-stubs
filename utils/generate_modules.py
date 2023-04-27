import argparse
import inspect
import re
import subprocess
from pathlib import Path
from types import ModuleType

import cv2

from .processing_utils import process_class, process_function


def add_module(module_name: str, module: ModuleType, output_path: Path) -> None:
    stubs: list[str] = []

    # Add classes
    for (name, member) in inspect.getmembers(module):
        if inspect.isclass(member) and not name.startswith("_"):
            full_name = f"{module_name}.{name}"
            process_class(full_name, stubs)

    # Add functions
    for (name, member) in inspect.getmembers(module):
        if inspect.isfunction(member) or inspect.isbuiltin(member) and not name.startswith("_"):
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
            print(f"    Adding constant: {name}")
            stubs.append(f"{name}: int")

    # Cleanup
    stubs_str = "\n".join(stubs)
    stubs_str = re.sub(r'"""\n        \n        """', '""""""', stubs_str)
    stubs_str = re.sub(r"\n        \n", "\n\n", stubs_str)

    with output_path.open("w", encoding="utf-8") as stub_file:
        stub_file.write(stubs_str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Script to generate the stubs for the opencv classes.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-path", "-o", type=Path, default=Path("src/cv2-stubs"), help="Output path.")
    args = parser.parse_args()

    output_path: Path = args.output_path

    modules = [(name, member)
               for (name, member) in inspect.getmembers(cv2)
               if inspect.ismodule(member) and name[0] != "_"
               and name not in ("Error", "cv", "numpy", "np", "os", "importlib", "sys")]

    for (module_name, module) in modules:
        print(f"Processing module {module_name}")

        has_submodule = False
        for (name, member) in inspect.getmembers(module):
            if (inspect.ismodule(member)
                and not name.startswith("_") and name not in ("cv", "cv2", "numpy", "np", "os", "sys", "builtins")):

                print(f"    Adding submodule: {name}")
                has_submodule = True
                (output_path / "modules" / module_name).mkdir(parents=True, exist_ok=True)
                full_module_name = f"{module_name}.{name}"
                stub_path = output_path / "modules" / module_name / f"{name}.pyi"
                add_module(full_module_name, member, stub_path)

        if has_submodule:
            stub_path = output_path / "modules" / module_name / "__init__.pyi"
            add_module(module_name, module, stub_path)

        if not has_submodule:
            stub_path = output_path / "modules" / f"{module_name}.pyi"
            add_module(module_name, module, stub_path)

        if has_submodule:
            stub_path = output_path / "modules" / module_name / "__init__.pyi"
            for (name, member) in inspect.getmembers(module):
                if inspect.ismodule(member) and not name.startswith("_"):
                    subprocess.run(["sed", "-i", f"1ifrom . import {name}", stub_path], stdout=subprocess.PIPE)


    print("Finished")


if __name__ == "__main__":
    main()
