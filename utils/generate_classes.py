import argparse
import inspect
import subprocess
from pathlib import Path
import re

import cv2


def process_function_signature(function_signature: str) -> str:
    function_signature = process_default_args(function_signature)
    function_signature = process_tuple_return(function_signature)
    function_signature = add_self(function_signature)
    return function_signature


def process_default_args(function_signature: str) -> str:
    """Function to convert OpenCV's way of representing arguments with a default value to python synthax.

    Examples:
        >>>s = "imdecode(buf, flags) -> retval"
        >>>process_default_args(s)
        "imdecode(buf, flags) -> retval"

        >>>s = "compute(images, keypoints[, descriptors]) -> keypoints, descriptors"
        >>>process_default_args(s)
        "compute(images, keypoints, descriptors = ...) -> keypoints, descriptors"

        >>>s = "imreadmulti(filename[, mats[, flags]]) -> retval, mats"
        >>>process_default_args(s)
        "imreadmulti(filename, mats = ..., flags = ...) -> retval, mats""
    """
    parts = function_signature.split("[")
    for i in range(1, len(parts)-1):
        parts[i] = parts[i] + " = ..."
    if len(parts) > 1:
        parts[-1] = re.sub("]+", " = ...", parts[-1])

    signature = "".join(parts)
    signature = re.sub(r"\(, ", "(", signature)  # If the function had only arguments with default values.
    return signature


def process_tuple_return(function_signature: str) -> str:
    """Function to convert OpenCV's way of representing multiple return values to python synthax.

    Examples:
        >>>s = "computeBitmaps(img[, tb[, eb]]) -> tb, eb"
        >>>process_tuple_return(s)
        "computeBitmaps(img[, tb[, eb]]) -> tuple[tb, eb]"
    """
    parts = function_signature.split(" -> ")
    return_values = parts[-1].split(", ")
    if len(return_values) > 1:
        parts[-1] = f"tuple[{parts[-1]}]"

    return " -> ".join(parts)


def add_self(function_signature: str) -> str:
    """Function to add self as the first argument of a method.

    Examples:
        >>>s = "writeComment(comment, append = ...) -> None:"
        >>>add_self(s)
        "writeComment(self, comment, append = ...) -> None:"

        >>>s = "getIntParam() -> retval:"
        >>>add_self(s)
        "getIntParam(self) -> retval:"
    """
    parts = function_signature.split("(", maxsplit=1)
    if parts[-1][0] != ")":
        parts[-1] = "self, " + parts[-1]
    else:
        parts[-1] = "self" + parts[-1]
    return "(".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Script to generate the stubs for the opencv classes.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-path", "-o", type=Path, default=Path("cv2_classes.pyi"), help="Output path.")
    parser.add_argument("--one-line-docstrings", "-one", action="store_true", help="Makes docstrings fit on one line.")
    args = parser.parse_args()

    output_path: Path = args.output_path
    one_line_docstrings: bool = args.one_line_docstrings

    stubs: list[str] = []
    for (name, member) in inspect.getmembers(cv2):
        if inspect.isclass(member):
            result = subprocess.run(["python", "-m", "pydoc", f"cv2.{name}"], stdout=subprocess.PIPE)
            help_text = result.stdout.split(b"\n\n", maxsplit=1)[1].decode()
            # print(help_text)
            help_text_lines = help_text.splitlines()

            stubs.append("")  # New line
            stubs.append(help_text_lines[0].split(" = ")[1] + ":")  # class definition

            line_idx = 1
            found_at_least_one_method = False
            while line_idx < len(help_text_lines):
                if "(...)" in help_text_lines[line_idx]:
                    found_at_least_one_method = True
                    # Add function signature.
                    line_idx += 1
                    stubs.append(f"    def {process_function_signature(help_text_lines[line_idx].lstrip(' | '))}:")
                    line_idx += 1
                    # Add docstring.
                    stubs.append('        """')
                    while (line_idx < len(help_text_lines)
                           and "(...)" not in help_text_lines[line_idx]
                           and " |  --" not in help_text_lines[line_idx]):
                        if "." in help_text_lines[line_idx]:
                            line = help_text_lines[line_idx].split(".", maxsplit=1)[1].lstrip()
                            if line == "@overload":
                                stubs.insert(-2, 4*" " + line)
                            if line == "* @overload":
                                stubs.insert(-2, 4*" " + "@overload")
                            elif "@param" in stubs[-1] and "@param" not in line:
                                stubs[-1] = stubs[-1] + " " + line  # pyright: ignore
                            else:
                                stubs.append(8*" " + line)
                        elif re.match(r" \|      [a-zA-Z]", help_text_lines[line_idx][:9]):
                            stubs.append('        """')
                            stubs.append("")  # New line
                            stubs.append(f"    def {process_function_signature(help_text_lines[line_idx][8:])}:")
                            stubs.append('        """')
                        line_idx += 1
                    stubs.append('        """')
                    stubs.append("")  # New line
                else:
                    line_idx += 1

                if line_idx >= len(help_text_lines) or "inherited" in help_text_lines[line_idx]:
                    break
            if not found_at_least_one_method:
                stubs.append("    ...")
                stubs.append("")  # New line

    # Cleanup
    stubs_str = "\n".join(stubs)
    stubs_str = re.sub(r'"""\n        \n        """', '""""""', stubs_str)
    stubs_str = re.sub(r"\n        \n", "\n\n", stubs_str)
    if one_line_docstrings:
        # TODO
        pass

    with output_path.open("w", encoding="utf-8") as stub_file:
        stub_file.write(stubs_str)
    print("Finished")



if __name__ == "__main__":
    main()
