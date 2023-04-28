import re
import subprocess


def process_function_signature(method_signature: str) -> str:
    method_signature = process_default_args(method_signature)
    method_signature = process_tuple_return(method_signature)
    return method_signature


def process_method_signature(method_signature: str) -> str:
    method_signature = process_default_args(method_signature)
    method_signature = process_tuple_return(method_signature)
    method_signature = add_self(method_signature)
    return method_signature


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


def add_self(method_signature: str) -> str:
    """Function to add self as the first argument of a method.

    Examples:
        >>>s = "writeComment(comment, append = ...) -> None:"
        >>>add_self(s)
        "writeComment(self, comment, append = ...) -> None:"

        >>>s = "getIntParam() -> retval:"
        >>>add_self(s)
        "getIntParam(self) -> retval:"
    """
    if len(method_signature) == 0:
        return "DELETE ME!"
    parts = method_signature.split("(", maxsplit=1)
    if parts[-1][0] != ")":
        parts[-1] = "self, " + parts[-1]
    else:
        parts[-1] = "self" + parts[-1]
    return "(".join(parts)


def process_class(name: str, stubs: list[str]) -> None:
    """Adds the methods of `name` to the stubs."""
    print(f"    Adding class: {name}")
    result = subprocess.run(["python", "-m", "pydoc", f"cv2.{name}"], stdout=subprocess.PIPE)
    help_text = result.stdout.split(b"\n\n", maxsplit=1)[1].decode()
    help_text_lines = help_text.splitlines()

    class_path, class_signature = help_text_lines[0].split(" = ")

    # Not in root module
    if len(class_path[4:].split("_")) != 1 and "." not in class_path[4:]:
        stubs.append("")
        stubs.append(f"{class_path[4:]} = {class_path[4:].split('_')[0]}.{class_signature.split(' ')[1].split('(')[0]}")
        stubs.append("")
        return

    stubs.append("")  # New line
    stubs.append(class_signature + ":")  # class definition

    line_idx = 1
    found_at_least_one_method = False
    while line_idx < len(help_text_lines):
        if "(...)" in help_text_lines[line_idx]:
            found_at_least_one_method = True
            # Add function signature.
            line_idx += 1
            stubs.append(f"    def {process_method_signature(help_text_lines[line_idx].replace(' | ', '', 1))}:")
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
                    stubs.append(f"    def {process_method_signature(help_text_lines[line_idx][8:])}:")
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


def process_function(name: str, stubs: list[str]) -> None:
    print(f"    Adding function: {name}")
    result = subprocess.run(["python", "-m", "pydoc", f"cv2.{name}"], stdout=subprocess.PIPE)
    help_text_lines = result.stdout.decode().splitlines()

    if not len(help_text_lines) >= 4:
        return

    function_stubs: list[str] = []
    function_stubs.append("")
    function_stubs.append(f"def {process_function_signature(help_text_lines[3].lstrip())}:")
    function_stubs.append('    """')

    # Pre-process all the lines to make it easier to handle newlines later.
    for i in range(4, len(help_text_lines)):
        help_text_lines[i] = help_text_lines[i].replace("    .    * ", "")
        help_text_lines[i] = help_text_lines[i].replace("    .    *", "")
        help_text_lines[i] = help_text_lines[i].replace("    .   ", "")

    for line in help_text_lines[4:]:
        if re.match(r"    [a-zA-Z].*", line):
            # Remove any blank lines at the end of the docstring.
            while function_stubs[-1].strip() == "":
                function_stubs.pop()
            function_stubs.append('    """')
            function_stubs.append("")
            function_stubs.append("@overload")
            if function_stubs[1] != "@overload":
                function_stubs.insert(1, "@overload")
            function_stubs.append(f"def {process_function_signature(help_text_lines[3].lstrip())}:")
            function_stubs.append('    """')
        else:
            if line == "":
                function_stubs.append("")
            else:
                function_stubs.append("    " + line)

    # Remove any blank lines at the end of the docstring.
    while function_stubs[-1].strip() == "":
        function_stubs.pop()

    function_stubs.append('    """')

    stubs.extend(function_stubs)
