import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Script to merge stubs with the Microsoft opencv stubs.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--microsoft_stubs_path", "-m", type=Path, default=Path("./microst_stubs.pyi"),
                        help="Path to pre-existing stubs.")
    parser.add_argument("--new_stubs_path", "-n", type=Path, default=Path("src/cv2-stubs/__init__.pyi"),
                        help="Path to my stubs.")
    args = parser.parse_args()

    microsoft_stubs: Path = args.microsoft_stubs_path
    my_stubs: Path = args.new_stubs_path

    print("Reading pre-existing stubs")
    functions: dict[str, str] = {}  # function_signature: doctring
    done_functions_names: set[str] = set()
    head: list[str] = []
    with my_stubs.open("r", encoding="utf-8") as stubs:
        stub_lines = stubs.readlines()
    for i in range(len(stub_lines)):
        if stub_lines[i][:3] == "def":
            function_name = stub_lines[i][4:].split("(")[0]
            done_functions_names.add(function_name)
            functions[stub_lines[i]] = stub_lines[i+1]

        if len(done_functions_names) == 0:
            head.append(stub_lines[i])
    print(f"Found {len(done_functions_names)} functions.")

    print("Reading Microsoft stubs")
    with microsoft_stubs.open("r", encoding="utf-8") as stubs:
        stub_lines = stubs.readlines()
    for i in range(len(stub_lines)):
        if stub_lines[i][:3] == "def" and stub_lines[i][4:].split("(")[0] not in done_functions_names:
            functions[stub_lines[i]] = stub_lines[i+1]

    print("Writing new stubs")
    with my_stubs.open("w", encoding="utf-8") as new_file:
        new_file.writelines(head[:-1])

        functions = dict(sorted(functions.items(), key=lambda item: item[0]))
        for signature, docstring in functions.items():
            new_file.write("\n")
            new_file.write(signature)
            new_file.write(docstring)
            new_file.write("    ...\n")

    print("Done")

    # Replace commands that can be used to add some typings automatically:
    # sed -i "s/src: Mat/src: npt.NDArray[TImg]/" src/cv2-stubs/__init__.pyi
    # sed -i "s/dst: Mat/dst: npt.NDArray[TImg]/" src/cv2-stubs/__init__.pyi
    # sed -i "s/\(dst: npt.NDArray\[TImg\].*\) -> typing.Any:/\1 -> npt.NDArray[TImg]:/" src/cv2-stubs/__init__.pyi
    # sed -i "s/(image: Mat/(image: npt.NDArray[TImg]/" src/cv2-stubs/__init__.pyi
    # sed -i "s/borderMode=.../borderMode: int = .../" src/cv2-stubs/__init__.pyi
    # sed -i "s/borderValue=.../borderValue: TColor  = .../" src/cv2-stubs/__init__.pyi
    # sed -i "s/src1: Mat/src1: npt.NDArray[TImg]/" src/cv2-stubs/__init__.pyi
    # sed -i "s/src2: Mat/src2: npt.NDArray[TImg]/" src/cv2-stubs/__init__.pyi
    # sed -i "s/typing.Tuple\[/tuple\[/" src/cv2-stubs/__init__.pyi
    # sed -i "s/mask: Mat/mask: TMask/g" src/cv2-stubs/__init__.pyi


if __name__ == "__main__":
    main()
