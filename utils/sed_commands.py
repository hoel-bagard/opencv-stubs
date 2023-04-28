import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Script to run sed commands.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--path", "-p", type=Path, default=Path("src/cv2-stubs"), help="Path")
    args = parser.parse_args()

    path: Path = args.path

    for _file_path in path.rglob("*.pyi"):
        pass
        # if "retval" in file_path.read_text():
        #     subprocess.run(["sed", "-i", f"1ifrom typing import Any, TypeAlias", str(file_path)])
        #     subprocess.run(["sed", "-i", f"2iretval: TypeAlias = Any", str(file_path)])

        # if "dst" in file_path.read_text():
        #     if "from typing import Any, TypeAlias" not in file_path.read_text():
        #         subprocess.run(["sed", "-i", f"1ifrom typing import Any, TypeAlias", str(file_path)])
        #     subprocess.run(["sed", "-i", f"2idst: TypeAlias = Any", str(file_path)])

        # if "cv2" in file_path.read_text():
        #     subprocess.run(["sed", "-i", f"4icv2 = cv2_stubs", str(file_path)])


if __name__ == "__main__":
    main()
