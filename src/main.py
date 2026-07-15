from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from SpeedrEye import main  # noqa: E402


if __name__ == "__main__":
    main()

