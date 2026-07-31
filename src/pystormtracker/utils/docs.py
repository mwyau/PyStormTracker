import subprocess
import sys


def build_docs() -> None:
    """Helper to build documentation via sphinx-build."""
    cmd = ["sphinx-build", "-b", "html", "docs", "docs/_build/html"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error building documentation: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: sphinx-build not found. Ensure sphinx is installed.")
        sys.exit(1)


if __name__ == "__main__":
    build_docs()
