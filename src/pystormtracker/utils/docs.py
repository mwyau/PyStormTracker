import logging
import subprocess
import sys

LOGGER = logging.getLogger(__name__)


def build_docs() -> None:
    """Helper to build documentation via sphinx-build."""
    cmd = ["sphinx-build", "-b", "html", "docs", "docs/_build/html"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        LOGGER.error("Error building documentation: %s", e)
        sys.exit(1)
    except FileNotFoundError:
        LOGGER.error("sphinx-build not found; ensure Sphinx is installed")
        sys.exit(1)


if __name__ == "__main__":
    build_docs()
