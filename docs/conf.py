# Configuration file for the Sphinx documentation builder.

from datetime import datetime
from importlib.util import find_spec
from pathlib import Path

# -- Project information -----------------------------------------------------
project = "PyStormTracker"
copyright = f"{datetime.now().year}, Albert M. W. Yau"
author = "Albert M. W. Yau"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

_has_mermaid = find_spec("sphinxcontrib.mermaid") is not None
if _has_mermaid:
    extensions.append("sphinxcontrib.mermaid")

    # Use GitHub-native ```mermaid fences in Markdown and interpret the same
    # syntax as the Mermaid directive in Sphinx/Read the Docs.
    myst_fence_as_directive = ["mermaid"]
    mermaid_params = [
        "-p",
        str(Path(__file__).with_name("puppeteer-config.json")),
    ]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- MyST Parser configuration -----------------------------------------------
myst_heading_anchors = 3
myst_enable_extensions = [
    "dollarmath",
    "attrs_inline",
]
