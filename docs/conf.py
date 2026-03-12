# Configuration file for the Sphinx documentation builder.

from datetime import datetime

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

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_extra_path = ["IntercomparisonProtocol.pdf"]

# -- MyST Parser configuration -----------------------------------------------
myst_heading_anchors = 3
