# Configuration file for the Sphinx documentation builder.

import posixpath
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path

from docutils import nodes
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment

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
html_extra_path = ["../LICENSE", "../CONTRIBUTING.md"]

# -- MyST Parser configuration -----------------------------------------------
myst_heading_anchors = 3
myst_enable_extensions = [
    "dollarmath",
    "attrs_inline",
]

# Root-relative Markdown links work on GitHub, while these copies keep the
# same links valid in the generated documentation.
_ROOT_FILE_LINKS = {
    "LICENSE": "LICENSE",
    "../CONTRIBUTING.md": "CONTRIBUTING.md",
}


def _resolve_root_file_link(
    app: Sphinx,
    _env: BuildEnvironment,
    node: addnodes.pending_xref,
    contnode: nodes.Element,
) -> nodes.reference | None:
    target = _ROOT_FILE_LINKS.get(node.get("reftarget"))
    if target is None:
        return None

    doc_uri = app.builder.get_target_uri(node["refdoc"])
    doc_dir = posixpath.dirname(doc_uri) or "."
    reference = nodes.reference(
        "",
        "",
        internal=True,
        refuri=posixpath.relpath(target, start=doc_dir),
    )
    reference.append(contnode)
    return reference


def setup(app: Sphinx) -> None:
    app.connect("missing-reference", _resolve_root_file_link)
