# SPDX-License-Identifier: Apache-2.0
"""Sphinx documentation built from installed, dependency-light public APIs."""

import os
import subprocess
import sys
from pathlib import Path

import loraforge

DOCS = Path(__file__).resolve().parent
sys.path.insert(0, str(DOCS / "_ext"))

project = "LoRAForge"
author = "LoRAForge contributors"
copyright = "2026, LoRAForge contributors"
release = loraforge.__version__
version = release
language = "en"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx_design",
    "sphinx_copybutton",
    "experimental_api",
]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build", "_ext", "_templates", "README.md"]
myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath", "deflist"]
myst_heading_anchors = 4
autodoc_member_order = "bysource"
autodoc_typehints = "signature"
autodoc_preserve_defaults = True
autodoc_type_aliases = {"Tensor": "torch.Tensor"}
add_module_names = False
toc_object_entries_show_parents = "hide"
napoleon_use_param = True
napoleon_use_rtype = False
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

html_theme = "pydata_sphinx_theme"
html_title = "LoRAForge documentation"
html_short_title = "LoRAForge"
html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.svg"
html_static_path = ["_static"]
html_css_files = ["loraforge.css"]
templates_path = ["_templates"]
html_baseurl = "https://lishenghui.github.io/loraforge/"
html_show_sourcelink = True
html_theme_options = {
    "logo": {"text": "LoRAForge"},
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "show_nav_level": 2,
    "navigation_depth": 3,
    "show_toc_level": 2,
    "header_links_before_dropdown": 5,
    "show_prev_next": True,
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/lishenghui/loraforge",
            "icon": "fa-brands fa-github",
        }
    ],
    "footer_start": ["copyright"],
    "footer_center": ["build-status"],
    "footer_end": ["sphinx-version"],
}

try:
    source_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=DOCS, text=True, stderr=subprocess.DEVNULL
    ).strip()
except (OSError, subprocess.CalledProcessError):
    source_revision = "main"
source_revision = os.environ.get("DOCS_SOURCE_REF", source_revision)
html_context = {
    "github_user": "lishenghui",
    "github_repo": "loraforge",
    "github_version": os.environ.get("DOCS_EDIT_REF", "main"),
    "doc_path": "docs",
    "source_revision": source_revision,
}
