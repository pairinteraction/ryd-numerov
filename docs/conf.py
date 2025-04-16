# ruff: noqa: INP001

import ryd_numerov

# -- Project information -----------------------------------------------------

project = "ryd-numerov"
copyright = "2025, Rydberg Numerov Developers"  # noqa: A001
author = "Rydberg Numerov Developers"

version = ryd_numerov.__version__  # The short X.Y version, use via |version|
release = version  # The full version, including alpha/beta/rc tags, use via |release|

language = "en"


# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.extlinks",
    "nbsphinx",
    "sphinx.ext.inheritance_diagram",
    "sphinx_autodoc_typehints",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "_doctrees", "Thumbs.db", ".DS_Store"]  # Ignore these source files and folders
source_suffix = ".rst"
master_doc = "index"
pygments_style = "sphinx"  # syntax highlighting
todo_include_todos = False


# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"


# -- Options for jupyter notebooks -------------------------------------------------
nbsphinx_prolog = """
{% set docname = env.doc2path(env.docname, base=None).split("/")[-1] %}

.. raw:: html

    <style>
        .nbinput .prompt,
        .nboutput .prompt {
            display: none;
        }
    </style>

    <div class="admonition note">
      This page was generated from the Jupyter notebook
      <a class="reference external" href="{{ docname|e }}">{{ docname|e }}</a>.
    </div>
"""


# -- Options forautosummary -------------------------------------------
autosummary_ignore_module_all = False


# -- Options for autodoc -------------------------------------------
autodoc_class_signature = "mixed"  # combine class and __init__ doc
autodoc_typehints = "both"
