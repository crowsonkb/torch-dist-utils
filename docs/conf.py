# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib
import inspect
from pathlib import Path
import sys

base_path = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(base_path))

import dist_utils as du


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "dist-utils"
copyright = "2023, Katherine Crowson"
author = "Katherine Crowson"
version = release = du.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.linkcode",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
pygments_style = "sphinx"

# -- Options for autodoc extension -------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "description"

# -- Options for linkcode extension ------------------------------------------

code_url = "https://github.com/crowsonkb/dist-utils/blob/main"


def linkcode_resolve(domain, info):
    if domain != "py":
        return None

    mod = importlib.import_module(info["module"])
    if "." in info["fullname"]:
        objname, attrname = info["fullname"].split(".")
        obj = getattr(mod, objname)
        try:
            # object is a method of a class
            obj = getattr(obj, attrname)
        except AttributeError:
            # object is an attribute of a class
            return None
    else:
        obj = getattr(mod, info["fullname"])

    try:
        while True:
            obj = obj.__wrapped__
    except AttributeError:
        pass

    try:
        file = inspect.getsourcefile(obj)
        lines = inspect.getsourcelines(obj)
    except TypeError:
        # e.g. object is a typing.Union
        return None
    try:
        file = Path(file).relative_to(base_path)
    except ValueError:
        return None
    start, end = lines[1], lines[1] + len(lines[0]) - 1

    return f"{code_url}/{file}#L{start}-L{end}"
