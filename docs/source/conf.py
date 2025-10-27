# Minimal Sphinx config for Breathe + Napoleon
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))  # allow importing python package if needed

project = "conehead"
extensions = [
    "breathe",
    "sphinx.ext.napoleon",  # NumPy-style Python docstrings
    "sphinx.ext.autodoc",  # if you want Python autodoc pages
]
# Where Doxygen XML will be generated (matches Doxyfile XML_OUTPUT)
breathe_projects = {"conehead": os.path.abspath("../../docs/doxygen/xml")}
breathe_default_project = "conehead"

templates_path = ["_templates"]
exclude_patterns = []
html_theme = "sphinx_rtd_theme"
