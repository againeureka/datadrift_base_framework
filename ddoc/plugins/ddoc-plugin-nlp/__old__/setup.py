
# ddoc/plugins/ddoc-plugin-nlp/setup.py
from setuptools import setup, find_packages

setup(
    name="ddoc-plugin-nlp",
    version="0.1.0",
    description="NLP transforms for ddoc",
    python_requires=">=3.9",
    packages=find_packages(include=["ddoc_plugin_nlp", "ddoc_plugin_nlp.*"]),
    install_requires=[
        "pluggy>=1.5.0",
        # add your heavy deps only if needed, e.g. "transformers"
    ],
    entry_points={
        "ddoc": [
            "ddoc_nlp = ddoc_plugin_nlp.nlp_impl:DDOCNlpPlugin",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)