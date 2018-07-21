try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from codecs import open

setup(
    name="saplings",
    description="Library of generic algorithms for working with abstract syntax trees",
    version="v1.0.2",
    packages=["saplings"],
    python_requires=">=3",
    url="https://github.com/shobrook/saplings",
    author="shobrook",
    author_email="shobrookj@gmail.com",
    keywords=["saplings", "abstract-syntax-trees", "ast", "trees", "tree-algorithms"],
    license="MIT"
)
