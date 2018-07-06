# saplings

`saplings` is a Python library for analyzing and manipulating abstract syntax trees. It holds a collection of algorithms (saplings) that work with Python's built-in [ast](https://docs.python.org/3/library/ast.html) module. The primary features are:
* Traversals: Search for nodes, generate frequency maps for specific nodes, and apply transformations to specific nodes
* Analyses: Perform basic type-inference (powered by the [MyPy TypeChecker](https://github.com/python/mypy/wiki/Type-Checker)) and generate a frequency map for all imported packages and their features

## Install

Compiled binaries are available for [every release,](https://github.com/shobrook/saplings/releases) and you can also install `saplings` with pip:

`$ pip install saplings`

Requires Python 3.0 or higher.

## Features

## Contributing

If you've written an algorithm related to ASTs that isn't included in this library, feel free to make a contribution! Just fork the repo, make your changes, and then submit a pull request. If you do contribute, please try to adhere to the existing style.

If you've discovered a bug or have a feature request, create an [issue](https://github.com/shobrook/saplings/issues/new) and I'll take care of it!
