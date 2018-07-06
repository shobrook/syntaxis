# saplings

`saplings` is a Python library for analyzing and manipulating abstract syntax trees. It holds a collection of algorithms (saplings) that work with Python's built-in [ast](https://docs.python.org/3/library/ast.html) module. The primary features are:
* __Traversals:__ Search for nodes, generate frequency maps for specific nodes, and apply custom transformations/extensions to the tree
* __Analyses:__ Perform basic type-inference (powered by the [MyPy TypeChecker](https://github.com/python/mypy/wiki/Type-Checker)) and generate frequency maps for all imported packages and their features

## Install

Compiled binaries are available for [every release,](https://github.com/shobrook/saplings/releases) and you can also install `saplings` with pip:

`$ pip install saplings`

Requires Python 3.0 or higher.

## Features

<!--saplings essentially acts as an enhancement to the standard AST object-->
To get started with `saplings`, import the `Seed` object and pass in a parsed AST. 

```python
from saplings import Seed

your_ast = Seed("/path/to/your/python/file") # Method 1
your_ast = Seed("x = 9\n y = x*x") # Method 2
your_ast = Seed(other_ast) # Method 3
```
Usage: Create your AST and then pass it into a Saplings() object. That object will have a bunch of methods you can call, like .query(node) or .freq_map("literals"). Inherits all of the methods for a normal ast, like .walk()

Traversals (query, manipulate, annotate, and generate frequency maps). Analyses (module tree, type-inference)

Pass in transformation functions when making a search.

## Contributing

If you've written an algorithm related to ASTs that isn't included in this library, feel free to make a contribution! Just fork the repo, make your changes, and then submit a pull request. If you do contribute, please try to adhere to the existing style. <!--Give actual instructions for where in the file you should contribute-->

If you've discovered a bug or have a feature request, create an [issue](https://github.com/shobrook/saplings/issues/new) and I'll take care of it!
