<p align="center"><img width="30%" src="./logo.png" /></p>

---

`saplings` is a Python library for searching, analyzing, and transforming [Abstract Syntax Trees (ASTs).](https://en.wikipedia.org/wiki/Abstract_syntax_tree) It provides some generic algorithms (saplings) that work with Python's built-in [ast](https://docs.python.org/3/library/ast.html) module. Each sapling belongs to one of two categories:
* __Traversals:__
  * Searching for nodes by type, id, attribute, or scope
  * Generating frequency maps for specific nodes
  * Applying custom transformations to the tree
* __Analyses:__
  * Calculating [Halstead complexity metrics](https://en.wikipedia.org/wiki/Halstead_complexity_measures) like volume and difficulty
  * Performing basic type inference (coming soon)

## Installation

Compiled binaries are available for [every release](https://github.com/shobrook/saplings/releases), and you can also install `saplings` with pip:

`$ pip install saplings`

Requires Python 3.0 or higher.

## API

To get started, import the `Harvester` object from `saplings` and initialize it with the root node of your AST. The `Harvester` object holds your AST and exposes instance methods (aka saplings) for traversing and analyzing that tree.

```python
import ast
from saplings import Harvester

my_file = open("path/to/your_file.py", 'r').read()
my_ast = ast.parse(my_file)
my_harvester = Harvester(my_ast)
```

### `Harvester` Object

`Harvester` holds the root node of your AST and inherits from `ast.NodeVisitor`. Every traversal is depth-first by default. The following saplings are currently available:

#### `find(nodes=[], skip=[]) -> List[ast.Node]`

Returns a list of matching AST nodes. `nodes` is a list of node types to retrieve and the `skip` parameter is a list of subtrees to skip in the traversal.<!--and the `all` parameter is a boolean indicating whether to return the first match or all matches.--> Both parameters are optional, and by default `find()` will return a list of all nodes contained in the AST.

```python
# Retrieves all list, set, and dictionary comprehension nodes
# from the AST, but skips nodes contained in functions

comprehensions = my_harvester.find(
     nodes=[ast.ListComp, ast.SetComp, ast.DictComp],
     skip=[ast.FunctionDef]
)
print(comprehensions)
# stdout: [<_ast.ListComp object at 0x102a8dd30>, <_ast.ListComp object at 0x102b1a128>, <_ast.DictComp object at 0x102c2b142>]
```

#### `get_freq_map(nodes=[], skip=[]) -> Dict[str, int]`

Returns a dictionary mapping node types to their frequency of occurence in the AST. `nodes` is a list of nodes to analyze and the `skip` parameter is a list of subtrees to skip in the traversal. Both are optional, and by default `get_freq_map()` will return a dictionary containing all node types in the tree and their frequencies.

```python
# Counts the number of 'while' and 'for' loops present in the AST

loop_freqs = my_harvester.get_freq_map(nodes=[ast.While, ast.For])
print(loop_freqs)
# stdout: {ast.While: 19, ast.For: 12}
```

#### `transform(nodes=[], transformer=lambda node: node) -> ast.Node`

Applies a user-defined transformation to specific nodes in the AST, and returns the root node of the modified AST. `nodes` is a list of nodes to apply the transformation to and the `transformer` parameter is a function that takes a node as input and returns a modified version. Both are optional, and by default `transform()` will return the root node of the original AST, unchanged.

```python
# Replaces the value of all "olive" strings with "apple"

def str_transformer(node):
     if node.s == "olive":
          node.s = "apple"

     return node

apple_tree = my_harvester.transform(nodes=[ast.Str], transformer=str_transformer)
```

#### `get_halstead(metric) -> float`

Calculates and returns a Halstead complexity metric for the AST. `metric` is a string specifying the name of the metric to calculate. The following metrics are supported:
* __Volume:__ describes the implementation size of the program in mathematical bits
* __Time:__ estimates how long it might take to write the program in seconds
* __Bugs:__ estimates the number of errors in the program

```python
# All possible method calls

volume = my_harvester.get_halstead("volume")
time = my_harvester.get_halstead("time")
bugs = my_harvester.get_halstead("bugs")
```

#### `get_type(nodes) -> Dict[ast.Node, str]`

Coming soon: basic type inference powered by [MyPy's TypeChecker.](https://github.com/python/mypy/blob/master/mypy/checker.py)

### Node Groups

Coming soon!

## Planting a Sapling

If you've written an AST-related algorithm that isn't in this library, feel free to make a contribution! Just fork the repo, make your changes, and then submit a pull request. If you do contribute, please follow the guidelines in the [contributing guide.](https://github.com/shobrook/saplings/blob/master/CONTRIBUTING.md)

If you've discovered a bug or have a feature request, just create an [issue](https://github.com/shobrook/saplings/issues/new) and I'll take care of it!
