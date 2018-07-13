# saplings

`saplings` is a simple library for searching, analyzing, and transforming [Abstract Syntax Trees (ASTs).](https://en.wikipedia.org/wiki/Abstract_syntax_tree) It provides some generic algorithms (saplings) that work with Python's built-in [ast](https://docs.python.org/3/library/ast.html) module. Each sapling belongs to one of two categories:
* __Traversals:__
  * Searching for nodes by type, id, attribute, or scope
  * Generating frequency maps for specific nodes
  * Applying custom transformations to the tree
* __Analyses:__
  * Calculating [Halstead complexity metrics](https://en.wikipedia.org/wiki/Halstead_complexity_measures) like volume and difficulty
  * Generating `PackageTree` objects that represent the tree's usage of imported Python packages
  * Performing basic type inference

## Installation

Compiled binaries are available for [every release](https://github.com/shobrook/saplings/releases), and you can also install `saplings` with pip:

`$ pip install saplings`

Requires Python 3.0 or higher.

## API

To get started, import the `Harvester` object from `saplings` and initialize it with the root node of your AST. The `Harvester` object holds your AST and exposes instance methods (aka saplings) for traversing and analyzing that tree.

```python
import ast
from saplings import Harvester

your_ast = ast.parse("path/to/your_file.py")
your_harvester = Harvester(your_ast)
```

### `Harvester` Object

`Harvester` holds the root node of your AST and inherits from `ast.NodeVisitor`. Every traversal is depth-first by default. The following saplings are available:

#### `search_by_type(nodes, skip=[])`

Returns a list of nodes belonging to a particular class (or classes). `nodes` is a list of node classes to retrieve, and the `skip` parameter is a list of subtrees to skip in the traversal.

For example, the following code retrieves all list, set, and dictionary comprehension nodes from your AST, but skips all nodes contained in functions.

```python
comprehensions = your_harvester.search_by_type(
     nodes=[ast.ListComp, ast.SetComp, ast.DictComp],
     skip=[ast.FunctionDef]
)
print(comprehensions)
# stdout: [<_ast.ListComp object at 0x102a8dd30>, <_ast.ListComp object at 0x102b1a128>, <_ast.DictComp object at 0x102c2b142>]
```

#### `get_freq_map(nodes=[], skip=[])`

Returns a dictionary mapping node types to their frequency of occurence in the AST. `nodes` is a list of nodes to retrieve, and the `skip` parameter is a list of subtrees to skip in the traversal. Both are optional, and by default, `get_freq_map()` will return a dictionary containing all node types in the tree and their frequences.

For example, the following code counts the number of `while` and `for` loops used in your AST.

```python
loop_counts = your_harvester.get_freq_map(nodes=[ast.While, ast.For])
print(loop_counts)
# stdout: {ast.While: 19, ast.For: 12}
```

#### `transform(nodes, transformer=lambda node: node)`

Applies a user-defined transformation to specific nodes in the AST, and returns the root node of the modified AST. `nodes` is a list of nodes to apply the transformation to, and the `transformer` parameter is a function that takes a node as input and returns a modified version. By default, `transformer` returns the input node unchanged.

For example, the following code replaces the value of all strings in your AST with `"New String Value"`.

```python
def str_transformer(node):
     node.s = "New String Value"
     return node

uniform_str_tree = your_harvester.transform(nodes=[ast.Str], transformer=str_transformer)
```
<!--You can also chain these functions-->

#### `get_type(nodes)`

Coming soon: basic type inference powered by [MyPy's TypeChecker.](https://github.com/python/mypy/blob/master/mypy/checker.py)

#### `get_halstead_metric(metric_name)`

Calculates and returns a Halstead complexity metric for the AST. `metric_name` is a string specifying the name of the metric to calculate. The following metrics are supported:
* __Vocabulary:__ n = n<sub>1</sub> + n<sub>2</sub>
* __Length:__ N = N<sub>1</sub> + N<sub>2</sub>
* __Volume:__ V = N x log<sub>2</sub>n
* __Difficulty:__ D = (n<sub>1</sub> / 2) x (N<sub>2</sub> / n<sub>2</sub>)
* __Time:__ T = (D x V) / 18sec
* __Bugs:__ B = V / 3000

Where:
* n<sub>1</sub> = no. of distinct operators
* n<sub>2</sub> = no. of distinct operands
* N<sub>1</sub> = total no. of operators
* N<sub>2</sub> = total no. of operands

__Difficulty__ is an estimate of how difficult the program is to understand. __Time__ is an estimate of how long it might take to write the program. __Bugs__ is an estimate of the no. of errors in the program.
<!--For example,--> 

#### `get_pkg_tree(module_names=[])`

Documentation coming soon!
<!--(See below for more details)-->

### `PackageTree` Object

Documentation coming soon!

#### `flatten()`

Documentation coming soon!

#### `to_dict()`

Documentation coming soon!

## Planting a Sapling

If you've written an AST-related algorithm that isn't in this library, feel free to make a contribution! Just fork the repo, make your changes, and then submit a pull request. If you do contribute, please follow the guidelines in the [Contributing guide](https://github.com/alichtman/saplings/blob/master/CONTRIBUTING.md). <!--Give actual instructions for where in the file you should contribute-->

If you've discovered a bug or have a feature request, just create an [issue](https://github.com/shobrook/saplings/issues/new) and I'll take care of it!
