# saplings

`saplings` is a simple library for searching, analyzing, and transforming [abstract syntax trees.](https://en.wikipedia.org/wiki/Abstract_syntax_tree) It provides some generic algorithms (saplings) that work with Python's built-in [ast](https://docs.python.org/3/library/ast.html) module. Each sapling belongs to one of two categories:
* __Traversals:__ 
  * Searching for specific nodes by class, id, attributes, or scope
  * Generating frequency maps for specific nodes
  * Pretty-printing the tree to stdout or a text file
* __Analyses:__ 
  * Applying custom transformations to the tree
  * Generating a `PackageTree` object that represents and tracks a program's usage of a particular Python package
  * Performing basic type inference

## Install

Compiled binaries are available for [every release,](https://github.com/shobrook/saplings/releases) and you can also install `saplings` with pip:

`$ pip install saplings`

Requires Python 3.0 or higher.

## Features

To get started, import the `Harvester` object from `saplings` and initialize it with the root AST node for your program. The `Harvester` object holds your program's AST and "saplings" (instance methods) for traversing and analyzing that tree. 

For example, let's say you wanted to count the number of `while` and `for` loops used in your program:

```python
import ast
from saplings import Harvester

your_ast = ast.parse("path/to/your_file.py")
your_harvester = Harvester(your_ast)

loop_counts = your_harvester.get_freq_map(nodes=[ast.While, ast.For])
print(loop_counts)
# {ast.While: 19, ast.For: 12}
```

### `Harvester` Object

`Harvester` holds the root node of your `AST` and inherits from `ast.NodeVisitor`. Every traversal is depth-first by default. The following instance methods are available:

#### `search_by_type(node_types, skip=[])`

Returns a list of nodes belonging to a particular class (or classes). The `node_type` parameter is the list of node classes to retrieve, and the `skip` parameter is a list of subtrees (node classes) to skip in the traversal.

For example, the following code retrieves all list, set, and dictionary comprehension nodes from your AST, but skips all nodes contained in functions.

```python
comprehensions = your_harvester.search_by_type(
     node_types=[ast.ListComp, ast.SetComp, ast.DictComp], 
     skip=[ast.FunctionDef]
)
print(comprehensions)
# [<_ast.ListComp object at 0x102a8dd30>, <_ast.ListComp object at 0x102b1a128>, <_ast.DictComp object at 0x102c2b142>]
```

#### `search_by_id(node_id, skip=[])`

Coming soon! 

#### `get_freq_map(node_types=[], node_ids=[])`

Returns a dictionary mapping node types (or ids) to their no. of occurences in the AST.


<!--`nodes` is an optional parameter. By default, `get_freq_maps()` will return a dictionary of literals, comprehensions, etc.-->

#### `pretty_print()`

#### `transform()`

The ability to pass in a transform function and apply it to certain nodes is coming soon!

#### `get_type(nodes)`

Basic type inference coming soon!

#### `get_pkg_tree(module_names=[])`

Documentation coming soon!
<!--(See below for more details)-->

### `PackageTree` Object

Documentation coming soon!

#### `flatten()`

Documentation coming soon!
<!--- flatten() instance method-->

## Contributing

If you've written an algorithm related to ASTs that isn't included in this library, feel free to make a contribution! Just fork the repo, make your changes, and then submit a pull request. If you do contribute, please try to adhere to the existing style. <!--Give actual instructions for where in the file you should contribute-->

If you've discovered a bug or have a feature request, create an [issue](https://github.com/shobrook/saplings/issues/new) and I'll take care of it!
