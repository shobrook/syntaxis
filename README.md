<h1 align="center">
  <img width="30%" src="./logo.png" />
  <br />
</h1>

---

`saplings` is a static analysis library for Python, fit with tools for calculating various software metrics and analyzing [Abstract Syntax Trees (ASTs)](https://en.wikipedia.org/wiki/Abstract_syntax_tree). Its most notable feature is an algorithm that mines all the ways in which imported packages<!--imported package APIs--> are used in a Python program, and visualizes their usage patterns as parse trees. <!--Mention some applications of this algorithm?-->Other features include:
- Cyclomatic Complexity
- Halstead Metrics
- Maintainability Index
- AST Node Retrieval
- Frequency Analysis
- Detection of Partial, Recursive, or Curried functions
- Afferent and Efferent Couplings (COMING SOON)
- Function Rankings (COMING SOON)

<!--Think of saplings as the 'BeautifulSoup' for Python source code.-->

## Installation

Compiled binaries are available for [every release](https://github.com/shobrook/saplings/releases), and you can also install `saplings` with pip:

`$ pip install saplings`

Requires Python 3.0 or higher.

## API

To get started, import the `Saplings` object from `saplings` and initialize it with the root node of your AST (or the path of a Python file). Think of this object as the "BeautifulSoup" for Python source code.

**Initializing with a File Path**
```python
from saplings import Saplings

my_saplings = Saplings("path/to/your_program.py")
```
**Initializing with an AST Root**
```python
import ast
from saplings import Saplings

my_program = open("path/to/your_program.py", 'r').read()
program_ast = ast.parse(my_program)
my_saplings = Saplings(program_ast)
```

The `Saplings` object exposes various algorithms for analyzing your Python program.

#### `get_api_forest()`

#### `find(nodes=[]) -> List[ast.Node]`

Returns a list of matching AST nodes. `nodes` is a list of node types to retrieve and the `skip` parameter is a list of subtrees to skip in the (depth-first) traversal. Both parameters are optional, and by default `find()` will return a list of all nodes contained in the AST.

```python
# Retrieves all list, set, and dictionary comprehension nodes
# from the AST, but skips nodes contained in functions

comprehensions = my_saplings.find(
     nodes=[ast.ListComp, ast.SetComp, ast.DictComp],
     skip=[ast.FunctionDef]
)
print(comprehensions)
# stdout: [<_ast.ListComp object at 0x102a8dd30>, <_ast.ListComp object at 0x102b1a128>, <_ast.DictComp object at 0x102c2b142>]
```

#### `get_freq_map(nodes=[], built_ins=False, skip=[]) -> Dict[str, int]`

Returns a dictionary mapping node types and/or built-in functions to their frequency of occurrence in the AST. `nodes` is a list of nodes to analyze, `built_ins` is a flag for whether built-in functions should be analyzed, and the `skip` parameter is a list of subtrees to skip in the traversal. All are optional, and by default `get_freq_map()` will return a dictionary mapping all node types in the tree to their frequencies.

```python
# Counts the number of 'while' and 'for' loops present in the AST

loop_freqs = my_saplings.get_freq_map(nodes=[ast.While, ast.For])
print(loop_freqs)
# stdout: {ast.While: 19, ast.For: 12}
```

#### `get_cyclomatic_complexity()`

#### `get_halstead_metrics()`

#### `get_maintainability_index()`

## Planting a Sapling

If you've written an AST-related algorithm that isn't in this library, feel free to make a contribution! Just fork the repo, make your changes, and then submit a pull request. If you do contribute, please follow the guidelines in the [contributing guide.](https://github.com/shobrook/saplings/blob/master/CONTRIBUTING.md)

If you've discovered a bug or have a feature request, just create an [issue](https://github.com/shobrook/saplings/issues/new) and I'll take care of it!

## Acknowledgments

The author thanks [@rubik](https://github.com/rubik/) – the Halstead and Complexity metrics were based entirely on the [radon](https://github.com/rubik/radon/) source code.
