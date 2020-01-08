<h1 align="center">
  <img width="30%" src="./logo.png" />
  <br />
</h1>

`saplings` is a library for pulling data out of [Abstract Syntax Trees (ASTs)](https://en.wikipedia.org/wiki/Abstract_syntax_tree) in Python. It provides ... <!--value proposition-->

The hallmark feature of `saplings` is an algorithm that maps out the API of an imported module based on its usage in a program. <!--Because I care about the environment, I built `saplings` to represent APIs as parse trees.-->

<h1 align="center" display="flex">
  <img height="30%" src="./saplings_code_demo.png" />
  <img height="30%" src="./final_saplings_output.gif" />
</h1>

```python
import requests

def make_post_request(url, payload={}):
  response = requests.post(url, payload)
  status = response.status_code
  if status == requests.code.ok:
    print("Success!")
  elif status == requests.code.not_found:
    print("Failure!")

  return response

gh_response = make_post_request("https://api.github.com/")
print("Response text:", gh_response.text)
```

Other features include:
- Cyclomatic Complexity
- Halstead Metrics
- Maintainability Index
- Frequency Analysis
- Detection of Partial, Recursive, or Curried functions
- Afferent and Efferent Couplings (COMING SOON)
- Function Rankings (COMING SOON)

## Installation

> Requires Python 3.0 or higher.

You can also install `saplings` with pip:

`$ pip install saplings`

## Quick Start

Import the `Saplings` object and initialize it with the path of a Python file (or the root node of its AST). Think of `Saplings` as a wrapper for a program's AST that exposes methods for extracting data from that tree.

```python
from saplings import Saplings

my_saplings = Saplings("path_to_your_program.py")
```
_or_
```python
import ast
from saplings import Saplings

my_program = open("path_to_your_program.py", "r").read()
program_ast = ast.parse(my_program)
my_saplings = Saplings(program_ast)
```

This object makes the following methods public:

### `Saplings.api_transducer()`



### `Saplings.get_freq_map()`


## API

To get started, import the `Saplings` object from `saplings` and initialize it with the path of a Python file (or the root node of its AST). Think of `Saplings` as a wrapper for a program's Abstract Syntax Tree, much like `BeautifulSoup` wraps a website's DOM tree and exposes methods for doing things to that tree.

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

The `Saplings` object exposes the following algorithms for analyzing your Python program:

#### `get_api_forest()`

Saplings is a type of Finite State Transducer (FST) called a Tree Transducer (TT), which takes a tree as input (a program’s AST) and outputs a forest. Each tree in the forest represents the subset of an imported package’s API that was used in the program.

- Specifically, it’s a deterministic Top-Down Tree Transducer
- TODO: Formally define this (incl. rules). Diagram it?

```python

```

#### `find(nodes=[], skip=[]) -> List[ast.Node]`

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

#### `get_cyclomatic_complexity(method_level=False)`

#### `get_halstead_metrics(method_level=False)`

#### `get_maintainability_index(method_level=False)`

## Planting a Sapling

If you've written an AST-related algorithm that isn't in this library, feel free to make a contribution! Just fork the repo, make your changes, and then submit a pull request. If you do contribute, please follow the guidelines in the [contributing guide.](https://github.com/shobrook/saplings/blob/master/CONTRIBUTING.md)

If you've discovered a bug or have a feature request, just create an [issue](https://github.com/shobrook/saplings/issues/new) and I'll take care of it!

## Acknowledgments

The author thanks [@rubik](https://github.com/rubik/) – the Halstead and Complexity metrics were based entirely on the [radon](https://github.com/rubik/radon/) source code.
