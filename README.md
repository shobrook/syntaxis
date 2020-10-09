<h1 align="center">
  <img width="30%" src="./logo.png" />
  <br />
</h1>

`saplings` is a static analysis tool for Python. Given a Python program, `saplings` builds object hierarchies for imported modules based on their usage in the program. If the input program used every construct in a module, then this tree would represent its entire API.<!--Too strong of a statement?-->

<img src="./demo.gif" />

<!-- This library also provides simple methods for calculating software metrics, including:

- Halstead Metrics (Volume, Difficulty, Estimated Length, etc.)
- Afferent and Efferent Couplings
- Abstractness
- Instability
- Function Rankings
- # of Partial, Recursive, and Curried Functions
- # of Lines of Code
- Cyclomatic Complexity (COMING SOON)
- Maintainability Index (COMING SOON) -->

## Installation

> Requires Python 3.X.

You can install `saplings` with `pip`:

```bash
$ pip install saplings
```

## Getting Started

Import the `Saplings` object and initialize it with the root node of an AST (you'll need the `ast` module for this).

```python
import ast
from saplings import Saplings

my_program = open("path_to_your_program.py", "r").read()
program_ast = ast.parse(my_program)
my_saplings = Saplings(program_ast)
```

Initializing `Saplings` does ...

Here's how to print out the d-trees, save them as JSON, etc. ...

How to interpret the d-tree (e.g. how to interpret __index__, ())
- Function calls include subscripts (__index__), comparisons, and binary operations

## Limitations

`saplings` can fail to track object flow in some situations, and can sometimes produce inaccurate trees. Here are all the failure modes I am aware of (and currently working on fixing):

### Data Structures

As of right now, saplings can't track assignments to comprehensions, generator expressions, dictionaries, lists, tuples, or sets. For example, consider the following:

```python
import module

my_var = [module.foo(1), module.foo(2), module.foo(3)]
my_var[0].bar()
```

Here, `bar` would not be captured and added to the d-tree for `module`. However, this isn't to say `saplings` doesn't capture module constructs used _inside_ data structures. In the example above, a node for `foo` would still be created and appended to the d-tree.

### Functions

```python
import module

module.foo(5).attr
module.foo(10).attr.bar()
```

`module -> foo -> attr -> bar`
Even though `module.foo()` could output a different object (that happens to have an attribute named `attr`) when the input is `10`.

#### Recursion

#### Currying

#### Closures

#### Generators

### Control Flow

Handling control flow is tricky. Tracking module objects in `if`, `try`/`except`, `for`, `while`, and `with` blocks requires making assumptions about what code actually executes. For example, consider the following program:

```python
import module

for item in module.items():
  print(item.foo())
```

If `module.items()` returns an empty list, then `item.foo` will never be called. In that situation, adding the `__index__ -> foo` subtree to `module -> items` would be a false positive. To handle this, saplings _should_ branch out and produce two possible trees for this module (see issue #X): `module -> items` and `module -> items -> __index__ -> foo`. But as of right now, saplings will only produce the latter tree –– that is, we assume the bodies of `for` loops are always executed.

#### `while` loops

`while` loops are processed under the same assumption as `for` loops –– that is, the body of the loop is assumed to execute.

#### `if`/`else` blocks

We assume the bodies of `if` blocks execute, and that `elif`/`else` blocks do not execute. That is, changes to the namespace made in the first `if` block in a series of `if`s, `elif`s, and/or `else`s are the only changes assumed to persist. For example, consider this code and the d-tree saplings produces:

```python
import module

var1 = module.foo()
var2 = module.bar()

if condition:
  var1 = module.attr1
  var2 = None
else:
  var1 = None
  var2 = module.attr2

var1.fizzle()
var2.shizzle()
```

```
module
 +-- foo
 +-- bar
 +-- attr1
 |   +-- fizzle
 +-- attr2
```

Notice that our assumption can produce false positives and negatives. If it turns out `condition` is `False` and the `else` block executes, then `attr1 -> fizzle` would be a false positive and the exclusion of `attr2 -> shizzle` would be a false negative. Ideally, saplings would branch out and produce two separate trees for this module –– one for when the `if` block executes and the other for when the `else` executes.

Our assumption applies to ternary expressions too. For example, the assignment `var = module.foo() if condition else module.bar()` is, under our assumption, equivalent to `var = module.foo()`.

#### `try`/`except` blocks

`try` blocks are assumed to always execute, and the `except` block is assumed not to execute. Like with `if`/`else` blocks, this assumption does not mean the `except` body is ignored. <!--Object flow-->Module usage is still tracked inside the `except` block, but changes to the namespace do not persist outside the block.

#### `continue` and `break` statements

If `continue` or `break` is used inside a loop, we assume the code underneath the statement does not execute. For example, consider this program:

```python
import module

for i in range(10):
  continue
  module.foo()
```

Here, `foo` would not be captured and added to the `module` d-tree, which _may_ be a false negative, although it isn't necessarily. Notably, our assumption will not produce any false positives.

### Miscellaneous

Code in `exec` statements is ignored. `nonlocals` is ignored.

#### `Saplings.analyze_module_usage(conservative=False, namespace={})`

Tracks the state of the namespace as we traverse through the AST of the program.

This method uses some basic type inference to track the usage of an imported module. It then maps out all the used attributes of the module (functions, instances, types) as a dependency tree and assigns a frequency value to each node. In theory, if your program used every construct in a module, then this tree would represent its entire API. The tree is returned as a dictionary with the following structure:

<!--Give example of dictionary structure side-by-side with tree visualization-->

**Arguments**

- `conservative: Bool`: Because Python is a dynamic language, multiple module trees may be extracted for a single module, each corresponding to a possible execution path through the program. If set to `True`, GAT will only return a tree derived from code that is _sure_ to execute. If `False`, then multiple trees might be returned for each module. <!--Give example-->
  - Maximum # of groups of dep. trees = Cyclomatic Complexity
- `namespace: Dict`: Lets you factor in the attributes of imported local modules.

**Limitations**

Notably, `analyze_module_usage` is not a type inference algorithm. For example, consider the following program:

```python
import torch

my_tensor = torch.tensor([1,2,3])
my_ndarray = my_tensor.numpy()
print(my_ndarray.dtype)
```

<!--Give visualization of tree-->

`analyze_module_usage` will _not_ tell you that calling `.numpy()` on a `tensor` returns an object of type `numpy.ndarray`. It will, however, tell you that `.numpy()` returns an object with a `dtype` attribute. <!--This is similar in principle to duck typing, where the attributes of an object are what define its type.-->

Another limitation is the lack of any formal proof that `analyze_module_usage` works correctly for every possible usage of a module. While it is able to track object flow in many complex situations, I haven't tested every edge case, and there are already some known failure modes, listed in the "Limitations" section.
