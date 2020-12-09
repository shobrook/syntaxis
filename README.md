<h1 align="center">
  <img width="30%" src="./logo.png" />
  <br />
</h1>

`saplings` is a static analysis tool for Python. Given a program, `saplings` will build object hierarchies for every module imported in the program. Object hierarchies are dependency trees where the root node represents a module and each child represents an attribute of its parent. These can be useful for making inferences about a module's API, mining patterns in how a module is used, and [duck typing](https://en.wikipedia.org/wiki/Duck_typing).

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

## Usage

Using `saplings` takes only two steps. First, convert your input program into an [Abstract Syntax Tree (AST)](https://en.wikipedia.org/wiki/Abstract_syntax_tree) using the `ast` module. Then, import the `Saplings` object and initialize it with the root node of the AST.

```python
import ast
from saplings import Saplings

my_program = open("path_to_your_program.py", "r").read()
program_ast = ast.parse(my_program)
my_saplings = Saplings(program_ast)
```

That's it. To access the object hierarchies, simply call the `get_trees` method in your `Saplings` object, like so:

```python
my_saplings.get_trees() # => [ObjectNode(), ObjectNode(), ..., ObjectNode()]
```

For more advanced usage of the `Saplings` object, read the docstring [here]().

### Printing an Object Hierarchy

`get_trees` returns a list of `ObjectNode`s, each representing the root node of an object hierarchy and which has the following attributes:
* **`name` _(str)_:** Name of the object
* **`is_callable` _(bool)_:** Whether the object is callable (i.e. has `__call__` defined)
* **`order` _(int)_:** Indicates the type of connection to the parent node (e.g. `0` is an attribute of the parent, `1` is an attribute of the output of the parent when called, etc.); `-1` if node is root
* **`children` _(list)_:** List of child nodes

To pretty-print a tree, simply pass its root node into the `render_tree` generator, like so:

```python
from saplings.rendering import render_tree

trees = my_saplings.get_trees()
root_node = trees[0]
for branches, node in render_tree(root_node):
  print(f"{branches}{node}")
```
```
numpy (NC, -1)
 +-- random (NC, 0)
 |   +-- randn (C, 0)
 |       +-- __sub__ (C, 1)
 |       |   +-- shape (NC, 1)
 |       |   +-- __index__ (C, 1)
 |       +-- sum (C, 1)
 +-- matmul (C, 0)
 +-- expand_dims (C, 0)
     +-- T (NC, 1)
```

Here, `NC` means indicates a non-callable node and `C` a callable node. `-1`/`0`/`1` indicate the order of the node's connection to its parent.

To create a dictionary representation of a tree, pass its root node into the `dictify_tree` function, like so:

```python
from saplings.rendering import dictify_tree

dictify_tree(root_node)
```
```python
{
  "numpy": {
    "is_callable": False,
    "order": -1,
    "children": [
      {"random": ...},
      {"matmul": ...},
      {"expand_dims": ...}
    ]
  }
}
```

### Interpreting the Object Hierarchy

Each node is an _object_ and an object can either be _callable_ (i.e. has `__call__` defined) or _non-callable_. Connections between nodes each have an _order_ –– a number which describes the relationship between a node and its parent. If a node is a 0th-order child of its parent object, then it's an attribute of that object. If it's a 1st-order child, then it's an attribute of the output of the parent object when it's called. For example:

```python
my_parent = module.my_obj

my_parent.my_attr # my_attr is a 0th-order child of my_obj
my_parent().T # my_attr is a 1st-order child of my_obj
my_parent()().T # my_attr is a 2nd-order child of my_obj
```

#### What counts as a function?

In Python, subscripts, comparisons, and binary operations are all just syntactic sugar for function calls, and are treated by `saplings` as such. Here are some common "translations:"

```python
my_obj['my_sub'] # => my_obj.__index__('my_sub')
my_obj + 10 # => my_obj.__add__(10)
my_obj == None # => my_obj.__eq__(None)
```

## Limitations

Saplings _statically_ analyzes the usage of a module in a program, meaning it doesn't actually execute any code. Instead, it traverses the program's AST and tracks "object flow," i.e. how an object is passed through a program via assignments and calls of user-defined functions and classes. Consider this example of currying:

<p align="center">
  <img width="75%" src="currying.png" />
</p>

Saplings identifies `tensor` as an attribute of `torch`, then follows the object as it's passed into `composed_func`. Because saplings has an understanding of how `composed_func` is defined, it can capture the `T` and `sum` sub-attributes.

While saplings can track object flow through many complex paths in a program, I haven't tested every edge case, and there are some situations where saplings produces inaccurate trees. Here are all the failure modes I'm aware of (and currently working on fixing):

### Data Structures

As of right now, `saplings` can't track _assignments_ to comprehensions, generator expressions, dictionaries, lists, tuples, or sets. It can, however, track object flow _inside_ these data structures. For example, consider the following:

<p align="center">
  <img width="75%" src="data_structures.png" />
</p>

Here, `mean` would not be captured and added to the `numpy` object hierarchy, but `array` would.

Notably, functions that return multiple values with one `return` statement (e.g. `return a, b, c`) are considered to return tuples, and hence won't be tracked by Saplings.

### Control Flow

Handling control flow is tricky. Tracking object flow in loops and conditionals requires making assumptions about what code actually executes. For example, consider the following:

```python
import numpy as np

for x in np.array([]):
  print(x.mean())
```

If `np.array([])` is an empty list, then the print statement, and therefore `x.mean()`, will never execute. In that situation, adding the `__index__ -> mean` subtree to `numpy -> array` would be a false positive. To handle this, `saplings` _should_ branch out and produce two possible trees for this module:

<p align="center">
  <img width="50%" src="for_loop.png" />
</p>

But as of right now, `saplings` will only produce the tree on the right –– that is, we assume the bodies of `for` loops are always executed.

#### `while` loops

`while` loops are processed under the same assumption as `for` loops –– that is, the body of the loop is assumed to execute.

#### `if`/`else` blocks

We assume the bodies of `if` blocks execute, and that `elif`/`else` blocks do not execute. That is, changes to the namespace made in `if` blocks are the only changes assumed to persist into the parent scope, whereas changes in `elif` or `else` blocks do not persist. For example, consider the following:

<p align="center">
  <img width="75%" src="if_else.png" />
</p>

Notice how our assumption can produce false negatives and positives. If it turns out `condition` is `False` and the `else` block executes, then the `sum` node would be a false positive and the exclusion of the `max` node would be a false negative. Ideally, `saplings` would branch out and produce two separate trees for this module –– one for when `if` block executes and the other for when the `else` executes, like so:

<p align="center">
  <img width="65%" src="if_else_double_trees.png" />
</p>

Our assumption applies to ternary expressions too. For example, the assignment `a = b.c if condition else b.d` is, under our assumption, equivalent to `a = b.c`.

#### `try`/`except` blocks

`try` blocks are assumed to always execute, without throwing an exception, and the `except` block is assumed not to execute. Like with `if`/`else` blocks, this assumption does not the `except` body is ignored. Object flow is still tracked inside the `except` block, but any changes made to the namespace within this block do not persist outside that scope.

#### `return`, `break`, and `continue` statements

All code underneath a `return`, `break`, or `continue` statement is assumed not to execute and will not be analyzed. This is not so much a "limitation" as it is an assumption, but it can produce some false negatives. For example, consider this:

```python
import numpy as np

for x in range(10):
  y = np.array([x])
  continue
  y.mean()
```

It may be the case that `mean` is an attribute of `np.array`, but saplings will not capture this since `y.mean()` would never be executed.

### Functions

#### Conditional return types

`saplings` can generally track module and user-defined functions, but there are some edge cases it cannot handle. For example, because module functions must be treated as black-boxes to `saplings`, conditional return types cannot be accounted for. Consider this code and the hierarchy `saplings` produces:

```python
import module

module.foo(5).bar1()
module.foo(10).bar2()
```

```
module
 +-- foo
 |   +-- bar1
 |   +-- bar2
```

However, if `module.foo` is defined as:

```python
def foo(x):
  if x <= 5:
    return ObjectA()
  else:
    return ObjectB()
```

and `ObjectB` doesn't have `bar1` as an attribute, then `saplings` will treat `bar1` and `bar2` as attributes of the same object.

#### Recursion

Saplings cannot process recursive function calls. Consider the following example:

```python
import module

def my_recursive_func(input):
  if input > 5:
    return my_recursive_func(input - 1)
  elif input > 1:
    return module.foo
  else:
    return module.bar

output = my_recursive_func(5)
output.attr()
```

We know this function returns `module.foo`, but Saplings cannot tell which base case is hit, and therefore cannot track the output. To avoid false positives, we assume this function returns nothing, and thus `attr` will not be captured and added to the object hierarchy.

#### Generators

Generators aren't processed as iterables. Instead, `saplings` ignores `yield`/`yield from` statements and treats the generator like a normal function. For example:

```python
import module

def my_generator():
  yield from module.some_items

for item in my_generator():
  print(item.name)
```

`__index__().fizz` won't be added as a subtree to `module.bar`.

#### Decorators

User-defined decorators don't actually modify functions.

#### Anonymous Functions

Assignments/Calls are ignored.

### Classes

Saplings can keep track of the state of each instance of a user-defined class. It also understands the difference between class, static, and instance methods. For example:

```python

```

#### Inheritance

If you have:
```python
import module

class MyClass(module.blah):
  def __init__(self, input):
    x = self.parent_func(input)
    self.y = x + 10
```

Then `module.blah.parent_func` will not be captured. In other words, inheritance is ignored.
TODO: Handle super()

#### Metaclasses

### Miscellaneous

#### `global` statements

#### `eval` statements, and other built-in functions

#### `nonlocals` function
