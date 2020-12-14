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
- Cyclomatic Complexity
- Maintainability Index -->

## Installation

> Requires Python 3.X.

You can install `saplings` with `pip`:

```bash
$ pip install saplings
```

## Usage

Using saplings takes only two steps. First, convert your input program into an [Abstract Syntax Tree (AST)](https://en.wikipedia.org/wiki/Abstract_syntax_tree) using the `ast` module. Then, import the `Saplings` object and initialize it with the root node of the AST.

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

(Here, `NC` means indicates a non-callable node and `C` a callable node. `-1`/`0`/`1` indicate the order of the node's connection to its parent).

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

my_parent.attr # attr is a 0th-order child of my_obj
my_parent().attr # attr is a 1st-order child of my_obj
my_parent()().attr # attr is a 2nd-order child of my_obj
```

#### What counts as a function?

In Python, subscripts, comparisons, and binary operations are all just syntactic sugar for function calls, and are treated by saplings as such. Here are some common "translations:"

```python
my_obj['my_sub'] # => my_obj.__index__('my_sub')
my_obj + 10 # => my_obj.__add__(10)
my_obj == None # => my_obj.__eq__(None)
```

## Limitations

Saplings _[statically analyzes](https://en.wikipedia.org/wiki/Static_program_analysis)_ the usage of a module in a program, meaning it doesn't actually execute any code. Instead, it traverses the program's AST and tracks "object flow," i.e. how an object is passed through a program via assignments and calls of user-defined functions and classes. To demonstrate this idea, consider this example of [currying](https://en.wikipedia.org/wiki/Currying):

```python
import torch

def compose(g, f):
  def h(x):
    return g(f(x))

  return h

def F(x):
  return x.T

def G(x):
  return x.sum()

composed_func = compose(F, G)
composed_func(torch.tensor())
```

<p align="center">
  <img width="25%" src="currying.png" />
</p>

Saplings identifies `tensor` as an attribute of `torch`, then follows the object as it's passed into `composed_func`. Because saplings has an understanding of how `composed_func` is defined, it can capture the `T` and `sum` sub-attributes.

While saplings can track object flow through many complex paths in a program, I haven't tested every edge case, and there are some situations where saplings produces inaccurate trees. Below is a list of all the failure modes I'm aware of (and currently working on fixing). If you discover a bug or missing feature that isn't listed here, please create an issue for it so I can add it to this list and work on fixing it.

### Data Structures

As of right now, saplings can't track _assignments_ to comprehensions, generator expressions, dictionaries, lists, tuples, or sets. It can, however, track object flow _inside_ these data structures. For example, consider the following:

<p align="center">
  <img width="75%" src="data_structures.png" />
</p>

Here, `mean` would not be captured and added to the `numpy` object hierarchy, but `array` would.

Notably, functions that return multiple values with one `return` statement (e.g. `return a, b, c`) are considered to return tuples, and hence won't be tracked by saplings. The same logic applies to variable unpacking with `*` and `**`.

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

But as of right now, saplings will only produce the tree on the right –– that is, we assume the bodies of `for` loops are always executed.

#### `while` loops

`while` loops are processed under the same assumption as `for` loops –– that is, the body of the loop is assumed to execute.

#### `if`/`else` blocks

We assume the bodies of `if` blocks execute, and that `elif`/`else` blocks do not execute. That is, changes to the namespace made in `if` blocks are the only changes assumed to persist into the parent scope, whereas changes in `elif` or `else` blocks do not persist. For example, consider the following:

<p align="center">
  <img width="75%" src="if_else.png" />
</p>

Notice how our assumption can produce false negatives and positives. If it turns out `condition` is `False` and the `else` block executes, then the `sum` node would be a false positive and the exclusion of the `max` node would be a false negative. Ideally, saplings would branch out and produce two separate trees for this module –– one for when `if` block executes and the other for when the `else` executes, like so:

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

It may be the case that `mean` is actually an attribute of `np.array`, but saplings will not capture this since `y.mean()` would never be executed.

### Functions

<!--#### Conditional return types

`saplings` can generally track module and user-defined functions, but there are some edge cases it cannot handle. For example, because module functions must be treated as black-boxes to `saplings`, conditional return types cannot be accounted for. Consider the following code and trees that saplings produces:

```python
import my_module

my_module.foo(5).attr1()
my_module.foo(10).attr2()
```

However, suppose `my_module.foo` is defined in the backend as:

```python
def foo(x):
  if x <= 5:
    return ObjectA()
  else:
    return ObjectB()
```

and `ObjectB` doesn't have `attr1` as an attribute. Then, saplings will have incorrectly treated `attr1` and `attr2` as attributes of the same object.-->

#### Recursion

Saplings cannot process recursive function calls. Consider the following example:

```python
import some_module

def my_recursive_func(input):
  if input > 5:
    return my_recursive_func(input - 1)
  elif input > 1:
    return some_module.foo
  else:
    return some_module.bar

output = my_recursive_func(5)
output.attr()
```

We know this function returns `some_module.foo`, but saplings cannot tell which base case is hit, and therefore can't track the output. To avoid false positives, we assume this function returns nothing, and thus `attr` will not be captured and added to the object hierarchy. The tree saplings produces is:

<!--Add visualization-->

#### Generators

Generators aren't processed as iterables. Instead, saplings ignores `yield`/`yield from` statements and treats the generator like a normal function. For example:

```python
import some_module

def my_generator():
  yield from some_module.some_items

for item in my_generator():
  print(item.name)
```

Here, `__index__ -> name` won't be added as a subtree to `some_module -> some_items`, and so the tree produced by saplings will look like:

<!--Add visualization-->

Notably, this limitation will only produce false negatives (i.e. failing to add objects to the hierarchy), not false positives (i.e. adding the wrong objects to the hierarchy).

#### Decorators

Saplings doesn't process the application of decorators, and thus assumes that user-defined decorators do not extend the functionality of the functions they're applied to. For example:

```python
import module

def my_decorator(func):
  def wrapper():
    output = func()
    return output.bar

  return wrapper

@my_decorator
def my_func():
  return module.foo

my_func()
```

For this, saplings _should_ produce the following tree:

<!--Tree visualization-->

But instead, saplings won't capture `bar` as an attribute of `module.foo` because it doesn't apply `my_decorator` to `my_func`. This is actually a major limitation (high on my list of things to fix) as it can produce both type I and II errors. Saplings also assumes that decorators defined by imported modules don't modify the user-defined functions they're applied to, which can cause similar problems.

#### Anonymous Functions

While the _bodies_ of anonymous functions (`lambda`s) are processed, object flow through assignments and calls of those functions is not tracked. For example:

```python
import numpy as np

transpose_and_diag = lambda x: np.diagonal(x.T)
transpose_and_diag(np.random.randn(5, 5))
```

Saplings will produce the following tree:

<!--Tree visualization-->

Notice that `T` is not captured as an attribute of `numpy.random.randn`, but `diagonal` is captured as an attribute of `numpy`. This is because the body of the `lambda` function is processed by saplings, but the assignment to `transpose_and_diag` is not recognized, and therefore the call of `transpose_and_diag` is not processed.

### Classes

Saplings can track object flow in static, class, and instance methods, getter and setter methods, class and instance variables, and classes defined within classes. Notably, it can keep track of the state of each instance of a user-defined class. Consider the following program and the tree saplings produces:

```python
import torch.nn as nn
from torch import tensor

class Perceptron(nn.Module):
  loss = None

  def __init__(self, in_channels, out_channels):
    super(NeuralNet, self).__init__()
    self.layer = nn.Linear(in_channels, out_channels)
    self.output = Perceptron.create_output_layer()

  @staticmethod
  def create_output_layer():
    def layer(x):
      return x.mean()

    return layer

  @classmethod
  def calculate_loss(cls, output, target):
    cls.loss = output - target
    return cls.loss

  def __call__(self, x):
    x = self.layer(x)
    return self.output(x)

model = Perceptron(1, 8)
output = model(tensor([10]))
loss = Perceptron.calculate_loss(output, 8)
```



However, there is some functionality with classes that has yet to be implemented.

#### Propagating changes to class variables to class instances

```python
Perceptron.loss.item() # Works
model.loss.item() # Doesn't work
```

#### Class Closures

i.e. functions that return a class

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
`super()` doesn't do shit.

#### Metaclasses

### Miscellaneous

#### `global` statements

#### `eval`, `nonlocals`, and other built-in functions
