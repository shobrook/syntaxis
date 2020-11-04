<h1 align="center">
  <img width="30%" src="./logo.png" />
  <br />
</h1>

QUESTION: Are these nodes OBJECTS or ATTRIBUTES?
TODO: Replace word 'context' with 'scope'.

`saplings` is a static analysis tool for Python. Given a program, `saplings` will build object hierarchies for every module imported in the program. An object hierarchy is a tree where the root node represents the module and each child represents a descendant attribute of that module. These trees are useful for making inferences about a module's API, mining patterns in how a module is used, and [duck typing](https://en.wikipedia.org/wiki/Duck_typing).

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

Using `saplings` takes only two steps. First, convert your input program into an [Abstract Syntax Tree (AST)](https://en.wikipedia.org/wiki/Abstract_syntax_tree) –– you'll need to import the `ast` module for this. Then, import the `Saplings` object and initialize it with the root node of the AST.

```python
import ast
from saplings import Saplings

my_program = open("path_to_your_program.py", "r").read()
program_ast = ast.parse(my_program)
my_saplings = Saplings(program_ast)
```

That's it. To access the object hierarchies, use the `trees` attribute of your `Saplings` object, like so:

```python
my_saplings.trees # => [ObjectNode(), ObjectNode(), ..., ObjectNode()]
```

### Printing an Object Hierarchy

`trees` holds a list of `ObjectNode`s, each representing the root node of an object hierarchy. Each `ObjectNode` has the following attributes:
* **`name` _(str)_:** Name of the object
* **`is_callable` _(bool)_:** Whether the object was called in the code
* **`order` _(int)_:** Indicates the type of connection to the parent node (e.g. `0` is an attribute of the parent, `1` is an attribute of the output of the parent when called, etc.)
* **`children` _(list)_:** List of child nodes

TODO: Create a function for nicely printing an object hierarchy given its root node.

## Understanding the Object Hierarchy

How to interpret the d-tree (e.g. how to interpret __index__, ())
- Function calls include subscripts (__index__), comparisons, and binary operations.

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

and `ObjectB` doesn't have `bar1` as an attribute, then
`saplings` will treat `bar1` and `bar2` as attributes of the same object.

***

Everything underneath a return, break, or continue statement will not be tracked.

***

Sapligns _can_ track object flow through curried functions. For example:

```python
import module

def compose(g, f):
    def h(x):
        return g(f(x))
    return h

def func1(t):
  return t.foo

def func2(t):
  return t.bar

convert = compose(func1, func2)
convert(module.attr)
```

## Limitations

`saplings` can fail to track object flow in some situations, and can sometimes produce inaccurate trees. Here are all the failure modes I am aware of (and currently working on fixing):

### Data Structures

As of right now, `saplings` can't track assignments to comprehensions, generator expressions, dictionaries, lists, tuples, or sets. For example, consider the following:

```python
import module

my_var = [module.foo(1), module.foo(2), module.foo(3)]
my_var[0].bar()
```

Here, `bar` would not be captured and added to the hierarchy for `module`. However, `saplings` will still capture objects used _inside_ data structures. In the example above, a node for `foo` would still be created and appended to the hierarchy.

### Control Flow

Handling control flow is tricky. Tracking module objects in loops and conditionals requires making assumptions about what code actually executes. For example, consider the following program:

```python
import module

for item in module.items():
  print(item.foo())
```

If `module.items()` returns an empty list, then `item.foo` will never be called. In that situation, adding the `__index__ -> foo` subtree to `module -> items` would be a false positive. To handle this, saplings _should_ branch out and produce two possible trees for this module: `module -> items` and `module -> items -> __index__ -> foo`. But as of right now, saplings will only produce the latter tree –– that is, we assume the bodies of `for` loops are always executed.

#### `while` loops

`while` loops are processed under the same assumption as `for` loops –– that is, the body of the loop is assumed to execute.

#### `if`/`else` blocks

We assume the bodies of `if` blocks execute, and that `elif`/`else` blocks do not execute. That is, changes to the namespace made in the first `if` block in a series of `if`s, `elif`s, and/or `else`s are the only changes assumed to persist into the parent context. For example, consider this code and the d-tree saplings produces:

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

<!-- #### `continue` and `break` statements

If `continue` or `break` is used inside a loop, we assume the code underneath the statement does not execute. For example, consider this program:

```python
import module

for i in range(10):
  continue
  module.foo()
```

This isn't really a limitation, but we do ignore any module usage patterns in the code underneath `continue` statements, even if they are legitimate. Notably, our assumption will never produce any false positives. -->

### Functions

#### Unevaluated Functions

Functions aren't processed until they're called, as the object flow can depend on the types of the inputs. But if a funciton is never called, they're processed in the namespace in which they were defined, with no values for its inputs.

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

For now, generators are ignored during  

#### Decorators

#### Single-Star Arguments

#### Anonymous Functions

### Miscellaneous

Code in `exec` statements is ignored. `nonlocals` is ignored. Star and double-star arguments.

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
