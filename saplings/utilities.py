# Standard Library
import ast
from collections import defaultdict


######################
# PARSE TREE UTILITIES
######################


class Node(object):
    def __init__(self, id, children=[]):
        """
        Parse tree node constructor. Each node represents a feature in a
        package's API. A feature is defined as an object, function, or variable
        that can only be used by importing the package.

        @param id: original identifier for the node.
        @param children: connected sub-nodes.
        """

        self.id = str(id)
        self.children = []
        self.count = 1

        for child in children:
            self.add_child(child)

    def __repr__(self):
        return self.id

    def __iter__(self):
        return iter(self.children)

    def __eq__(self, node):
        if isinstance(node, type(self)):
            return self.id == node.id

        return False

    def __ne__(self, node):
        return not self.__eq__(node)

    ## Instance Methods ##

    def increment_count(self):
        self.count += 1

    def add_child(self, node):
        for child in self.children:
            if child == node: # Child already exists
                child.increment_count()
                return child

        self.children.append(node)
        return node

    def depth_first(self):
        yield self
        for node in self:
            yield from node.depth_first()

    def breadth_first(self):
        node_queue = [self]
        while node_queue:
            node = node_queue.pop(0)
            yield node
            for child in node.children:
                node_queue.append(child)

    def paths(self):
        if not self.children:
            return [[self.id]]

        paths = []
        for child in self.children:
            for path in child.paths():
                paths.append([self.id] + path)

        return paths

    def to_dict(self, debug=False):
        default = {
            "id": self.id,
            "count": self.count
        }

        children = [child.to_dict(debug) for child in self.children]
        if children:
            return {**default, **{"children": children}}

        return default


#####################
# TOKENIZER UTILITIES
#####################


class StrToken(object):
    def __init__(self, str):
        self.str = str

    def __repr__(self):
        return self.str


class NameToken(object):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


class ArgsToken(object):
    def __init__(self, args):
        self.args = args

    def __iter__(self):
        yield from self.args

    def __repr__(self):
        return "()"


class ArgToken(object):
    def __init__(self, arg, arg_name=''):
        self.arg, self.arg_name = arg, arg_name

    def __iter__(self):
        yield from self.arg


class IndexToken(object):
    def __init__(self, slice):
        self.slice = slice

    def __repr__(self):
        if len(self.slice) == 1 and isinstance(self.slice[0], StrToken):
            return '[' + str(self.slice[0]) + ']'

        return "[]"


class DictToken(object):
    def __init__(self, keys, vals):
        self.keys, self.vals = keys, vals

    def __repr__(self):
        return "{}"


class ArrayToken(object):
    def __init__(self, elts):
        self.elts = elts

    def __repr__(self):
        return "[]"


class ComprehensionToken(object):
    def __init__(self, iterables, targets, val, key=None):
        self.iterables, self.targets = iterables, targets
        self.key, self.val = key, val
        # self.if = if

    def __repr__(self):
        return "[]"


def recursively_tokenize_node(node, tokens): # DOES ITS JOB SO FAR
    """
    Takes an AST node and recursively unpacks it into it's constituent nodes.
    For example, if the input node is "x.y.z()", this function will return
    [('x', "instance"), ('y', "instance"), ('z', "call")].

    The base tokens are "names" (instances). These are variable references. We
    want to deconstruct every input node into a series of "name" tokens.

    The following tokens contain nested tokens:
        - "args" (call)
        - "index" (subscript)
        -

    @param node: AST node.
    @param tokens: token accumulator.

    @return: list of tokenized nodes (tuples in form of (identifier, type)).
    """

    if isinstance(node, ast.Name): # x
        tokens.append(NameToken(node.id))
        return tokens[::-1]
    elif isinstance(node, ast.Call):
        tokenized_args = []

        for arg in node.args:
            arg = ArgToken(
                arg=recursively_tokenize_node(arg, []),
                arg_name=''
            )
            tokenized_args.append(arg)

        for keyword in node.keywords:
            arg = ArgToken(
                arg=recursively_tokenize_node(keyword.value, []),
                arg_name=keyword.arg
            )
            tokenized_args.append(arg)

        tokens.append(ArgsToken(tokenized_args))
        return recursively_tokenize_node(node.func, tokens)

        if isinstance(node.func, ast.Name): # y()
            tokens.append(NameToken(node.func.id))
            return tokens[::-1]
        elif isinstance(node.func, ast.Attribute): # x.y()
            tokens.append(NameToken(node.func.attr))
            return recursively_tokenize_node(node.func.value, tokens)
    elif isinstance(node, ast.Attribute): # x.y
        tokens.append(NameToken(node.attr))
        return recursively_tokenize_node(node.value, tokens)
    elif isinstance(node, ast.Subscript): # x[]
        slice = []
        if isinstance(node.slice, ast.Index):
            slice = recursively_tokenize_node(node.slice.value, [])
        else: # ast.Slice (i.e. [1:2]), ast.ExtSlice (i.e. [1:2, 3])
            slice = [] # TODO

        tokens.append(IndexToken(slice))
        return recursively_tokenize_node(node.value, tokens)
    elif isinstance(node, ast.Dict):
        keys = [recursively_tokenize_node(n, []) for n in node.keys]
        vals = [recursively_tokenize_node(n, []) for n in node.values]

        tokens.append(DictToken(keys, vals))
        return tokens[::-1]
    elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        elts = [recursively_tokenize_node(n, []) for n in node.elts]

        tokens.append(ArrayToken(elts))
        return tokens[::-1]
    elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        iterables, targets = [], []
        for generator in node.generators:
            iterables.append(recursively_tokenize_node(generator.iter, []))
            targets.append(recursively_tokenize_node(generator.target, []))

        key, val = [], []
        if hasattr(node, "elt"):
            val = recursively_tokenize_node(node.elt, [])
        elif hasattr(node, "key") and hasattr(node, "value"):
            key = recursively_tokenize_node(node.key, [])
            val = recursively_tokenize_node(node.value, [])

        tokens.append(ComprehensionToken(
            iterables=iterables,
            targets=targets,
            val=val,
            key=key
        ))
        return tokens[::-1]
    elif isinstance(node, ast.Lambda):
        return [] # TODO
    elif isinstance(node, ast.IfExp):
        return [] # TODO: Handle ternary assignments
    elif isinstance(node, ast.BinOp):
        return [] # TODO: Handle stuff like 'x + y'
    elif isinstance(node, ast.BoolOp):
        return [] # TODO: Handle stuff like 'x or y'
    elif isinstance(node, ast.Compare):
        return [] # TODO: Handle stuff like 'x > y'
    elif isinstance(node, ast.Str):
        tokens.append(StrToken("\"" + node.s + "\""))
        return tokens[::-1]
    elif isinstance(node, ast.Num):
        tokens.append(StrToken(str(node.n)))
        return tokens[::-1]
    else:
        return []


def stringify_tokenized_nodes(tokens):
    stringified_tokens = ''
    for index, token in enumerate(tokens):
        if index and isinstance(token, NameToken):
            stringified_tokens += '.' + str(token)
        else:
            stringified_tokens += str(token)

    return stringified_tokens


############
# DECORATORS
############


# def context_handler(func):
#     """
#     Decorator.
#     Wrapper method around generic_visit that updates the context stack
#     before traversing a subtree, and pops from the stack when the traversal
#     is finished.
#     """
#
#     def wrapper(self, node):
#         new_ctx = func.__name__.replace("visit_", '')
#         adj_ctx = [new_ctx, node.name] if hasattr(node, "name") and node.name else [new_ctx]
#         self._context_stack.append('#'.join(adj_ctx) + str(node.lineno))
#
#         func(self, node)
#         self.generic_visit(node)
#
#         self._context_stack.pop()
#
#     return wrapper


def reference_handler(func):
    def wrapper(self, node):
        tokens = recursively_tokenize_node(node, [])
        self._recursively_process_tokens(tokens)

    return wrapper


# def visit_body_nodes(self, nodes):
#     for node in nodes:
#         try:
#             node_name = type(node).__name__
#             custom_visitor = getattr(self, ''.join(["visit_", node_name]))
#             custom_visitor(node)
#         except AttributeError:
#             self.generic_visit(node)
