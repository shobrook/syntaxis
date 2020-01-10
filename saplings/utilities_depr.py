#########
# GLOBALS
#########


import ast
from collections import defaultdict


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

    # def depth_first(self):
    #     yield self
    #     for node in self:
    #         yield from node.depth_first()

    def breadth_first(self):
        node_queue = [self]
        while node_queue:
            node = node_queue.pop(0)
            yield node
            for child in node.children:
                node_queue.append(child)

    def flatten(self):
        paths = [[self.id]]

        for node in self.children:
            useable_paths = node.flatten()
            for path in useable_paths:
                paths.append([self.id, path])

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

def default_handler(func):
    def wrapper(self, node):
        is_non_module_func_call = func(self, node)

        if not is_non_module_func_call:
            tokens = recursively_tokenize_node(node, [])
            nodes = self._recursively_process_tokens(tokens)

    return wrapper

def recursively_tokenize_node(node, tokens): # DOES ITS JOB SO FAR
    """
    Takes an AST node and recursively unpacks it into it's constituent nodes.
    For example, if the input node is "x.y.z()", this function will return
    [('x', "instance"), ('y', "instance"), ('z', "call")].

    @param node: AST node.
    @param tokens: token accumulator.

    @return: list of tokenized nodes (tuples in form of (identifier, type)).
    """

    if isinstance(node, ast.Name): # x
        tokens.append((node.id, "instance"))
        return tokens[::-1]
    elif isinstance(node, ast.Call):
        tokenized_args = []

        for arg in node.args:
            tokenized_args.append(recursively_tokenize_node(arg, []))
        for keyword in node.keywords: # keyword(arg, value)
            tokenized_args.append(recursively_tokenize_node(keyword.value, []))

        tokens.append((tokenized_args, "call"))
        if isinstance(node.func, ast.Name): # y()
            tokens.append((node.func.id, "instance"))
            return tokens[::-1]
        elif isinstance(node.func, ast.Attribute): # x.y()
            tokens.append((node.func.attr, "instance"))
            return recursively_tokenize_node(node.func.value, tokens)
    elif isinstance(node, ast.Attribute): # x.y
        tokens.append((node.attr, "instance"))
        return recursively_tokenize_node(node.value, tokens)
    elif isinstance(node, ast.Subscript): # x[]
        slice = []
        if isinstance(node.slice, ast.Index):
            slice = recursively_tokenize_node(node.slice.value, [])
        else: # ast.Slice (i.e. [1:2]), ast.ExtSlice (i.e. [1:2, 3])
            return tokens[::-1]

        tokens.append((slice, "subscript"))
        return recursively_tokenize_node(node.value, tokens)
    elif isinstance(node, ast.Dict):
        keys = [recursively_tokenize_node(n, []) for n in node.keys]
        vals = [recursively_tokenize_node(n, []) for n in node.values]

        tokens.append((zip(keys, vals), "hashmap"))
        return tokens[::-1]
    elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        elts = [recursively_tokenize_node(n, []) for n in node.elts]

        tokens.append((elts, "array"))
        return tokens[::-1]
    elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        token = []
        for generator in node.generators:
            token.append((recursively_tokenize_node(generator.iter, []), "iterable"))
            token.append((recursively_tokenize_node(generator.target, []), "target"))
            # token.append((recursively_tokenize_node(), "if")) # TODO: Handle ifs

        if hasattr(node, "elt"):
            token.append((recursively_tokenize_node(node.elt, []), "elt"))
        elif hasattr(node, "key") and hasattr(node, "value"):
            token.append((recursively_tokenize_node(node.key, []), "elt"))
            token.append((recursively_tokenize_node(node.value, []), "elt"))

        tokens.append((token, "comprehension"))
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
        tokens.append(("\"" + node.s + "\"", "str"))
        return tokens[::-1]
    elif isinstance(node, ast.Num):
        tokens.append((str(node.n), "num"))
        return tokens[::-1]
    else:
        return []

def stringify_tokenized_nodes(tokens, ignore_non_aliases=False):
    stringified_tokens = ''
    for content, type in tokens:
        if type == "call":
            stringified_tokens += "()"
        elif type == "subscript":
            if len(content) == 1 and content[0][1] in ("str", "num"):
                stringified_tokens += '[' + content[0][0] + ']'
            else:
                stringified_tokens += "[]"
        elif stringified_tokens:
            stringified_tokens += '.' + str(content)
        else:
            stringified_tokens = str(content) # Why?

    return stringified_tokens

#---

def visit_body_nodes(self, nodes):
    for node in nodes:
        try:
            node_name = type(node).__name__
            custom_visitor = getattr(self, ''.join(["visit_", node_name]))
            custom_visitor(node)
        except AttributeError:
            self.generic_visit(node)
