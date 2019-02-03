#########
# GLOBALS
#########


import ast
from collections import defaultdict


######################
# METRICS.PY UTILITIES
######################


def dispatch(func):
    def wrapper(self, node):
        results = func(self, node)

        types = {
            ast.Num: 'n',
            ast.Name: "id",
            ast.Attribute: "attr"
        }

        self.operators += results[0]
        self.operands += results[1]
        self.operators_seen.update(results[2])
        for operand in results[3]:
            new_operand = getattr(operand, types.get(type(operand), ''), operand)
            self.operands_seen.add((self.context, new_operand))

        self.generic_visit(node) # super().generic_visit(node)

    return wrapper


#####################
# FOREST.PY UTILITIES
#####################


class Node(object):
    def __init__(self, id, type="instance", context='', alias='', children=[]):
        """
        Parse tree node constructor. Each node represents a feature in a
        package's API. A feature is defined as an object, function, or variable
        that can only be used by importing the package.

        @param id: original identifier for the node.
        @param type: type of node (either module, instance, or implicit).
        @param context: scope in which the node is defined.
        @param alias: alias of the node when it's first defined.
        @param children: connected sub-nodes.
        """

        self.id, self.type = str(id), type
        self.children = []
        self.aliases = defaultdict(lambda: set())
        self.dead_aliases = defaultdict(lambda: set()) # TODO: Explain this
        self.count = 1

        if alias: self.add_alias(context, alias)
        for child in children:
            self.add_child(child)

    def __repr__(self):
        return self.id

    def __iter__(self):
        return iter(self.children)

    def __eq__(self, node):
        if isinstance(node, type(self)):
            return self.id == node.id and self.type == node.type

        return False

    def __ne__(self, node):
        return not self.__eq__(node)

    ## Instance Methods ##

    def increment_count(self):
        self.count += 1

    def add_alias(self, context, alias):
        self.aliases[context] |= {alias}

    def del_alias(self, context, alias):
        if context in self.aliases: # Alias exists in context
            self.aliases[context].discard(alias)
        else: # TODO: Explain this
            self.dead_aliases[context] |= {alias}

    def add_child(self, node):
        for child in self.children:
            if child == node: # Child already exists; update aliases
                for context, aliases in node.aliases.items():
                    self.aliases[context] |= aliases

                return

        self.children.append(node)

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

    def flatten(self):
        paths = [([self.id], self.type)]

        for node in self.children:
            useable_paths = node.flatten()
            for path in useable_paths:
                paths.append(([self.id] + path[0], path[1]))

        return paths

    def to_dict(self, debug=True):
        default = {
            "id": self.id,
            "type": self.type,
            "count": self.count
        }

        listify_aliases = lambda alias_dict: {
            ctx: list(aliases)
        for ctx, aliases in alias_dict.items() if aliases} if debug else None

        aliases = listify_aliases(self.aliases)
        dead_aliases = listify_aliases(self.dead_aliases)
        children = [child.to_dict(debug) for child in self.children]

        if children and aliases:
            return {**default, **{
                "aliases": aliases,
                "dead_aliases": dead_aliases,
                "children": children
            }}
        elif children and not aliases:
            return {**default, **{"children": children}}
        elif not children and aliases:
            return {**default, **{
                "aliases": aliases,
                "dead_aliases": dead_aliases
            }}

        return default

def visit_body_nodes(self, nodes):
    for node in nodes:
        try:
            node_name = type(node).__name__
            custom_visitor = getattr(self, ''.join(["visit_", node_name]))
            custom_visitor(node)
        except AttributeError:
            self.generic_visit(node)

def context_handler(func):
    """
    Decorator.
    Wrapper method around generic_visit that updates the context stack
    before traversing a subtree, and pops from the stack when the traversal
    is finished.
    """

    def wrapper(self, node):
        new_ctx = func.__name__.replace("visit_", '')
        adj_ctx = [new_ctx, node.name] if hasattr(node, "name") and node.name else [new_ctx]
        self._context_stack.append('#'.join(adj_ctx) + str(node.lineno))

        func(self, node)
        self.generic_visit(node)

        self._context_stack.pop()

    return wrapper

def conditional_handler(func):
    def wrapper(self, node):
        new_ctx = func.__name__.replace("visit_", '')
        self._context_stack.append(''.join([new_ctx, str(node.lineno)]))
        self._in_conditional = True

        body_nodes = [node.test] + node.body if hasattr(node, "test") else node.body
        visit_body_nodes(self, body_nodes)

        contexts = [self._context_to_string()]

        self._in_conditional = False
        self._context_stack.pop()

        additional_contexts = func(self, node)
        contexts.extend(additional_contexts)

        for context in contexts:
            for handler in self._conditional_handlers[context]:
                handler()

            del self._conditional_handlers[context]

    return wrapper

def default_handler(func):
    def wrapper(self, node):
        tokens = recursively_tokenize_node(node, [])
        nodes = self._recursively_process_tokens(tokens)

    return wrapper

def find_matching_node(subtree, id, type_pattern, context=None):
    """
    @param subtree:
    @param id:
    @param type_pattern: slash-separated string of node types to search for, in
    order of match preference.
    @param context:

    @return: parse tree node.
    """

    def exists_in_context(aliases, context):
        if context in aliases:
            if id in aliases[context]:
                return True

        return False

    def has_matching_alias(node, exact_match):
        # Alias is defined in current context
        if exists_in_context(node.aliases, context):
            return True

        # Pop down the context stack until an alias match is found
        if not exact_match and context:
            context_stack = context.split('.')
            for idx in range(1, len(context_stack)):
                adjusted_context = '.'.join(context_stack[:-idx])

                if exists_in_context(node.dead_aliases, adjusted_context):
                    return False # Node was monkey-patched in a parent context
                elif exists_in_context(node.aliases, adjusted_context):
                    return True # Node is well-defined in a parent context

        return node.id == id # No context given; ignore aliases, check IDs

    types = type_pattern.split('/')
    exact_context_matches, inexact_context_matches = [], []
    for node in subtree.breadth_first():
        if has_matching_alias(node, True):
            match_batch = exact_context_matches
        elif has_matching_alias(node, False):
            match_batch = inexact_context_matches
        else:
            continue

        for tier, type in enumerate(types):
            if node.type == type: # Matching type
                match_batch.append((tier, node))

    # TODO: Define rules for what should be returned

    get_matches = lambda batch: [m[1] for m in sorted(batch, key=lambda m: m[0])]
    matches = get_matches(exact_context_matches) + get_matches(inexact_context_matches)

    return matches[0] if matches else None

def recursively_tokenize_node(node, tokens): # DOES ITS JOB SO FAR
    """
    Takes an AST node and recursively unpacks it into it's constituent nodes.
    For example, if the input node is "x.y.z()", this function will return
    [('x', "instance"), ('y', "instance"), ('z', "function")].

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

            # if isinstance(node.slice.value, ast.Str): # ["a"]
            #     slice.append(("\"" + node.slice.value.s + "\"", "str"))
            # elif isinstance(node.slice.value, ast.Num): # [0]
            #     slice.append((str(node.slice.value.n), "num"))
            # else: # [y()], [x], [x.y], ...
            #     slice = recursively_tokenize_node(node.slice.value, [])
        else: # ast.Slice (i.e. [1:2]), ast.ExtSlice (i.e. [1:2, 3])
            return tokens[::-1] # TODO: Handle this properly

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
    elif isinstance(node, ast.Str):
        tokens.append(("\"" + node.s + "\"", "str"))
        return tokens[::-1]
    elif isinstance(node, ast.Num):
        tokens.append((str(node.n), "num"))
        return tokens[::-1]
    else:
        return [] # TODO

def stringify_tokenized_nodes(tokens):
    stringified_tokens = ''
    for token in tokens:
        content, type = token

        if type in ("call", "subscript"):
            stringified_tokens += content
        elif stringified_tokens:
            stringified_tokens += '.' + str(content)
        else:
            stringified_tokens = str(content) # Why?

    return stringified_tokens


#################
# OTHER UTILITIES
#################


# OPTIMIZE: Redundant subtree traversal
def get_max_lineno(subtree):
    try:
        return max(n.lineno for n in ast.walk(subtree) if hasattr(n, "lineno"))
    except ValueError:
        return None
