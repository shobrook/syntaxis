#########
# GLOBALS
#########


import ast


######
# MAIN
######


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
    elif isinstance(node, ast.Call): # TODO: Tokenize nested function calls
        tokenized_args = []

        for arg in node.args:
            tokenized_args.append(recursively_tokenize_node(arg, []))
        for keyword in node.keywords: # keyword(arg, value)
            tokenized_args.append(recursively_tokenize_node(keyword.value, []))

        tokens.append((tokenized_args, "args"))
        if isinstance(node.func, ast.Name): # y()
            tokens.append((node.func.id, "function"))
            return tokens[::-1]
        elif isinstance(node.func, ast.Attribute): # x.y()
            tokens.append((node.func.attr, "function"))
            return recursively_tokenize_node(node.func.value, tokens)
    elif isinstance(node, ast.Attribute): # x.y
        tokens.append((node.attr, "instance"))
        return recursively_tokenize_node(node.value, tokens)
    elif isinstance(node, ast.Subscript): # x[]
        slice = []
        if isinstance(node.slice, ast.Index):
            if isinstance(node.slice.value, ast.Str): # ["a"]
                slice.append(("\"" + node.slice.value.s + "\"", "str"))
            elif isinstance(node.slice.value, ast.Num): # [0]
                slice.append((str(node.slice.value.n), "num"))
            else: # [y()], [x], [x.y], ...
                slice = recursively_tokenize_node(node.slice.value, [])
        else: # ast.Slice (i.e. [1:2]), ast.ExtSlice (i.e. [1:2, 3])
            return tokens[::-1]

        tokens.append((slice, "subscript"))
        return recursively_tokenize_node(node.value, tokens)
    elif isinstance(node, ast.ListComp):
        return [] # TODO
    elif isinstance(node, ast.GeneratorExp):
        return [] # TODO
    elif isinstance(node, ast.DictComp):
        return [] # TODO
    elif isinstance(node, ast.SetComp):
        return [] # TODO
    elif isinstance(node, ast.Dict):
        return [] # TODO
    elif isinstance(node, ast.List):
        return [] # TODO
    elif isinstance(node, ast.Tuple):
        return [] # TODO
    elif isinstance(node, ast.Set):
        return [] # TODO
    elif isinstance(node, ast.Lambda):
        return [] # TODO
    elif isinstance(node, ast.Compare):
        return [] # TODO
    elif isinstance(node, ast.IfExp):
        return [] # TODO
    else:
        return [] # TODO

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
    matches = {
        "primary": {tier: [] for tier in range(len(types))}, # Exact context matches
        "secondary": {tier: [] for tier in range(len(types))} # Secondary context matches
    }

    for node in subtree.breadth_first():
        if has_matching_alias(node, exact_match=True):
            match_tier = "primary"
        elif has_matching_alias(node, exact_match=False):
            match_tier = "secondary"
        else:
            continue

        for type_tier, type in enumerate(types):
            if node.type == type: # Matching type
                matches[match_tier][type_tier].append(node)

    # TODO: Define rules for what should be returned

    for tier, match in matches["primary"].items():
        if match:
            return match[0]

    for tier, match in matches["secondary"].items():
        if match:
            return match[0]

    return None

def preprocess_token(tokens):
    """
    Stringifies a node token and pulls its type.

    Merges all subscript tokens into one string
    (i.e. id + slice1 + slice2 + ...)
    """

    not_last_token = len(tokens) > 1
    token_name, token_type = tokens[0]

    # TODO: Handle subscripts
    # QUESTION: Do subscripts even need to get handled in here?

    # if not_last_token and tokens[1][1] == "subscript":
    #     token_type = "subscript"
    #     for next_token in tokens[1:]:
    #         next_token_name, next_token_type = next_token
    #         if next_token_type != "subscript":
    #             break
    #
    #         token_name += next_token_name

    return token_name, token_type

def stringify_tokenized_nodes(tokens):
    stringified_tokens = ''
    token_stack = []
    for token in tokens:
        content, type = token

        if type == "args":
            stringified_tokens += "()"
        elif type == "subscript":
            stringified_tokens += "[%s]" % str(content)
        elif stringified_tokens:
            stringified_tokens = '.'.join([stringified_tokens, str(content)])
        else:
            stringified_tokens = str(content)

    return stringified_tokens

def get_max_lineno(subtree):
    try:
        return max(n.lineno for n in ast.walk(subtree) if hasattr(n, "lineno"))
    except ValueError:
        return None

def visitor(func):
    """
    @param node: AST node being visited.
    @param callback: node visiter handler.
    @param visit_children: flag for whether the node's children should be
    traversed or not.

    @return: AST node being visited (necessary for the ast.NodeTransformer
    base class).
    """

    def wrapper(self, node):
        if self._generate_parse_tree:
            output = func(self, node)

        if True: # visit_children:
            self.generic_visit(node)

        return node

    return wrapper

def context_manager(type):
    """
    Wrapper method around generic_visit that updates the context stack
    before traversing a subtree, and pops from the stack when the traversal
    is finished.

    @param context: name of new context.
    @param subtree: root node of subtree to traverse.
    """

    def real_context_manager(func):
        def wrapper(self, node):
            output = func(self, node)

            self._context_stack.append('#'.join([type, node.name]))
            self.generic_visit(node)
            self._context_stack.pop()

            return node

        return wrapper

    return real_context_manager

def sapling():
    def wrapper():
        pass

    return wrapper
