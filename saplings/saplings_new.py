#########
# GLOBALS
#########


import ast
import json
from collections import defaultdict


##################
# PARSE TREE NODES
##################


class Node(object):
    def __init__(self, id, type="instance", context='', alias='', children=[]):
        self.id, self.type = id, type
        self.children = []
        self.aliases = defaultdict(lambda: set())
        self.count = None if type == "module" else 1

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

    def change_type(self, new_type):
        self.type = new_type

    def add_alias(self, context, alias):
        self.aliases[context] |= {alias}

    def del_alias(self, context, alias):
        self.aliases[context].discard(alias)

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
        aliases = {
            ctx: list(aliases)
        for ctx, aliases in self.aliases.items() if aliases} if debug else None
        children = [child.to_dict(debug) for child in self.children]

        if children and aliases:
            return {**default, **{
                "aliases": aliases,
                "children": children
            }}
        elif children and not aliases:
            return {**default, **{"children": children}}
        elif not children and aliases:
            return {**default, **{"aliases": aliases}}

        return default

class ImplicitNode(Node):
    def __init__(self, id, order=1, context='', alias='', children=[]):
        # i.e. id = [0], for find_all()[0], and alias = x for x = find_all()[0]
        # If this node ends up being a leaf, just delete it from the tree

        super().__init__(
            id=id,
            type="implicit({})".format(order),
            context=context,
            alias=alias,
            children=children
        )
        self.order, self.count = order, 1

    ## Instance Methods ##

    def increment_count(self):
        self.count += 1


###########
# UTILITIES
###########


def recursively_tokenize_node(node, tokens):
    """
    Takes an AST node and recursively unpacks it into it's constituent nodes.
    For example, if the input node is "x.y.z()", this function will return
    [('x', "instance"), ('y', "instance"), ('z', "function")].

    @param node: AST node.
    @param tokens: token accumulator.

    @return: list of node tokens (tuples in form of (identifier, type)).
    """

    if isinstance(node, ast.Name): # x
        tokens.append((node.id, "instance"))
        return tokens[::-1]
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name): # y()
            tokens.append((node.func.id, "function"))
            return tokens[::-1]
        elif isinstance(node.func, ast.Attribute): # x.y()
            tokens.append((node.func.attr, "function"))
            return recursively_tokenize_node(node.func.value, tokens)
    elif isinstance(node, ast.Attribute): # x.y
        tokens.append((node.attr, "instance"))
        return recursively_tokenize_node(node.value, tokens)
    elif isinstance(node, ast.Subscript):
        slice = ''
        if isinstance(node.slice, ast.Index):
            if isinstance(node.slice.value, ast.Str): # ["a"]
                slice = ''.join(['\"', node.slice.value.s, '\"'])
            elif isinstance(node.slice.value, ast.Num): # [0]
                slice = node.slice.value.n
            else: # [y()], [x], [x.y], ...
                slice = recursively_tokenize_node(node.slice.value, tokens)
        elif isinstance(node.slice, ast.Slice): # [1:2]
            pass # TODO
        elif isinstance(node.slice, ast.ExtSlice): # [1:2, 3]
            pass # TODO

        tokens.append((slice, "subscript"))
        return recursively_tokenize_node(node.value, tokens)
    elif isinstance(node, ast.ListComp):
        return None # TODO
    elif isinstance(node, ast.GeneratorExp):
        return None # TODO
    elif isinstance(node, ast.DictComp):
        return None # TODO
    elif isinstance(node, ast.SetComp):
        return None # TODO
    elif isinstance(node, ast.Dict):
        return None # TODO
    elif isinstance(node, ast.List):
        return None # TODO
    elif isinstance(node, ast.Tuple):
        return None # TODO
    elif isinstance(node, ast.Set):
        return None # TODO
    elif isinstance(node, ast.Lambda):
        return None # TODO
    elif isinstance(node, ast.Compare):
        return None # TODO
    elif isinstance(node, ast.IfExp):
        return None # TODO
    else:
        return None

def find_matching_node(subtree, id, type, context=None, default_type=None):
    """
    @param subtree:
    @param id:
    @param type:
    @param context:
    @param default_type: "backup" node type to search for when nodes of the
    given type cannot be found.

    @return: parse tree node
    """

    def is_matching_node(node):
        if not context and node.id == id:
            return True
        elif context:
            if id in node.aliases[context]:
                return True

            context_path = context.split('.')
            for idx in range(1, len(context_path)):
                if id in node.aliases['.'.join(context_path[:-idx])]:
                    return True

        return False

    default_node = None
    for node in subtree.breadth_first():
        if is_matching_node(node):
            if node.type == type:
                return node
            elif node.type == default_type:
                default_node = node

    return default_node

def stringify_tokenized_node(tokenized_node):
    token_stack = []
    for idx in range(len(tokenized_node)):
        token_id = tokenized_node[idx][0]
        token_type = tokenized_node[idx][1]

        not_last_token = idx < len(tokenized_node) - 1
        if not_last_token and tokenized_node[idx + 1][1] == "subscript":
            for token in tokenized_node[idx + 1:]:
                if token[1] == "subscript":
                    token_id += token[0]

                break

        if token_type == "subscript":
            continue
        elif token_type == "instance":
            # BUG: What about x[1]()?
            if not_last_token and tokenized_node[idx + 1][1] in ("instance", "function"):
                token_id += token_id + '.'
        elif token_type == "function": # BUG: Doesn't factor in func args
            token_id += "()"

        token_stack.append(token_id)

    return ''.join(token_stack)


###########
# HARVESTER
###########


class Harvester(ast.NodeTransformer):
    def __init__(self, tree):
        if isinstance(tree, ast.Module): # Already parsed Python source
            self.tree = tree
        elif isinstance(tree, str): # Relative path to Python file
            self.tree = ast.parse(open(tree, 'r').read())
        else:
            raise Exception # TODO: Create custom exception

        self._context_stack = ["global"] # Keeps track of current context
        self._parse_trees = [] # Holds root nodes of parse trees

        self._call_table = {} # For holding user-defined funcs and classes
        self._iterator_table = {} # For holding user-defined ListComps, etc.

        self._generate_parse_tree = False

        # OPTIMIZE: Turns AST into doubly-linked AST
        for node in ast.walk(self.tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node # QUESTION: Does this lead to double counting?

        self._queried_nodes = []
        self._retrieved_nodes = []
        self._freq_map = defaultdict(lambda: 0)
        self._transformer = lambda node: node

    def generic_visit(self, node):
        super().generic_visit(node)

        node_is_queried = any(isinstance(node, n) for n in self._queried_nodes)
        if node_is_queried or not self._queried_nodes:
            self._retrieved_nodes.append(node)
            self._freq_map[type(node).__name__] += 1

            return ast.copy_location(self._transformer(node), node)

        return node

    ## Helper Methods ##

    def _update_context(self, context, subtree):
        """
        Wrapper method around generic_visit that updates the context stack
        before traversing a subtree, and pops from the stack when the traversal
        is finished.

        @param context: name of new context.
        @param subtree: root node of subtree to traverse.
        """

        self._context_stack.append(context)
        self.generic_visit(subtree)
        self._context_stack.pop()

    def _context_to_string(self):
        """
        @return: period-separated string representing the current context.
        """

        return '.'.join(ctx for ctx in self._context_stack)

    def _visiter_wrapper(self, node, callback):
        if self._generate_parse_tree:
            callback()

        self.generic_visit(node)
        return node

    def _process_module(self, module, context, exclude_root_alias=False):
        """
        @param module:
        @param context:
        @param exclude_root_alias:

        @return:
        """

        sub_modules = module.split('.')
        last_node = None

        for root in self._parse_trees:
            matching_module = find_matching_node(
                subtree=root,
                id=sub_modules[0],
                type="module"
            )

            if matching_module:
                last_node = matching_module
                break

        if not last_node:
            last_node = Node(
                id=sub_modules[0],
                type="module",
                context=context,
                alias=sub_modules[0] if not exclude_root_alias else ''
            )
            self._parse_trees.append(last_node)

        for sub_module in sub_modules[1:]:
            matching_sub_module = find_matching_node(
                subtree=last_node,
                id=sub_module,
                type="instance"
            )

            if matching_sub_module:
                last_node = matching_sub_module
            else:
                new_sub_module = Node(
                    id=sub_module,
                    type="instance",
                    context=context
                )
                last_node.add_child(new_sub_module)
                last_node = new_sub_module

        return last_node

    def _process_tokenized_node(self, tokenized_node):
        """
        Takes a list of tokens (either call, name, or subscript nodes) and
        searches for an equivalent node path in the package forest. Once the
        tail of the known path has been reached, the algorithm create additional
        nodes and adds them to the path.

        @param tokenized_node: list of token tuples, (name, type).

        @return: list of nodes representing node paths.
        """

        # BUG: Edge case
        # An assignment is visited, and the target, x.y.z, is stored as an alias
        # for it's value (i.e. stored as "x.y.z"). But when you search for nodes,
        # you break apart the attributes and go step by step, thereby missing
        # concatenated aliases. Growing window as possible solution?

        node_stack = []
        curr_context = self._context_to_string()

        print("\nTOKENIZED_NODE:", tokenized_node)

        if tokenized_node: # [("var_name", "type"), ("func_name", "type"), ...]
            for idx, token in enumerate(tokenized_node):
                not_last_token = idx < len(tokenized_node) - 1
                token_name, token_type = token

                # Merges all subscript tokens into one string (i.e. id + slice1 + slice2 + ...)
                if not_last_token and tokenized_node[idx + 1][1]  == "subscript":
                    for next_token in tokenized_node[idx + 1:]:
                        if next_token[1] == "subscript":
                            token_name += next_token[0]
                        else:
                            break

                if not node_stack and not idx: # Beginning of iteration
                    for root in self._parse_trees:
                        matching_node = find_matching_node(
                            subtree=root,
                            id=token_name,
                            type=token_type,
                            context=curr_context,
                            default_type="module" # BUG: Not always a module
                        )

                        if matching_node: # Found base node, initializing chain
                            print("\tFound first match:", matching_node) # TEMP
                            node_stack.append(matching_node)
                            break
                elif node_stack: # Chain exists, continue adding to it
                    matching_node = find_matching_node(
                        subtree=node_stack[-1],
                        id=token_name,
                        type=token_type,
                        context=curr_context,
                        default_type="instance"
                    )

                    if matching_node: # Found child node
                        print("\tFound child node:", matching_node) # TEMP
                        if not not_last_token: # Increment terminal token
                            matching_node.increment_count()
                        node_stack.append(matching_node)
                    else: # No child node found, creating one
                        child_node = Node(
                            id=token_name,
                            type=token_type,
                            context=curr_context,
                            alias=token_name
                        )

                        print("\tNo child node found, creating one:", child_node) # TEMP

                        node_stack[-1].add_child(child_node)
                        node_stack.append(child_node)
                else: # Base token doesn't exist, abort processing
                    break

        return node_stack

    def _handle_assignment(self, target, value):
        curr_context = self._context_to_string()
        tokenized_target = recursively_tokenize_node(target, [])

        targ_matches = self._process_tokenized_node(tokenized_target)
        val_matches = self._process_tokenized_node(value)

        add_alias = lambda tail: tail.add_alias(
            context=curr_context,
            alias=stringify_tokenized_node(tokenized_target)
        )
        del_alias = lambda tail: tail.del_alias(
            context=curr_context,
            alias=stringify_tokenized_node(tokenized_target)
        )

        if targ_matches and val_matches: # Known node reassigned to known node
            add_alias(val_matches[-1])
            del_alias(targ_matches[-1])
        elif targ_matches and not val_matches: # Known node reassigned to unknown node
            del_alias(targ_matches[-1])
        elif not targ_matches and val_matches: # Unknown node assigned to known node
            add_alias(val_matches[-1])

    ## Context Managers ##

    def visit_Global(self, node):
        return self._visiter_wrapper(node, lambda: None)

    def visit_Nonlocal(self, node):
        return self._visiter_wrapper(node, lambda: None)

    def visit_ClassDef(self, node):
        callback = lambda: self._update_context('#'.join(["class", node.name]), node)
        return self._visiter_wrapper(node, callback)

    def visit_FunctionDef(self, node):
        callback = lambda: self._update_context('#'.join(["function", node.name]), node)
        return self._visiter_wrapper(node, callback)

    def visit_AsyncFunctionDef(self, node):
        callback = lambda: self._update_context('#'.join(["async_function", node.name]), node)
        return self._visiter_wrapper(node, callback)

    def visit_Lambda(self, node):
        return self._visiter_wrapper(node, lambda: None)

    ## Node Visitors ##

    def visit_Import(self, node):
        def callback():
            curr_context = self._context_to_string()
            for module in node.names:
                alias = module.asname if module.asname else module.name
                module_leaf_node = self._process_module(
                    module=module.name,
                    context=curr_context,
                    exclude_root_alias=bool(module.asname)
                )
                module_leaf_node.add_alias(curr_context, alias)

        return self._visiter_wrapper(node, callback)

    def visit_ImportFrom(self, node):
        def callback():
            curr_context = self._context_to_string()
            module_parent_node = self._process_module(
                module=node.module,
                context=curr_context,
                exclude_root_alias=True
            )

            for alias in node.names:
                child_exists = False
                alias_id = alias.asname if alias.asname else alias.name

                for child in module_parent_node.children:
                    if alias.name == child.id:
                        child_exists = True
                        if not alias_id in child.aliases[curr_context]:
                            child.add_alias(curr_context, alias_id)

                        break

                if not child_exists:
                    module_parent_node.add_child(Node(
                        id=alias.name,
                        type="instance",
                        context=curr_context,
                        alias=alias_id
                    ))

        return self._visiter_wrapper(node, callback)

    def visit_Assign(self, node):
        def callback():
            curr_context = self._context_to_string()

            if isinstance(node.value, ast.Tuple):
                values = tuple(recursively_tokenize_node(elt, []) for elt in node.value.elts)
            else:
                values = recursively_tokenize_node(node.value, [])

            for target in node.targets:
                if isinstance(target, ast.Tuple):
                    for idx, elt in enumerate(target.elts):
                        if isinstance(values, tuple):
                            self._handle_assignment(elt, values[idx])
                        else:
                            self._handle_assignment(elt, values)

                    continue

                if isinstance(values, tuple):
                    for value in values:
                        self._handle_assignment(target, value)
                else:
                    self._handle_assignment(target, values)

        return self._visiter_wrapper(node, callback)

    def visit_Call(self, node):
        def callback():
            if not isinstance(node.parent, ast.Assign) or not isinstance(node.parent, ast.Attribute):
                id = recursively_tokenize_node(node, [])
                print("\n\nCALL ID:", id, "\n\n")
                self._process_tokenized_node(id)

        return self._visiter_wrapper(node, callback)

    def visit_Attribute(self, node):
        def callback():
            if not isinstance(node.parent, ast.Call):
                print("PARENT:", node.parent)
                ids = recursively_tokenize_node(node, [])
                print("\n\nATTRIBUTE IDS:", ids, "\n\n")
                self._process_tokenized_node(ids)

        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        def callback():
            if isinstance(node.ctx, ast.Del):
                pass  # TODO: Delete alias from tree
            elif isinstance(node.ctx, ast.Load):
                pass  # TODO: Increment count of node

        return self._visiter_wrapper(node, callback)

    ## Public Methods ##

    def find(self, nodes=[]):
        """
        Both parameters are optional, and by default find() will return a list
        of all nodes contained in the AST.

        @param nodes: list of node types to retrieve.

        @return: list of matching AST nodes.
        """

        self._generate_parse_tree = False
        self._queried_nodes = nodes
        self._retrieved_nodes = []

        self.visit(self.tree)

        return self._retrieved_nodes

    def transform(self, nodes=[], transformer=lambda node: node):
        """
        Both are optional, and by default `transform()` will return the root
        node of the original AST, unchanged.

        @param nodes: list of node types to transform.
        @param transformer: user-defined function that takes an AST node as
        input and returns a modified version.

        @return: root node of the transformed AST.
        """

        self._generate_parse_tree = False
        self._queried_nodes = nodes
        self._transformer = transformer

        return self.visit(self.tree)

    def get_freq_map(self, nodes=[]):
        """
        Both parameters are optional, and by default get_freq_map() will return
        a dictionary containing all node types in the tree and their
        frequencies.

        @param nodes: list of node types to analyze.

        @return: dictionary mapping node types to their frequency of occurence
        in the AST.
        """

        self._generate_parse_tree = False
        self._queried_nodes = nodes
        self._freq_map = defaultdict(lambda: 0)

        self.visit(self.tree)

        return dict(self._freq_map)

    def get_halstead(metric_name):
        pass # TODO

    def get_api_usage_tree(self, flattened=False):
        self._generate_parse_tree = True
        self.visit(self.tree)

        parse_tree = {} if flattened else []
        for root in self._parse_trees:
            if flattened:
                parse_tree[root.id] = {}
                for path in root.flatten()[1:]:
                    parse_tree[root.id]['.'.join(path[0])] = path[1]
            else:
                # NOTE: 'True' for debugging purposes
                parse_tree.append(root.to_dict(True))

        return parse_tree

if __name__ == "__main__":
    with open("./output.json", 'w') as output:
        harvester = Harvester("../test.py")
        output.write(json.dumps(harvester.get_api_usage_tree()))
