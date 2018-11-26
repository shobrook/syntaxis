#######
# NOTES
#######


# AST to JSON: https://github.com/fpoli/python-astexport
# AST Parser that's version-agnostic: https://github.com/serge-sans-paille/gast
# AST Unparser: https://github.com/simonpercivall/astunparse
# AST Utilities: https://github.com/mutpy/astmonkey

# Rename "parse tree" to "API tree" and rename "context" to "scope"
# Remove "function" nodes and keep only module, instance, and implicit nodes
# Make _process_tokenized_nodes recursive
# Inside functions, block searching parent contexts for nodes with aliases equivalent to the function parameter names (unless it's self)
# Make sure dot-aliasing works properly
# Add implicit nodes (implicit(1), implicit(2), ...)
# Ignore imports with dots in front (i.e. `import .local_module` or `from .local_module import X`)
# Infer whether return types of user-defined functions are package-related
    # Create a field mapping user-defined function names to...


#########
# GLOBALS
#########


import ast
import json
from collections import defaultdict

import utils


##################
# PARSE TREE NODES
##################


class Node(object):
    def __init__(self, id, type="instance", context='', alias='', children=[]):
        self.id, self.type = str(id), type
        self.children = []
        self.aliases = defaultdict(lambda: set())
        self.dead_aliases = defaultdict(lambda: set())
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

    def change_type(self, new_type):
        self.type = new_type

    def add_alias(self, context, alias):
        self.aliases[context] |= {str(alias)}

    def del_alias(self, context, alias):
        if context in self.aliases:
            self.aliases[context].discard(str(alias))
        else:
            self.dead_aliases[context] |= {str(alias)}

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


##########
# SAPLINGS
##########


class Saplings(ast.NodeTransformer):
    def __init__(self, tree):
        """
        Initializes everyone for everything.

        @param tree: either the path to a Python file or an already parsed AST
        for a Python file.
        """

        if isinstance(tree, ast.Module): # Already parsed Python AST
            self.tree = tree
        elif isinstance(tree, str): # Path to Python file
            self.tree = ast.parse(open(tree, 'r').read())
        else:
            raise Exception # TODO: Create custom exception

        # OPTIMIZE: Turns AST into doubly-linked AST
        for node in ast.walk(self.tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node

        self._context_stack = ["global"] # Keeps track of current context
        self._parse_trees = [] # Holds root nodes of parse trees

        self._call_table = {} # For holding user-defined funcs and classes
        self._iterator_table = {} # For holding user-defined ListComps, etc.

        self._generate_parse_tree = False

        self._queried_nodes = [] # TODO: Figure out a better variable name
        self._retrieved_nodes = []
        self._freq_map = defaultdict(lambda: 0)
        self._transformer = lambda node: node

        self._method_to_loc_map = {} # Holds a mapping from method names to a LOC range
        self._complexity = 1 # if off else 0
        self._functions = []
        self._classes = []

        self._context_to_string = lambda: '.'.join(self._context_stack)

    ## Parse Tree Utilities ##

    def _recursively_process_tokens(self, tokens):
        """
        Takes a list of tokens (types: instance, function, args, or subscript)
        and searches for an equivalent nodes in each parse tree. Once the leaf
        of the path of equivalent nodes has been reached, the algorithm creates
        additional nodes and adds them to the parse tree.

        @param tokens: list of tokenized nodes, each as (token(s), type).

        @return: list of references to parse tree nodes corresponding to the
        tokens.
        """

        node_stack = []
        curr_context = self._context_to_string()

        # Flattens nested tokens
        flattened_tokens = []
        for idx, token in enumerate(tokens):
            content, type = token

            adjunctive_content = []
            if type == "args":
                for sub_tokens in content:
                    adjunctive_content.append(self._recursively_process_tokens(sub_tokens))
                content = id(str(adjunctive_content)) if adjunctive_content else id(str(content))
            elif type == "subscript":
                adjunctive_content = self._recursively_process_tokens(content)
                content = id(str(adjunctive_content)) if adjunctive_content else id(str(content))

            flattened_tokens.append((content, type))

        for idx, token in enumerate(flattened_tokens):
            content, type = token
            token_id = utils.stringify_tokenized_nodes(flattened_tokens[:idx + 1])

            print("\nTOKEN ID:", token_id)
            print("ORIGINAL TOKEN:", token)

            type_pattern = adjusted_type = "implicit"
            if type == "args":
                adjusted_id = "call"
            elif type == "subscript":
                adjusted_id = "sub"
            else:
                adjusted_id = content
                adjusted_type = "instance"
                type_pattern = "instance/implicit/module"

            if not idx: # Beginning of iteration
                for root in self._parse_trees:
                    matching_node = utils.find_matching_node(
                        subtree=root,
                        id=token_id,
                        type_pattern="instance/module/implicit",
                        context=curr_context
                    )

                    if matching_node: # Found base node, pushing to stack
                        node_stack.append(matching_node)
                        break
            elif node_stack: # Stack exists, continue pushing to it
                matching_node = utils.find_matching_node(
                    subtree=node_stack[-1],
                    id=token_id,
                    type_pattern=type_pattern,
                    context=curr_context
                )

                if matching_node: # Found child node
                    node_stack.append(matching_node)
                else: # No child node found, creating one
                    child_node = Node(
                        id=adjusted_id,
                        type=adjusted_type,
                        context=curr_context,
                        alias=token_id
                    )

                    node_stack[-1].add_child(child_node)
                    node_stack.append(child_node)
            else: # Base token doesn't exist, abort processing
                break

        return node_stack

    def _process_assignment(self, target, value):
        """
        @param target:
        @param value:
        """

        curr_context = self._context_to_string()
        tokenized_target = utils.recursively_tokenize_node(target, [])

        targ_matches = self._recursively_process_tokens(tokenized_target)
        val_matches = self._recursively_process_tokens(value)

        alias = utils.stringify_tokenized_nodes(tokenized_target)
        add_alias = lambda tail: tail.add_alias(curr_context, alias)
        del_alias = lambda tail: tail.del_alias(curr_context, alias)

        if targ_matches and val_matches: # Known node reassigned to known node
            add_alias(val_matches[-1])
            del_alias(targ_matches[-1])
        elif targ_matches and not val_matches: # Known node reassigned to unknown node
            del_alias(targ_matches[-1])
        elif not targ_matches and val_matches: # Unknown node assigned to known node
            add_alias(val_matches[-1])

    def _process_module(self, module, context, alias_root=True):
        """
        Takes the identifier for a module, sometimes a period-separated string
        of sub-modules, and searches the list of parse trees for a matching
        module. If no match is found, new module nodes are generated and
        appended to self._parse_trees.

        @param module: identifier for the module.
        @param context: context in which the module is imported.
        @param alias_root: flag for whether a newly created module node should
        be aliased.

        @return: reference to the terminal Node object in the list of
        sub-modules.
        """

        sub_modules = module.split('.') # For module.submodule1.submodule2...
        root_module = sub_modules[0]
        term_node = None

        for root in self._parse_trees:
            matching_module = utils.find_matching_node(
                subtree=root,
                id=root_module,
                type_pattern="module"
            )

            if matching_module:
                term_node = matching_module
                break

        if not term_node:
            term_node = Node(
                id=root_module,
                type="module",
                context=context,
                alias=root_module if alias_root else '' # For `from X import Y`
            )
            self._parse_trees.append(term_node)

        for idx in range(len(sub_modules[1:])):
            sub_module = sub_modules[idx + 1]
            sub_module_alias = '.'.join([root_module] + sub_modules[1:idx + 2])

            matching_sub_module = utils.find_matching_node(
                subtree=term_node,
                id=sub_module,
                type_pattern="instance",
                context=None # QUESTION: Should this be context?
            )

            if matching_sub_module:
                term_node = matching_sub_module
            else:
                new_sub_module = Node(
                    id=sub_module,
                    type="instance",
                    context=context,
                    alias=sub_module_alias if alias_root else '' # For `from X.Y import Z`
                )
                term_node.add_child(new_sub_module)
                term_node = new_sub_module

        return term_node

    ## Overloaded Methods ##

    def generic_visit(self, node):
        """
        Overloaded ast.NodeTransformer.generic_visit function that

        @param node:

        @return: either the original node or a modified version.
        """

        super().generic_visit(node)

        # QUESTION: Why aren't these in their respective visitor methods?
        if isinstance(node, ast.Try):
            self._complexity += len(node.handlers) + len(node.orelse)
        elif isinstance(node, ast.BoolOp):
            self._complexity += len(node.values) - 1
        elif isinstance(node, ast.With) or isinstance(node, ast.If) or isinstance(node, ast.IfExp) or isinstance(node, ast.AsyncWith):
            self._complexity += 1
        elif isinstance(node, ast.For) or isinstance(node, ast.While) or isinstance(node, ast.AsyncFor):
            self._complexity += bool(node.orelse) + 1
        elif isinstance(node, ast.comprehension):
            self._complexity += len(node.ifs) + 1
        elif isinstance(node, ast.Assert):
            self._complexity += 1

        # NOTE: Figure out how to handle lambda functions (see #68 in radon)

        node_is_queried = any(isinstance(node, n) for n in self._queried_nodes)
        if node_is_queried or not self._queried_nodes:
            self._retrieved_nodes.append(node)
            self._freq_map[type(node).__name__] += 1

            # Applies a user-defined modification to the node
            return ast.copy_location(self._transformer(node), node)

        return node

    # def visit_Global(self, node): # IDEA: Save pre-state of context_stack, set to ["global"], then set back to pre-state

    # def visit_Nonlocal(self, node):

    @utils.context_manager("class")
    def visit_ClassDef(self, node):
        return

    @utils.context_manager("function")
    def visit_FunctionDef(self, node):
        self._method_to_loc_map[node.name] = list(range(node.lineno, utils.get_max_lineno(node.body[-1]) + 1))
        return

    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node)

    # @utils.context_manager("lambda")
    # def visit_Lambda(self, node):
    #     return

    @utils.visitor
    def visit_Import(self, node):
        curr_context = self._context_to_string()
        for module in node.names:
            alias = module.asname if module.asname else module.name
            module_leaf_node = self._process_module(
                module=module.name,
                context=curr_context,
                alias_root=not bool(module.asname)
            )

            module_leaf_node.add_alias(curr_context, alias)

    @utils.visitor
    def visit_ImportFrom(self, node):
        curr_context = self._context_to_string()
        module_node = self._process_module(
            module=node.module,
            context=curr_context,
            alias_root=False
        )

        for alias in node.names:
            child_exists = False
            alias_id = alias.asname if alias.asname else alias.name

            for child in module_node.children:
                if alias.name == child.id:
                    child_exists = True
                    if not alias_id in child.aliases[curr_context]:
                        child.add_alias(curr_context, alias_id)

                    break

            if not child_exists:
                module_node.add_child(Node(
                    id=alias.name,
                    type="instance",
                    context=curr_context,
                    alias=alias_id
                ))

    @utils.visitor # TEMP: False
    def visit_Assign(self, node):
        curr_context = self._context_to_string()

        if isinstance(node.value, ast.Tuple):
            node_tokenizer = lambda elt: utils.recursively_tokenize_node(elt, [])
            values = tuple(map(node_tokenizer, node.value.elts))
        else:
            values = utils.recursively_tokenize_node(node.value, [])

        for target in node.targets:
            if isinstance(target, ast.Tuple):
                for idx, elt in enumerate(target.elts):
                    if isinstance(values, tuple):
                        self._process_assignment(elt, values[idx])
                    else:
                        self._process_assignment(elt, values)
            elif isinstance(values, tuple):
                for value in values:
                    self._process_assignment(target, value)
            else:
                self._process_assignment(target, values)

    @utils.visitor
    def visit_Call(self, node):
        tokens = utils.recursively_tokenize_node(node, [])
        nodes = self._recursively_process_tokens(tokens)

    @utils.visitor
    def visit_Attribute(self, node):
        # You could try searching up the node.parent.parent... path to find
        # out if attribute is inside a call node. If it is, let the call visiter
        # take care of it. If it isn't, then keep doing what you're doing.

        # Also possible that the best way to deal with this is by just having
        # one ast.Load visitor. Look into this more, i.e. what gets covered by
        # ast.Load.

        tokens = utils.recursively_tokenize_node(node, [])
        nodes = self._recursively_process_tokens(tokens)

    @utils.visitor
    def visit_Subscript(self, node):
        pass

    @utils.visitor
    def visit_comprehension(self, node):
        pass

    @utils.visitor
    def visit_Dict(self, node):
        pass

    @utils.visitor
    def visit_List(self, node):
        pass

    @utils.visitor
    def visit_Tuple(self, node):
        pass

    @utils.visitor
    def visit_Set(self, node):
        pass

    @utils.visitor
    def visit_Compare(self, node):
        pass

    @utils.visitor
    def visit_IfExp(self, node):
        pass

    @utils.visitor
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Del):
            pass  # TODO: Delete alias from tree
        elif isinstance(node.ctx, ast.Load):
            pass  # TODO: Increment count of node

    ## Public Methods (i.e. saplings) ##

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

    def get_halstead(self, metric_name):
        pass

    def get_complexity(self):
        pass

    def get_method_couplings(self):
        pass

    def get_method_to_loc_map(self):
        self._method_to_loc_map = {}
        self._generate_parse_tree = False
        self.visit(self.tree)

        return self._method_to_loc_map

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
    import sys
    sys.path = sys.path[1:] + ['']
    from saplings import Harvester, Roots

    with open("./output.json", 'w') as output:
        # saplings = Saplings("./test.py")
        # saplings = Saplings("./cases.py")
        saplings = Saplings("./test2.py")
        output.write(json.dumps(saplings.get_api_usage_tree()))

    with open("./old_output.json", 'w') as old_output:
        # file_ast = ast.parse(open("./test.py").read())
        file_ast = ast.parse(open("./test2.py").read())
        old_output.write(json.dumps(Roots(file_ast).to_dict(False)))


########################
# FLOATING CODE SNIPPETS
########################


# def export_json(tree):
#     return json.dumps(
#         DictExportVisitor().visit(tree),
#         sort_keys=True,
#         separators=(',', ':')
#     )
#
# class DictExportVisitor:
#     def visit(self, node):
#         method_name = "visit_" + type(node).__name__
#         method = getattr(self, method_name, self.default_visit)
#
#         return method(node)
#
#     def default_visit(self, node):
#         node_type = type(node).__name__
#         args = {"node": node_type}
#
#         # Visit fields
#         for field in node._fields:
#             method_name = "visit_field_" + node_type + '_' + field
#             method = getattr(self, method_name, self.default_visit_field)
#
#             args[field] = method(getattr(node, field))
#
#         # Visit attributes
#         for attr in node._attributes:
#             method_name = "visit_attribute_" + node_type + '_' + attr
#             method = getattr(self, method_name, self.default_visit_field)
#
#             args[attr] = method(getattr(node, attr, None))
#
#         return args
#
#     def default_visit_field(self, field):
#         if isinstance(field, ast.AST):
#             return self.visit(field)
#         elif isinstance(field, list) or isinstance(field, tuple):
#             return [self.visit(f) for f in field]
#
#         return field
#
#     # Special visitors
#
#     def visit_str(self, field):
#         return str(field)
#
#     def visit_Bytes(self, field):
#         return str(field.s)
#
#     def visit_NoneType(self, field):
#         return None
#
#     def visit_field_NameConstant_value(self, field):
#         return str(field)
#
#     def visit_field_Num_n(self, field):
#         if isinstance(field, int):
#             return {"node": "int", "n": field, "n_str": str(field)}
#         elif isinstance(field, float):
#             return {"node": "float", "n": field}
#         elif isinstance(field, complex):
#             return {"node": "complex", "n": field.real, "i": field.imag}
#
# with open("./ast.json", 'w') as ast_viz:
#     file_ast = ast.parse(open("./test.py").read())
#     ast_viz.write(export_json(file_ast))
