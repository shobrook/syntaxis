# Standard Library
import ast
from collections import defaultdict
from copy import deepcopy

# Local
import utilities_new as utils


class Transducer(ast.NodeVisitor):
    def __init__(self, tree, forest=[], namespace={}):
        # QUESTION: Add a `conservative` parameter? If True, only deductive
        # type inferences will be made. Else, predictive inferences will be
        # made too. Example: if a module reference/alias is redefined inside
        # the body of an 'if' statement, conservative=True always assume the
        # body was executed.

        # Looks up saplings nodes by their current alias
        self._node_lookup_table = namespace.copy() # QUESTION: Or deepcopy?
        self._unprocessed_funcs = []
        self._returns = []

        self.forest = forest  # Holds root nodes of output trees
        self.visit(tree)  # Begins traversal

    ## Main Processor ##

    def _recursively_process_tokens(self, tokens, no_increment=False):
        """
        This is the master function for appending to an API tree. Takes a
        potentially nested list of (token, type) tuples and searches
        for equivalent nodes in the API forest. Once the terminal node of a
        path of equivalent nodes has been reached, additional nodes are created
        and added as children to the terminal node.

        @param tokens: list of tokenized nodes, each as (token(s), type).
        @param no_increment:
        @param is_data_struct:

        @return: list of references to parse tree nodes corresponding to the
        tokens.
        """

        node_stack = []

        # Flattens nested tokens into a list s.t. if the current token
        # references a node, then the next token is a child of that node
        flattened_tokens = []
        # QUESTION: Does this need to be in a separate loop?
        for idx, token in enumerate(tokens):
            content, type = token

            flattened_content = []
            if type == "args": # Function call arguments
                for arg_tokens in content:
                    node = self._recursively_process_tokens(arg_tokens)
                    flattened_content.append(node)
            elif type == "index": # Subscript index
                if len(content) == 1: # Index is a literal (e.g. my_dict["key"])
                    index_token, index_type = content[0]
                    if index_type not in ("str", "num"):
                        node = self._recursively_process_tokens(content)
                        flattened_content.append(node)
                    else:
                        flattened_content.append((index_token, index_type))
                else: # Index is a list of tokens (e.g. my_dict[my.func()])
                    node = self._recursively_process_tokens(content)
                    flattened_content.append(node)
            elif type == "comprehension":
                # NOTE: Comprehensions have their own namespace; assuming there
                # is a visit_Comprehension function for updating nodes from
                # comprehensions, you can create a new Transducer instance for
                # this token (content should be the original comprehension node)
                # with the namespace updated with the local comprehension
                # variable(s).
                pass
            elif type in ("array", "dict"):
                # TODO: Handle these (don't use a new Transducer object since
                # data structures don't have their own namespaces)
                pass
            else:
                flattened_content = content

            flattened_tokens.append((flattened_content, type))

        for index, token in enumerate(flattened_tokens):
            content, type = token
            token_sequence = flattened_tokens[:index + 1]

            # TEMP: This can't handle something like [1,2,3, ...][0].foo()
            if type in ("comprehension", "array", "dict"):
                # QUESTION: If hit, will this always be the first token?
                break

            token_str = utils.stringify_tokenized_nodes(token_sequence)
            if token_str in self._node_lookup_table:
                node_info = self._node_lookup_table[token_str]
                node, node_type = node_info["node"], node_info["type"]

                if node_type == "func_def_node":
                    is_last_token = index == len(flattened_tokens) - 1
                    if not is_last_token:
                        next_content, next_type = flattened_tokens[index + 1]

                        # Locally defined function call
                        if next_type == "args": # QUESTION: Is next_type always gonna be "args"?
                            # TODO: Pass the function node into a Transducer with an
                            # updated namespace. You need to recover the call args to
                            # do this. tokens[idx] should be enough for this. Also,
                            # if the function returns a parse tree node, that needs
                            # to be added to node_stack (and you shouldn't break).

                            # unflattened_args_tokens = tokens[index + 1]

                            updated_namespace = self._node_lookup_table.copy()
                            args = zip((a.arg for a in node.args.args), next_content)
                            for arg_name, arg_node_info in args:
                                arg_node, arg_node_type = arg_node_info
                                if not arg_node:
                                    continue

                                updated_namespace[arg_name] = arg_node

                            # BUG: Recursive functions will cause this to enter
                            # an infinite loop

                            # func_transducer = Transducer(
                            #     ast.Module(body=node.body),
                            #     self.forest,
                            #     updated_namespace
                            # )
                            # re_node, re_node_type = func_transducer._returns[-1]
                            #
                            # func_call_str = utils.stringify_tokenized_nodes(
                            #     flattened_tokens[:index + 2]
                            # )

                            # self._node_lookup_table[func_call_str] = {
                            #     "node": re_node,
                            #     "type": re_node_type
                            # }
                            # return re_node, re_node_type
                            return None, None

                    # TODO: Test this (function reassignment, etc.)
                    return node, node_type

                    # NOTE: I am assuming that a locally defined function can
                    # only be referenced as a name (i.e. an attribute can't be
                    # called on it, it can't be subscripted, etc.) or as a call.

                if not no_increment:
                    node.increment_count()

                node_stack.append(node)
            elif node_stack:  # Base node exists; create its child
                if type == "args":
                    content = "()"
                elif type == "index" and len(content) == 1:
                    index_token, index_type = content[0]
                    if index_type not in ("str", "num"):
                        content = "[]"
                    else:
                        content = '[' + index_token + ']'
                elif type == "index":
                    content = "[]"

                child_node = node_stack[-1].add_child(utils.Node(content))
                self._node_lookup_table[token_str] = {
                    "node": child_node,
                    "type": "module_node"
                }
                node_stack.append(child_node)
            else: # Base node doesn't exist; abort processing
                break

        if not node_stack:
            return None, None

        return node_stack[-1], "module_node"

    ## Visitor Helpers ##

    def _process_module(self, module, alias_origin_module=True):
        """
        Takes the identifier for a module, sometimes a period-separated string
        of sub-modules, and searches the API forest for a matching module. If no
        match is found, new module nodes are generated and appended to
        self.forest.

        @param module: identifier for the module.
        @param alias_origin_module: flag for whether a newly created module node
        should be aliased.

        @return: reference to the terminal Node object in the list of
        sub-modules.
        """

        sub_modules = module.split('.')  # For module.submodule1.submodule2...
        root_module = sub_modules[0]
        term_node = None

        def find_matching_node(subtree, id):
            for node in subtree.breadth_first():
                if node.id == id:
                    return node

            return None

        for root in self.forest:
            matching_module = find_matching_node(root, root_module)

            if matching_module:
                # QUESTION: Do you need to add this to the alias lookup table?
                term_node = matching_module
                break

        if not term_node:
            root_node = utils.Node(root_module)
            if alias_origin_module:  # False if `from X import Y`
                self._node_lookup_table[root_module] = {
                    "node": root_node,
                    "type": "module_node"
                }

            term_node = root_node
            self.forest.append(term_node)

        for idx in range(len(sub_modules[1:])):
            sub_module = sub_modules[idx + 1]
            sub_module_alias = '.'.join([root_module] + sub_modules[1:idx + 2])

            matching_sub_module = find_matching_node(term_node, sub_module)

            if matching_sub_module:
                term_node = matching_sub_module
            else:
                new_sub_module = utils.Node(sub_module)
                if alias_origin_module:  # False if `from X.Y import Z`
                    self._node_lookup_table[sub_module_alias] = {
                        "node": new_sub_module,
                        "type": "module_node"
                    }

                term_node.add_child(new_sub_module)
                term_node = new_sub_module

        return term_node

    def _process_assignment(self, target, value):
        """
        @param target: AST node on the left-hand-side of the assignment.
        @param value: tokenized AST node on the right-hand-side of the
        assignment.
        """

        def delete_all_sub_aliases(targ_str):
            sub_aliases = []
            for alias in self._node_lookup_table.keys():
                is_attr_str = alias.startswith(targ_str + '(')
                is_call_str = alias.startswith(targ_str + '.')
                is_idx_str = alias.startswith(targ_str + '[')

                if is_attr_str or is_call_str or is_idx_str:
                    sub_aliases.append(alias)

            for alias in sub_aliases:
                del self._node_lookup_table[alias]

        tokenized_target = utils.recursively_tokenize_node(target, tokens=[])
        targ_node, targ_type = self._recursively_process_tokens(tokenized_target)

        tokenized_value = utils.recursively_tokenize_node(value, tokens=[])
        val_node, val_type = self._recursively_process_tokens(tokenized_value)

        # BUG: If value is a comprehension or data structure, self._recursively_process_tokens
        # will not return a node. But if we have: targ = [node(i) for i in range(10)],
        # \^targ\ should be an alias for node(). Ignore this for now.

        # NOTE: This is stringifying the non-flattened tokens
        targ_str = utils.stringify_tokenized_nodes(tokenized_target)
        val_str = utils.stringify_tokenized_nodes(tokenized_value)

        # Known node reassigned to other known node (K2 = K1)
        if targ_node and val_node:
            self._node_lookup_table[targ_str] = self._node_lookup_table[val_str]
            delete_all_sub_aliases(targ_str)
        # Known node reassigned to unknown node (K1 = U1)
        elif targ_node and not val_node:
            del self._node_lookup_table[targ_str]
            delete_all_sub_aliases(targ_str)
        # Unknown node assigned to known node (U1 = K1)
        elif not targ_node and val_node:
            self._node_lookup_table[targ_str] = self._node_lookup_table[val_str]

    ## Aliasing Handlers ##

    def visit_Import(self, node):
        for module in node.names:
            if module.name.startswith('.'):  # Ignores relative imports
                continue

            alias = module.asname if module.asname else module.name
            module_leaf_node = self._process_module(
                module=module.name,
                alias_origin_module=not bool(module.asname)
            )

            self._node_lookup_table[alias] = {
                "node": module_leaf_node,
                "type": "module_node"
            }

    def visit_ImportFrom(self, node):
        if node.level:  # Ignores relative imports
            return

        module_node = self._process_module(
            module=node.module,
            alias_origin_module=False
        )

        for alias in node.names:
            if alias.name == '*':  # Ignore star imports
                continue

            child_exists = False
            alias_id = alias.asname if alias.asname else alias.name

            for child in module_node.children:
                if alias.name == child.id:
                    child_exists = True
                    self._node_lookup_table[alias_id] = {
                        "node": child,
                        "type": "module_node"
                    }

                    break

            if not child_exists:
                new_child = utils.Node(alias.name)
                self._node_lookup_table[alias_id] = {
                    "node": new_child,
                    "type": "module_node"
                }

                module_node.add_child(new_child)

    def visit_Assign(self, node):
        values = node.value
        targets = node.targets if hasattr(node, "targets") else (node.target)
        for target in targets:
            if isinstance(target, ast.Tuple):
                for idx, elt in enumerate(target.elts):
                    if isinstance(values, ast.Tuple):
                        self._process_assignment(elt, values.elts[idx])
                    else:
                        self._process_assignment(elt, values)
            elif isinstance(values, ast.Tuple):
                for value in values.elts:
                    self._process_assignment(target, value)
            else:
                self._process_assignment(target, values)

    def visit_AnnAssign(self, node):
        self.visit_Assign(node)

    def visit_AugAssign(self, node):
        pass # TODO: How should this be handled?

    def visit_Delete(self, node):
        pass # TODO: Delete this alias

    ## Context Handlers ##

    def visit_ClassDef(self, node):
        pass

    def visit_FunctionDef(self, node):
        # TODO: Handle decorators

        if node.name in self._node_lookup_table:
            node_info = self._node_lookup_table[node.name]

            if node_info["type"] == "func_def_node" and not node_info["called"]:
                # If the function hasn't been called, then its subtree still
                # needs to be processed
                self._unprocessed_funcs.append(node_info)

        # QUESTION: Delete self._node_lookup_table[node.name] before saving copy
        # of namespace?

        namespace_when_defined = self._node_lookup_table.copy()
        self._node_lookup_table[node.name] = {
            "node": node,
            "namespace_when_defined": namespace_when_defined,
            "called": False,
            "type": "func_def_node"
        }

        # BUG: namespace_when_defined is too big. Try some sort of caching.

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    ## Control Flow Handlers ##

    # def visit_If(self, node):
    #     pass
    #
    # def visit_Try(self, node):
    #     pass
    #
    # def visit_While(self, node):
    #     pass
    #
    # def visit_For(self, node):
    #     pass
    #
    # def visit_AsyncFor(self, node):
    #     self.visit_For(node)
    #
    # def visit_With(self, node):
    #     pass
    #
    # def visit_AsyncWith(self, node):
    #     self.visit_With(node)
    #
    # def visit_withitem(self, node):
    #     pass

    ## Reference Handlers ##

    @utils.reference_handler
    def visit_Name(self, node):
        pass

    @utils.reference_handler
    def visit_Attribute(self, node):
        pass

    @utils.reference_handler
    def visit_Call(self, node):
        pass

    @utils.reference_handler
    def visit_Subscript(self, node):
        pass

    ## Miscellaneous ##

    def visit_Return(self, node):
        tokenized_node = utils.recursively_tokenize_node(node.value, [])
        self._returns.append(self._recursively_process_tokens(tokenized_node))

    ## Public Methods ##

    def trees(self, flattened=False):
        trees = {} if flattened else []
        for tree in self.forest:
            if flattened:
                trees[tree.id] = {}
                paths = tree.flatten()
                for path in paths[1:]:
                    trees[tree.id]['.'.join(path[0])] = path[1]
            else:
                trees.append(tree.to_dict(debug=True))

        return trees
