# Standard Library
import ast
from collections import defaultdict
from copy import deepcopy

# Local
import utilities as utils


class Transducer(ast.NodeVisitor):
    def __init__(self, tree, forest=[], namespace={}):
        # QUESTION: Add a `conservative` parameter? If True, only deductive
        # type inferences will be made. Else, predictive inferences will be
        # made too. Example: if a module reference/alias is redefined inside
        # the body of an 'if' statement, conservative=True always assume the
        # body was executed.

        # Looks up nodes (parse tree and AST) by their current alias
        self._node_lookup_table = namespace.copy() # QUESTION: Or deepcopy?
        self._uncalled_funcs, self._return_node = {}, None

        self.forest = forest # Holds root nodes of output trees
        self.visit(tree) # Begins traversal

        # Processes uncalled local functions
        for func_node, func_namespace in self._uncalled_funcs.items():
            self._process_local_func_call(
                func_node=func_node,
                namespace=func_namespace
            )

    ## Helpers ##

    def _process_func_args(self, args, defaults):
        num_args, arg_names = 0, []
        for arg in args:
            arg_names.append(arg.arg)
            num_args += 1

        num_defaults, defaults = len(defaults), {}
        for index, default in enumerate(defaults):
            arg_name_index = index + (num_args - num_defaults)

            # Only kw_defaults can be None
            if not default:
                continue

            tokenized_default = utils.recursively_tokenize_node(
                default,
                []
            )
            # BUG: Default values should be processed in a separate transducer
            # with the same namespace as the one the function was defined in
            default_node = self._recursively_process_tokens(tokenized_default)

            default_node, default_node_type = default_node
            if not default_node:
                continue

            arg_name = arg_names[arg_name_index]
            defaults[arg_name] = default_node

        return arg_names, defaults


    def _process_local_func_call(self, func_node, namespace, args=[]):
        func_params = func_node.args
        arg_names, default_nodes = self._process_func_args(
            func_params.args,
            func_params.defaults
        )
        kwonlyarg_names, kw_default_nodes = self._process_func_args(
            func_params.kwonlyargs,
            func_params.kw_defaults
        )

        if func_params.vararg:
            arg_names.append(func_params.vararg.arg)
        arg_names.extend(kwonlyarg_names)
        if func_params.kwarg:
            arg_names.append(func_params.kwarg.arg)

        namespace = {
            **namespace,
            **default_nodes,
            **kw_default_nodes
        }

        # TODO: Handle single-star args
        for index, arg_token in enumerate(args):
            arg_node = self._recursively_process_tokens(
                arg_token.arg
            )

            if not arg_node:
                continue

            if not arg_token.arg_name:
                arg_name = arg_names[index]
            else:
                arg_name = arg_token.arg_name

            namespace[arg_name] = arg_node

        # BUG: Recursive functions will cause this to enter an infinite loop

        func_transducer = Transducer(
            ast.Module(body=func_node.body),
            self.forest,
            namespace
        )
        return func_transducer._return_node

    def _process_data_structure(self, token):
        is_data_structure = True

        # TEMP: This can't handle something like [1,2,3, ...][0].foo()
        if isinstance(token, utils.ComprehensionToken):
            # NOTE: Comprehensions have their own namespace; assuming there
            # is a visit_Comprehension function for updating nodes from
            # comprehensions, you can create a new Transducer instance for
            # this token (content should be the original comprehension node)
            # with the namespace updated with the local comprehension
            # variable(s).
            pass
        elif isinstance(token, utils.ArrayToken):
            # TODO: Handle these (don't use a new Transducer object since
            # data structures don't have their own namespaces)
            pass
        elif isinstance(token, utils.DictToken):
            # TODO: Handle these (don't use a new Transducer object since
            # data structures don't have their own namespaces)
            pass
        elif isinstance(token, utils.StrToken):
            pass
        else:
            is_data_structure = False

        return is_data_structure

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

        # Flattens nested tokens into a list s.t. if the current token
        # references a node, then the next token is a child of that node

        node_stack = []
        for index, token in enumerate(tokens):
            arg_nodes = []
            is_data_structure = self._process_data_structure(token)

            if is_data_structure:
                break
            elif isinstance(token, utils.ArgsToken):
                for arg_token in token:
                    node = self._recursively_process_tokens(arg_token.arg)
                    arg_nodes.append(node)
            elif isinstance(token, utils.IndexToken):
                self._recursively_process_tokens(token.slice)

            token_sequence = tokens[:index + 1]
            token_str = utils.stringify_tokenized_nodes(token_sequence)
            if token_str in self._node_lookup_table:
                node = self._node_lookup_table[token_str]

                if isinstance(node, ast.FunctionDef):
                    is_last_token = index == len(tokens) - 1
                    if not is_last_token:
                        # Locally defined function call
                        if node in self._uncalled_funcs:
                            # Function is called for the first time
                            del self._uncalled_funcs[node]

                        return_node = self._process_local_func_call(
                            func_node=node,
                            namespace=self._node_lookup_table.copy(),
                            args=tokens[index + 1].args
                        )
                        if not return_node:
                            break # BUG: What if the next token is an ArgsToken?
                        elif isinstance(return_node, ast.FunctionDef):
                            return return_node

                        node_stack.append(return_node)
                        continue # BUG: This double-counts the next token (need to skip next token)

                    # TODO: Test this (function reassignment, etc.)
                    return node

                if not no_increment:
                    node.increment_count()

                node_stack.append(node)
            elif node_stack:  # Base node exists; create its child
                child_node = utils.Node(str(token))
                node_stack[-1].add_child(child_node)
                self._node_lookup_table[token_str] = child_node
                node_stack.append(child_node)
            else: # Base node doesn't exist; abort processing
                continue # QUESTION: break?

        if not node_stack:
            return None

        return node_stack[-1]

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
                self._node_lookup_table[root_module] = root_node

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
                    self._node_lookup_table[sub_module_alias] = new_sub_module

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
        targ_node = self._recursively_process_tokens(tokenized_target)

        tokenized_value = utils.recursively_tokenize_node(value, tokens=[])
        val_node = self._recursively_process_tokens(tokenized_value)

        # BUG: If value is a comprehension or data structure, self._recursively_process_tokens
        # will not return a node. But if we have: targ = [node(i) for i in range(10)],
        # \^targ\ should be an alias for node(). Ignore this for now.

        targ_str = utils.stringify_tokenized_nodes(tokenized_target)
        # val_str = utils.stringify_tokenized_nodes(tokenized_value)

        # Known node reassigned to other known node (K2 = K1)
        if targ_node and val_node:
            self._node_lookup_table[targ_str] = val_node
            delete_all_sub_aliases(targ_str)
        # Known node reassigned to unknown node (K1 = U1)
        elif targ_node and not val_node:
            del self._node_lookup_table[targ_str]
            delete_all_sub_aliases(targ_str)
        # Unknown node assigned to known node (U1 = K1)
        elif not targ_node and val_node:
            self._node_lookup_table[targ_str] = val_node

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

            self._node_lookup_table[alias] = module_leaf_node

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
                    self._node_lookup_table[alias_id] = child

                    break

            if not child_exists:
                new_child = utils.Node(alias.name)
                self._node_lookup_table[alias_id] = new_child

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

        # QUESTION: Delete self._node_lookup_table[node.name] before saving copy
        # of namespace?

        self._node_lookup_table[node.name] = node
        self._uncalled_funcs[node] =  self._node_lookup_table.copy()

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    ## Control Flow Handlers ##

    def visit_If(self, node):
        # test, body, orelse
        self.generic_visit(node.test)

        body_transducer = Transducer(
            tree=ast.Module(body=node.body),
            forest=self.forest,
            namespace=self._node_lookup_table.copy()
        )

        # TODO: Process each branch independently (i.e. process assuming that
        # the first If evaluates, then assuming the first If doesn't evaluate
        # but the first elif does, and so on).

        # QUESTION: For now, maybe just process the absolute minimum? So: if
        # a K2 = K1 or K = U assignment is made in the body, then remove it from
        # the parent context. If a U = K assignment is made, don't let it apply
        # to the parent context. This can be hwat happens when conservative=True

        for else_node in node.orelse:
            self.visit_If(else_node)

        # QUESTION: What about:
            # import foo
            # for x in y:
            #    if True:
            #        continue
            #    z = foo()
        # We can't know if `z = foo()` is ever evaluated.

    def visit_Try(self, node):
        body_transducer = Transducer(
            tree=ast.Module(body=node.body),
            forest=self.forest,
            namespace=self._node_lookup_table.copy()
        )

        # TODO: Process branches independently

        for except_handler_node in node.handlers:
            self.visit_ExceptHandler(except_handler_node)

        # Executes only if node.body doesn't throw an exception
        for else_node in node.orelse:
            self.visit_If(else_node)

        # node.finalbody evaluates no matter what
        self.generic_visit(ast.Module(body=node.finalbody))

    def visit_ExceptHandler(self, node):
        namespace = self._node_lookup_table.copy()

        if node.type:
            tokenized_exception = utils.recursively_tokenize_node(node.type, [])
            exception_node = self._recursively_process_tokens(tokenized_exception)

            if node.name: # except A as B
                tokenized_alias = utils.recursively_tokenize_node(node.name, [])
                alias_str = utils.stringify_tokenized_nodes(tokenized_alias)

                if not exception_node:
                    del namespace[alias_str]
                else:
                    namespace[alias_str] = exception_node

        body_transducer = Transducer(
            tree=ast.Module(body=node.body),
            forest=self.forest,
            namespace=namespace
        )

    def visit_While(self, node):
        self.generic_visit(node.test)

        body_transducer = Transducer(
            tree=ast.Module(body=node.body),
            forest=self.forest,
            namespace=self._node_lookup_table.copy()
        )

        # Only executed when the while condition becomes False. If you break out
        # of the loop or if an exeception is raised, it won't be executed.
        else_transducer = Transducer(
            tree=node.orelse,
            forest=self.forest,
            namespace=self._node_lookup_table.copy()
        )

    # def visit_For(self, node):
    #     pass
    #
    # def visit_AsyncFor(self, node):
    #     self.visit_For(node)

    def visit_withitem(self, node):
        if isinstance(node.optional_vars, ast.Name):
            self._process_assignment(
                target=node.optional_vars,
                value=node.context_expr
            )
        elif isinstance(node.optional_vars, ast.Tuple):
            pass # TODO
        elif isinstance(node.optional_vars, ast.List):
            pass # TODO

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
        self._return_node = self._recursively_process_tokens(tokenized_node)

    ## Public Methods ##

    def trees(self, flattened=False):
        trees = {} if flattened else []
        for tree in self.forest:
            if flattened:
                trees[tree.id] = tree.paths()
            else:
                trees.append(tree.to_dict(debug=True))

        return trees
