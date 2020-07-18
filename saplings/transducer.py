# Standard Library
import ast
from collections import defaultdict
from copy import copy

# Local
import utilities as utils

# BUG: x["sub1"] and x["sub2"] are treated as the same aliases

# TODO: When an uncalled function is evaluated, its namespace needs to be
# adjusted for the function's parameter names (remove them from namespace).

class Saplings(ast.NodeVisitor):
    def __init__(self, tree, forest=[], namespace={}, process_uncalled_funcs=True):
        """
        Extracts dependency trees from imported modules in a program.

        Parameters
        ----------
        tree : ast.AST
            the AST representation of the program to be analyzed, or a subtree
        forest : {list, optional}
            root nodes of existing module dependency trees
        namespace : {dict, optional}
            node lookup table (mapping of aliases to d-tree and AST nodes)
        uncalled_funcs : {dict, optional}
            user-defined functions and the namespaces in which they were
            defined
        """

        # Maps active aliases to dependency tree nodes (and ast.FunctionDef
        # nodes)
        self._node_lookup_table = namespace
        self._uncalled_funcs, self._called_funcs = {}, set()
        self._return_node = None

        self.forest = forest # Holds root nodes of dependency trees
        self.visit(tree) # Begins traversal of the AST

        # If a function is defined but never called, we process the function
        # body after the traversal is over. The body is processed in the state
        # of the namespace in which it was defined.
        if process_uncalled_funcs:
            uncalled_func_nodes = list(self._uncalled_funcs.keys())
            for func_def_node in uncalled_func_nodes:
                # Skip b/c this may be called in some other context (either the
                # parent or another child of the parent)
                if func_def_node == self._return_node:
                    continue

                print("self._uncalled_funcs:")
                print(func_def_node.name)
                print()

                self._process_user_defined_func(
                    func_def_node=func_def_node,
                    namespace=self._uncalled_funcs[func_def_node]
                )

    ## ... ##

    def _run_child_saplings_instance(self, tree, namespace):
        print("\tEntering child Saplings instance\n")
        child_obj = Saplings(tree=tree, namespace=namespace)
        for func_def_node in child_obj._called_funcs:
            if func_def_node in self._uncalled_funcs:
                del self._uncalled_funcs[func_def_node]
        print("\tPopping out of Saplings instance\n")
        return child_obj

    ## Processors ##

    def _process_arg_defaults(self, arg_names, defaults):
        """
        Takes arguments from a user-defined function's signature and processes
        their default values.

        Parameters
        ----------
        args : list
            list of argument names (strings)
        defaults : list
            list of default values for the args (ast.AST nodes)

        Returns
        -------
        dict
            mapping of argument names to their default value's d-tree node
        """

        num_args, num_defaults = len(arg_names), len(defaults)
        arg_to_default_node = {}
        for index, default in enumerate(defaults):
            if not default: # Only kw_defaults can be None
                continue

            arg_name_index = index + (num_args - num_defaults)
            arg_name = arg_names[arg_name_index]

            # TODO (V1): Default values should be processed in a separate Saplings
            # instance with the same namespace as the one the function was
            # defined in
            tokenized_default = utils.recursively_tokenize_node(default, [])
            default_node = self._process_connected_tokens(tokenized_default)

            if not default_node:
                continue

            arg_to_default_node[arg_name] = default_node

        return arg_to_default_node

    def _process_user_defined_func(self, func_def_node, namespace, arg_vals=[]):
        """
        Processes the arguments and body of a user-defined function.

        Note: If the function is recursive, the recursive calls are not
        processed (otherwise this would throw Saplings into an infinite loop).

        Parameters
        ----------
        func_def_node : ast.FunctionDef
            AST node holding the signature and body of the function being called
        namespace : dict
            namespace in which the function was defined
        arg_vals : list, optional
            arguments passed into the function when called (list of
            utils.ArgTokens)

        Returns
        -------
        {utils.Node, ast.FunctionDef, ast.AsyncFunctionDef, None}
            d-tree node corresponding to the return value of the function
        """

        # TODO (V2): Handle single-star args (can't be done until data structure
        # assignments can be handled)

        func_params = func_def_node.args

        arg_names = [arg.arg for arg in func_params.args]
        kwonlyarg_names = [arg.arg for arg in func_params.kwonlyargs]

        default_nodes = self._process_arg_defaults(
            arg_names=arg_names,
            defaults=func_params.defaults
        )
        kw_default_nodes = self._process_arg_defaults(
            arg_names=kwonlyarg_names,
            defaults=func_params.kw_defaults
        )

        # Consolidate arg names
        if func_params.vararg:
            arg_names.append(func_params.vararg.arg)
        arg_names.extend(kwonlyarg_names)
        if func_params.kwarg:
            arg_names.append(func_params.kwarg.arg)

        # Update namespace with default values
        func_namespace = {**namespace, **default_nodes, **kw_default_nodes}

        for index, arg_token in enumerate(arg_vals):
            arg_node = self._process_connected_tokens(arg_token.arg)

            if not arg_node:
                continue

            if not arg_token.arg_name:
                arg_name = arg_names[index]
            else:
                arg_name = arg_token.arg_name

            namespace[arg_name] = arg_node

        # TODO (V1): Simply create ast.Assign objects for each default argument and
        # prepend them to func_def_node.body –– let the Saplings instance
        # below do the rest.

        # This handles recursive functions by deleting all aliases to the
        # function node
        func_def_aliases = []
        for alias, node in namespace.items():
            if node == func_def_node:
                func_def_aliases.append(alias)

        for alias in func_def_aliases:
            del namespace[alias]

        self._called_funcs.add(func_def_node)
        func_saplings_obj = self._run_child_saplings_instance(
            ast.Module(body=func_def_node.body),
            namespace
        )
        return func_saplings_obj._return_node

    def _process_module(self, module, standard_import=False):
        """
        Takes a module and searches the d-tree forest for a matching root node.
        If no match is found, new module nodes are generated and appended to the
        forest.

        Parameters
        ----------
        module : string
            identifier for a module, sometimes a period-separated string of
            sub-modules
        standard_import : bool
            flag for whether the module was imported normally or the result of
            `from X import Y`

        Returns
        -------
        utils.Node
            reference to the terminal d-tree node for the module
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
                term_node = matching_module
                break

        if not term_node:
            root_node = utils.Node(root_module)
            if standard_import:
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
                if standard_import:
                    self._node_lookup_table[sub_module_alias] = new_sub_module

                term_node.add_child(new_sub_module)
                term_node = new_sub_module

        return term_node

    def _delete_all_sub_aliases(self, targ_str):
        """
        If `my_var` is an alias, then `my_var()` and `my_var.attr` are
        considered sub-aliases. This function takes aliases that have been
        deleted or reassigned and deletes all of its sub-aliases.

        Parameters
        ----------
        targ_str : string
            string representation of the target node in the assignment
        """

        sub_aliases = []
        for alias in self._node_lookup_table.keys():
            for sub_alias_signifier in ('(', '.'):
                if alias.startswith(targ_str + sub_alias_signifier):
                    sub_aliases.append(alias)
                    break

        for alias in sub_aliases:
            del self._node_lookup_table[alias]

    def _process_assignment(self, target, value):
        """
        Handles variable assignments. There are three types of variable
        assignments that we consider:
            1. An alias for a known d-tree node being reassigned to another
               known d-tree node (K = K)
            2. An alias for a known d-tree node being reassigned to an unknown
               AST node (K = U)
            3. An unknown alias being assigned to a known d-tree node (U = K)
        For any one of these, the current namespace (self._node_lookup_table) is
        modified.

        Parameters
        ----------
        target : ast.AST
            node representing the left-hand-side of the assignment
        value : ast.AST
            node representing the right-hand-side of the assignment
        """

        tokenized_target = utils.recursively_tokenize_node(target, tokens=[])
        targ_node = self._process_connected_tokens(tokenized_target)

        tokenized_value = utils.recursively_tokenize_node(value, tokens=[])
        val_node = self._process_connected_tokens(tokenized_value)

        # TODO (V2): Handle assignments to data structures. For an assignment like
        # foo = [bar(i) for i in range(10)], foo.__index__() should be an alias
        # for bar().

        targ_str = utils.stringify_tokenized_nodes(tokenized_target)
        # val_str = utils.stringify_tokenized_nodes(tokenized_value)

        # Type I: Known node reassigned to other known node (K2 = K1)
        if targ_node and val_node:
            self._node_lookup_table[targ_str] = val_node
            self._delete_all_sub_aliases(targ_str)
        # Type II: Known node reassigned to unknown node (K1 = U1)
        elif targ_node and not val_node:
            del self._node_lookup_table[targ_str]
            self._delete_all_sub_aliases(targ_str)
        # Type III: Unknown node assigned to known node (U1 = K1)
        elif not targ_node and val_node:
            self._node_lookup_table[targ_str] = val_node

    def _process_connected_tokens(self, tokens, increment_count=False):
        """
        This is the master function for appendign to a d-tree. `tokens` is a
        list of Name and Args tokens, representing a "connected construct" ––
        meaning that, if the first token is an alias for a node in the d-tree,
        the next token is a child of that node. Here's an example of a
        connected construct:

        ```python
        module.attr.foo(module.bar[0] + 1, key=lambda x: x ** 2)
        ```

        This expression is an `ast.Call` node which, when passed into
        `utils.recursively_tokenize_node`, returns:

        ```python
        [
            NameToken("module"),
            NameToken("attr"),
            NameToken("foo"),
            ArgsToken([
                ArgToken([
                    NameToken("module"),
                    NameToken("bar"),
                    NameToken("__index__"),
                    ArgsToken([
                        ArgToken([ast.Num(0)])
                    ]),
                    NameToken("__add__"),
                    ArgsToken([
                        ArgToken([ast.Num(1)])
                    ])
                ]),
                ArgToken([ast.Lambda(...)], arg_name="key")
            ])
        ]
        ```

        (Note that the `ast.Num` and `ast.Lambda` nodes are not connected to the
        other tokens, and are represented as AST nodes. These nodes are
        processed by `self.generic_visit`.)

        When these tokens are passed into `_process_connected_tokens`, the
        following subtrees are created and added as children to the `module`
        d-tree: `attr -> foo -> ()` and
        `bar -> __index__ -> () -> __add__ -> ()`. And if `increment_count` is
        set to `True`, then the frequency value of the terminal nodes in these
        subtrees is incremented.

        Parameters
        ----------
        tokens : list
            list of tokenized AST nodes
        increment_count : bool

        Returns
        -------
        {utils.Node, None}
            reference to the terminal node in the subtree
        """

        # TODO (V1): Handle user-defined function calls from CLASSES

        func_def_types = (ast.FunctionDef, ast.AsyncFunctionDef) # TODO (V1): Handle ast.Lambda
        node_stack, func_def_node = [], None
        for index, token in enumerate(tokens):
            if isinstance(token, utils.ArgsToken):
                if func_def_node: # Evaluate the function call
                    if func_def_node in self._uncalled_funcs:
                        del self._uncalled_funcs[func_def_node]

                    return_node = self._process_user_defined_func(
                        func_def_node=func_def_node,
                        namespace=self._node_lookup_table.copy(),
                        arg_vals=token.args
                    ) # TODO (V1): Handle tuples

                    if isinstance(return_node, func_def_types):
                        # TODO (V1): Return nodes like these need to be processed in
                        # the namespace in which they were defined. For example:
                            # def foo():
                            #     x = module.call()
                            #
                            #     def bar(y):
                            #         return x + y
                            #
                            #     return bar
                            #
                            # b = foo()
                            # b(10)
                        # Your code won't capture module.call().__add__(), but
                        # it should.

                        func_def_node = return_node
                    else:
                        func_def_node = None

                    if return_node:
                        node_stack.append(return_node)

                    continue
                else:
                    for arg_token in token:
                        arg_node = self._process_connected_tokens(
                            tokens=arg_token.arg,
                            increment_count=increment_count
                        )
            elif not isinstance(token, utils.NameToken): # token is ast.AST node
                self.generic_visit(token)
                continue

            if func_def_node:
                func_def_node = None

            token_sequence = tokens[:index + 1]
            token_str = utils.stringify_tokenized_nodes(token_sequence)

            if token_str in self._node_lookup_table:
                node = self._node_lookup_table[token_str]

                if isinstance(node, func_def_types):
                    func_def_node = node
                    continue

                node_stack.append(node)
            elif node_stack: # Base node exists –– create and append its child
                child_node = utils.Node(str(token))
                node_stack[-1].add_child(child_node)
                self._node_lookup_table[token_str] = child_node
                node_stack.append(child_node)

        if func_def_node:
            return func_def_node
        elif not node_stack:
            return None

        return node_stack[-1]

    ## Aliasing Handlers ##

    def visit_Import(self, node):
        for module in node.names:
            if module.name.startswith('.'): # Ignores relative imports
                continue

            alias = module.asname if module.asname else module.name
            module_leaf_node = self._process_module(
                module=module.name,
                standard_import=not bool(module.asname)
            )

            self._node_lookup_table[alias] = module_leaf_node

    def visit_ImportFrom(self, node):
        if node.level: # Ignores relative imports
            return

        module_node = self._process_module(
            module=node.module,
            standard_import=False
        )

        for alias in node.names:
            if alias.name == '*': # Ignore star imports
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
        # TODO (V1): I think tuple assignment is broken. Fix it!

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
        target = node.target
        value = ast.BinOp(left=copy(target), op=node.op, right=node.value)
        self._process_assignment(target, value)

    def visit_Delete(self, node):
        for target in node.targets:
            target_tokens = utils.recursively_tokenize_node(target, [])
            target_str = utils.stringify_tokenized_nodes(target_tokens)

            self._delete_all_sub_aliases(target_str)

    ## Function and Class Definition Handlers ##

    def visit_ClassDef(self, node):
        pass # TODO (V1)

    def visit_FunctionDef(self, node):
        """
        Handles user-defined functions. When a user-defined function is called,
        it can return a module construct (i.e. a reference to a d-tree node).
        But as Python is dynamically typed, we don't know the return type until
        the function is called. Thus, we only traverse the function body and
        process those nodes when it's called and we know the types of the
        inputs. And if a user-defined function is never called, we process it at
        the end of the AST traversal and in the namespace it was defined in.

        The processing is done by self._process_user_defined_func. This function
        just aliases the node (i.e. saves it to self._node_lookup_table) and
        adds it to the self._uncalled_funcs table, along with the namespace it's
        defined in.

        Parameters
        ----------
        node : ast.FunctionDef
            name : raw string of the function name
            args : ast.arguments node
            body : list of nodes inside the function
            decorator_list : list of decorators to be applied
            returns : return annotation (Python 3 only)
            type_comment : string containing the PEP 484 type comment
        """

        # TODO (V2): Handle decorators

        self._uncalled_funcs[node] = self._node_lookup_table.copy()
        self._node_lookup_table[node.name] = node

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    # TODO (V1)
    # def visit_Lambda(self, node):
    #     pass

    def visit_Return(self, node):
        if node.value:
            tokenized_node = utils.recursively_tokenize_node(node.value, [])
            self._return_node = self._process_connected_tokens(tokenized_node)

        # TODO (V1): Handle the following:
            # def foo():
            #     if x:
            #         return y
            #     else:
            #         return z

    ## Control Flow ##

    # TODO: Delete this
    def _diff_namespaces(self, namespace):
        aliases_to_ignore = []
        for alias, node in self._node_lookup_table.items():
            if alias in namespace:
                if namespace[alias] == node:
                    continue
                else: # K1 = K2
                    aliases_to_ignore.append(alias)
            else: # K = U
                aliases_to_ignore.append(alias)

        return aliases_to_ignore

    def visit_If(self, node):
        self.generic_visit(node.test)

        for else_node in node.orelse:
            self._run_child_saplings_instance(
                tree=else_node,
                namespace=self._node_lookup_table.copy()
            )

        self.generic_visit(ast.Module(body=node.body))

    def visit_For(self, node):
        """
        Parameters
        ----------
        node : ast.For
            target : node holding variable(s) the loop assigns to
            iter : node holding item to be looped over
            body : list of nodes to execute
            orelse : list of nodes to execute (only executed if the loop
                     finishes normally, rather than via a break statement)
            type_comment : string containing the PEP 484 comment
        """

        # We treat the target as a subscript of iter
        target_assignment = ast.Assign(
            target=node.target,
            value=ast.Subscript(
                value=node.iter,
                slice=ast.Index(value=ast.NameConstant(None)),
                ctx=ast.Load()
            )
        )
        self.generic_visit(ast.Module(body=[target_assignment] + node.body))

        # TODO: Only run this if there's no `break` statement in the body
        self.generic_visit(ast.Module(body=node.orelse))

    def visit_AsyncFor(self, node):
        self.visit_For(node)

    def visit_While(self, node):
        # TODO

        body_saplings_obj = self._run_child_saplings_instance(
            ast.Module(body=node.body),
            self._node_lookup_table.copy()
        )

        # Only executed when the while condition becomes False. If you break out
        # of the loop or if an exeception is raised, it won't be executed.
        else_saplings_obj = self._run_child_saplings_instance(
            ast.Module(body=node.orelse),
            self._node_lookup_table.copy()
        )

    def visit_Try(self, node):
        body_saplings_obj = self._run_child_saplings_instance(
            ast.Module(body=node.body),
            self._node_lookup_table.copy()
        )

        for alias in self._diff_namespaces(body_saplings_obj._node_lookup_table):
            del self._node_lookup_table[alias]

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
            exception_node = self._process_connected_tokens(tokenized_exception)

            if node.name: # except A as B
                tokenized_alias = utils.recursively_tokenize_node(node.name, [])
                alias_str = utils.stringify_tokenized_nodes(tokenized_alias)

                if not exception_node:
                    del namespace[alias_str]
                else:
                    namespace[alias_str] = exception_node

        body_saplings_obj = self._run_child_saplings_instance(
            ast.Module(body=node.body),
            namespace
        )
        for alias in self._diff_namespaces(body_saplings_obj._node_lookup_table):
            del self._node_lookup_table[alias]

    def visit_withitem(self, node):
        if isinstance(node.optional_vars, ast.Name):
            self._process_assignment(
                target=node.optional_vars,
                value=node.context_expr
            )
        elif isinstance(node.optional_vars, ast.Tuple):
            pass # TODO (V1)
        elif isinstance(node.optional_vars, ast.List):
            pass # TODO (V1)

    ## Connected Construct Handlers ##

    @utils.connected_construct_handler
    def visit_Name(self, node):
        pass

    @utils.connected_construct_handler
    def visit_Attribute(self, node):
        pass

    @utils.connected_construct_handler
    def visit_Call(self, node):
        pass

    @utils.connected_construct_handler
    def visit_Subscript(self, node):
        pass

    @utils.connected_construct_handler
    def visit_BinOp(self, node):
        pass

    @utils.connected_construct_handler
    def visit_Compare(self, node):
        pass

    ## Data Structure Handlers ##

    # TODO (V2): Allow Type I assignments to dictionaries, lists, sets, and tuples.
    # Right now, assignments to data structures are treated as Type II. For
    # example, "attr" would not be captured in the following script:
    #   import module
    #   my_var = [module.func0(), module.func1(), module.func2()]
    #   my_var[0].attr

    def _comprehension_helper(self, elts, generators):
        namespace = self._node_lookup_table.copy()
        child_ifs = []
        for index, generator in enumerate(generators):
            iter_node = ast.Subscript(
                value=generator.iter,
                slice=ast.Index(value=ast.NameConstant(None)),
                ctx=ast.Load()
            ) # We treat the target as a subscript of iter
            iter_tokens = utils.recursively_tokenize_node(iter_node, [])

            if not index:
                iter_tree_node = self._process_connected_tokens(iter_tokens)
            else:
                iter_tree_node = self._run_child_saplings_instance(
                    ast.Module(body=[]),
                    namespace
                )._process_connected_tokens(iter_tokens)

            # TODO (V1): Handle when generator.target is ast.Tuple
            targ_tokens = utils.recursively_tokenize_node(generator.target, [])
            targ_str = utils.stringify_tokenized_nodes(targ_tokens)

            if iter_tree_node: # QUESTION: Needed? If so, might be needed elsewhere too
                namespace[targ_str] = iter_tree_node

            child_ifs.extend(generator.ifs)

        self._run_child_saplings_instance(
            ast.Module(body=child_ifs + elts),
            namespace
        )

    def visit_ListComp(self, node):
        self._comprehension_helper([node.elt], node.generators)

    def visit_SetComp(self, node):
        return self.visit_ListComp(node)

    def visit_GeneratorExp(self, node):
        return self.visit_ListComp(node)

    def visit_DictComp(self, node):
        self._comprehension_helper([node.key, node.value], node.generators)

    ## Miscellaneous ##

    # TODO (V1): Handle globals, nonlocals, lambdas, ifexps

    ## Public Methods ##

    def trees(self, flattened=False):
        trees = {} if flattened else []
        for tree in self.forest:
            if flattened:
                trees[tree.id] = tree.paths()
            else:
                trees.append(tree.to_dict(debug=True))

        return trees
