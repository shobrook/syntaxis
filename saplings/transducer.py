# Standard Library
import ast
from collections import defaultdict
from copy import copy

# Local
import utilities as utils


# QUESTION: Should frequency values be based on # of times evaluated, or # of
# times a construct appears in the code?

class Saplings(ast.NodeVisitor):
    def __init__(self, tree, hierarchies=[], namespace={}):
        """
        Extracts object hierarchies for imported modules in a program, given its
        AST.

        Parameters
        ----------
        tree : ast.AST
            the AST representation of the program to be analyzed
        hierarchies : {list, optional}
            root nodes of existing object hierarchies
        namespace : {dict, optional}
            mapping of identifiers to object/FunctionDef/ClassDef nodes
        """

        # Maps identifiers to object hierarchy nodes, function definition nodes,
        # and class definition nodes
        self._namespace = namespace

        # Maps ast.FuntionDef nodes to the namespaces in which they were defined
        # and a flag for whether they've been evaluated in the current scope
        self._func_state_lookup_table = {}

        # Holds FunctionDef nodes that were defined in a different scope but
        # called in the current scope
        self._called_but_undefined_funcs = {}

        # Object hierarchy node (or FuncDef/ClassDef) corresponding to the first
        # evaluated return statement in the AST
        self._return_node = None

        # Holds root nodes of object hierarchy trees
        self._hierarchies = hierarchies

        # Traverses the AST
        self.visit(tree)

        # If a function is defined but never called, we process the function
        # body after the traversal is over. The body is processed in the state
        # of the namespace in which it was defined.
        for func_def_node, data in self._func_state_lookup_table.items():
            if data["called"]:
                continue

            # Skip to handle currying, as this function may be called in the
            # parent scope
            if func_def_node == self._return_node:
                continue

            self._process_user_defined_func(
                func_def_node=func_def_node,
                namespace=data["namespace"]
            )

    ## Helpers ##

    def _process_sub_context(self, tree, namespace):
        """
        Wrapper function for processing a subtree in the AST with a different
        namespace from its parent (e.g. a function), or such that changes to the
        namespace in the subtree don't apply to the parent namespace (e.g. in
        the body of an if block).

        Parameters
        ----------
        tree:
        namespace:
        """

        print("\tEntering child Saplings instance\n") # TEMP
        child_scope = Saplings(
            tree=tree,
            hierarchies=self._hierarchies,
            namespace=namespace
        )

        # Pushes any called functions in the child scope up to the parent scope
        for func_def_node in child_scope._called_but_undefined_funcs:
            if func_def_node in self._func_state_lookup_table:
                self._func_state_lookup_table[func_def_node]["called"] = True
            else:
                self._called_but_undefined_funcs.add(func_def_node)

        print("\tPopping out of Saplings instance\n") # TEMP
        return child_scope

    ## Processors ##

    def _process_arg_defaults(self, arg_names, defaults, namespace):
        """
        Takes arguments from a user-defined function's signature and processes
        their default values.

        Parameters
        ----------
        args : list
            list of argument names (strings)
        defaults : list
            list of default values for the args (ast.AST nodes)
        namespace : dict
            namespace in which the function with the defaults was defined

        Returns
        -------
        dict
            mapping of argument names to their default value's OH node
        list
            list of names of arguments whose defaults had no corresponding
            OH node
        """

        num_args, num_defaults = len(arg_names), len(defaults)
        arg_to_default_node, null_defaults = {}, []
        for index, default in enumerate(defaults):
            if not default: # Only kw_defaults can be None
                continue

            arg_name_index = index + (num_args - num_defaults)
            arg_name = arg_names[arg_name_index]

            default_node = self._process_sub_context(
                tree=ast.Module(body=[]),
                namespace=namespace
            )._process_connected_tokens(default)

            if not default_node:
                null_defaults.append(arg_name)
                continue

            arg_to_default_node[arg_name] = default_node

        return arg_to_default_node, null_defaults

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

        # TODO (V2): Handle single-star args

        func_params = func_def_node.args

        arg_names = [arg.arg for arg in func_params.args]
        kwonlyarg_names = [arg.arg for arg in func_params.kwonlyargs]

        # Object nodes corresponding to default values
        default_nodes, null_defaults = self._process_arg_defaults(
            arg_names=arg_names,
            defaults=func_params.defaults,
            namespace=namespace
        )
        kw_default_nodes, null_kw_defaults = self._process_arg_defaults(
            arg_names=kwonlyarg_names,
            defaults=func_params.kw_defaults,
            namespace=namespace
        )

        # Consolidate arg names
        if func_params.vararg:
            arg_names.append(func_params.vararg.arg)
        arg_names.extend(kwonlyarg_names)
        if func_params.kwarg:
            arg_names.append(func_params.kwarg.arg)

        # Update namespace with default values
        func_namespace = {**namespace, **default_nodes, **kw_default_nodes}
        for null_arg_name in null_defaults + null_kw_defaults:
            if null_arg_name in func_namespace:
                del func_namespace[null_arg_name]

        # Iterates through arguments in order they were passed into function
        for index, arg_token in enumerate(arg_vals):
            if not arg_token.arg_name:
                arg_name = arg_names[index]
            else:
                arg_name = arg_token.arg_name

            arg_node = self._process_connected_tokens(arg_token.arg)
            if not arg_node:
                if arg_name in func_namespace:
                    del func_namespace[arg_name]

                continue

            func_namespace[arg_name] = arg_node

        # TODO (V2): Simply create ast.Assign objects for each default argument
        # and prepend them to func_def_node.body –– let the Saplings instance
        # below do the rest. Not necessary but it would clean up your code.

        # Handles recursive functions by deleting all names of the function node
        for alias, node in list(func_namespace.items()):
            if node == func_def_node:
                del func_namespace[alias]

        func_saplings_obj = self._process_sub_context(
            ast.Module(body=func_def_node.body),
            func_namespace
        )
        # TODO (V1): Handle case where function was defined in the parent
        # context and therefore isn't in the lookup table
        if func_def_node in self._func_state_lookup_table:
            self._func_state_lookup_table[func_def_node]["called"] = True
        else:
            self._called_but_undefined_funcs.add(func_def_node)

        # Handles closures by adding returned function to
        # _func_state_lookup_table
        return_node = func_saplings_obj._return_node
        if isinstance(return_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._func_state_lookup_table[return_node] = {
                **func_saplings_obj._func_state_lookup_table[return_node],
                **{"called": False, "is_closure": True}
            }

        return return_node

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

        for root in self._hierarchies:
            matching_module = find_matching_node(root, root_module)

            if matching_module:
                term_node = matching_module
                break

        if not term_node:
            root_node = utils.Node(root_module)
            if standard_import:
                self._namespace[root_module] = root_node

            term_node = root_node
            self._hierarchies.append(term_node)

        for idx in range(len(sub_modules[1:])):
            sub_module = sub_modules[idx + 1]
            sub_module_alias = '.'.join([root_module] + sub_modules[1:idx + 2])

            matching_sub_module = find_matching_node(term_node, sub_module)

            if matching_sub_module:
                term_node = matching_sub_module
            else:
                new_sub_module = utils.Node(sub_module)
                if standard_import:
                    self._namespace[sub_module_alias] = new_sub_module

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

        # TODO (V1): Confirm this is right

        sub_aliases = []
        for alias in self._namespace.keys():
            for sub_alias_signifier in ('(', '.'):
                if alias.startswith(targ_str + sub_alias_signifier):
                    sub_aliases.append(alias)
                    break

        for alias in sub_aliases:
            del self._namespace[alias]

    def _process_assignment(self, target, value):
        """
        Handles variable assignments. There are three types of variable
        assignments that we consider:
            1. An alias for a known d-tree node being reassigned to another
               known d-tree node (K = K)
            2. An alias for a known d-tree node being reassigned to an unknown
               AST node (K = U)
            3. An unknown alias being assigned to a known d-tree node (U = K)
        For any one of these, the current namespace is modified.

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

        # TODO (V2): Handle assignments to data structures. For an assignment
        # like foo = [bar(i) for i in range(10)], foo.__index__() should be an
        # alias for bar().

        targ_str = utils.stringify_tokenized_nodes(tokenized_target)
        # val_str = utils.stringify_tokenized_nodes(tokenized_value)

        # Type I: Known node reassigned to other known node (K2 = K1)
        if targ_node and val_node:
            self._namespace[targ_str] = val_node
            self._delete_all_sub_aliases(targ_str)
        # Type II: Known node reassigned to unknown node (K1 = U1)
        elif targ_node and not val_node:
            del self._namespace[targ_str]
            self._delete_all_sub_aliases(targ_str)
        # Type III: Unknown node assigned to known node (U1 = K1)
        elif not targ_node and val_node:
            self._namespace[targ_str] = val_node

    def _process_connected_tokens(self, tokens, increment_count=False):
        """
        This is the master function for appending to an object hierarchy.
        `tokens` is a list of Name and Args tokens, representing a "connected
        construct" –– meaning that, if the first token is an identifier for an
        object node, the next token is a child of that node. Here's an example of
        a connected construct:

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

        Note that the `ast.Num` and `ast.Lambda` nodes are not connected to the
        other tokens, and are represented as AST nodes. These nodes are
        processed by `self.visit`.

        When these tokens are passed into `_process_connected_tokens`, the
        following subtrees are created and added as children to the `module`
        hierarchy: `attr -> foo -> ()` and
        `bar -> __index__ -> () -> __add__ -> ()`. And if `increment_count` is
        set to `True`, then the frequency value of the terminal nodes in these
        subtrees is incremented.

        Parameters
        ----------
        tokens : list
            list of tokenized AST nodes
        increment_count : bool
            whether or not to increment the frequency value of nodes

        Returns
        -------
        {utils.Node, None}
            reference to the terminal node in the subtree
        """

        # TODO (V1): Handle user-defined function calls from CLASSES

        def handle_user_defined_func_call(func_def_node, token):
            adjusted_namespace = self._namespace.copy()

            # If not, then function was defined in the parent scope
            if func_def_node in self._func_state_lookup_table:
                func_state = self._func_state_lookup_table[func_def_node]
                if func_state["is_closure"]:
                    adjusted_namespace = {
                        **self._namespace.copy(),
                        **func_state["namespace"]
                    }

            return_node = self._process_user_defined_func(
                func_def_node,
                adjusted_namespace,
                token.args
            ) # TODO (V1): Handle tuple returns

            return return_node

        func_def_types = (ast.FunctionDef, ast.AsyncFunctionDef)
        node_stack, curr_func_def_node = [], None
        for index, token in enumerate(tokens):
            if isinstance(token, utils.ArgsToken):
                if curr_func_def_node: # Evaluate call of user-defined function
                    return_node = handle_user_defined_func_call(
                        curr_func_def_node,
                        token
                    )

                    if isinstance(return_node, func_def_types):
                        curr_func_def_node = return_node
                    else:
                        curr_func_def_node = None

                    if isinstance(return_node, utils.Node):
                        node_stack.append(return_node)

                    continue
                else:
                    for arg_token in token:
                        arg_node = self._process_connected_tokens(
                            arg_token.arg,
                            increment_count
                        )
            elif not isinstance(token, utils.NameToken): # token is ast.AST node
                self.visit(token) # TODO (V2): Handle lambdas
                continue

            if curr_func_def_node: # TODO (V1): Check if this is ever hit
                curr_func_def_node = None

            token_sequence = tokens[:index + 1]
            token_str = utils.stringify_tokenized_nodes(token_sequence)

            if token_str in self._namespace:
                node = self._namespace[token_str]

                if isinstance(node, func_def_types):
                    curr_func_def_node = node
                    continue

                node_stack.append(node)
            elif node_stack: # Base node exists –– create and append its child
                child_node = utils.Node(str(token))
                node_stack[-1].add_child(child_node)
                self._namespace[token_str] = child_node
                node_stack.append(child_node)

        if curr_func_def_node:
            return curr_func_def_node
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

            self._namespace[alias] = module_leaf_node

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
                    self._namespace[alias_id] = child

                    break

            if not child_exists:
                new_child = utils.Node(alias.name)
                self._namespace[alias_id] = new_child

                module_node.add_child(new_child)

    def visit_Assign(self, node):
        # TODO (V1): I think tuple assignment is broken. Fix it!

        values = node.value
        targets = node.targets if hasattr(node, "targets") else (node.target,)
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

    ## Function and Class Handlers ##

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

        This processing is done by self._process_user_defined_func. All this
        visitor does is alias the node (i.e. saves it to self._namespace)
        and adds it to the self._func_state_lookup_table, along with a copy of
        the namespace in the state the function is defined in.

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

        # NOTE: namespace is only used if the function is never called or if its
        # a closure
        self._func_state_lookup_table[node] = {
            "namespace": self._namespace.copy(),
            "called": False,
            "is_closure": False
        }
        self._namespace[node.name] = node

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_Lambda(self, node):
        namespace = self._namespace.copy()
        args = node.args.args + node.args.kwonlyargs
        if node.args.vararg:
            args += [node.args.vararg]
        if node.args.kwarg:
            args += [node.args.kwarg]

        # node.args.default
        for arg in args:
            arg_name = arg.arg
            if arg_name not in namespace:
                continue

            del namespace[arg_name]

        self._process_sub_context(node.body, namespace)

        # TODO (V2): Handle assignments to lambdas and lambda function calls

    def visit_Return(self, node):
        if node.value:
            tokenized_node = utils.recursively_tokenize_node(node.value, [])
            self._return_node = self._process_connected_tokens(tokenized_node)

        # TODO (V1): Stop processing of everything under return statement

    ## Control Flow Handlers ##

    def visit_If(self, node):
        """
        Namespace changes in the first `If` block persist into the parent
        context, but changes made in `Elif` or `Else` blocks do not.

        Parameters
        ----------
        node : ast.If
            test ...
            body ...
            orelse ...
        """
        self.visit(node.test)

        for else_node in node.orelse:
            self._process_sub_context(
                tree=else_node,
                namespace=self._namespace.copy()
            )

        self.visit(ast.Module(body=node.body))

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
        self.visit(ast.Module(body=[target_assignment] + node.body))

        # TODO (V1): Only run this if there's no `break` statement in the body
        self.visit(ast.Module(body=node.orelse))

    def visit_AsyncFor(self, node):
        self.visit_For(node)

    def visit_While(self, node):
        self.visit(ast.Module(body=node.body))

        # TODO (V1): Only run this if there's no `break` statement in the body
        self.visit(ast.Module(body=node.orelse))

    def visit_Try(self, node):
        for except_handler_node in node.handlers:
            self.visit_ExceptHandler(except_handler_node)

        self.visit(ast.Module(body=node.body))

        # Executes only if node.body doesn't throw an exception
        for else_node in node.orelse:
            self.visit_If(else_node)

        # node.finalbody is executed no matter what
        self.visit(ast.Module(body=node.finalbody))

    def visit_ExceptHandler(self, node):
        namespace = self._namespace.copy()

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

        body_saplings_obj = self._process_sub_context(
            ast.Module(body=node.body),
            namespace
        )

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

    def visit_Continue(self, node):
        pass # TODO (V1)

    def visit_Break(self, node):
        pass # TODO (V1)

    ## Construct Handlers ##

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

    # TODO (V2): Allow Type I assignments to dictionaries, lists, sets, and
    # tuples. Right now, assignments to data structures are treated as Type II.
    # For example, "attr" would not be captured in the following script:
    #   import module
    #   my_var = [module.func0(), module.func1(), module.func2()]
    #   my_var[0].attr

    def _comprehension_helper(self, elts, generators):
        namespace = self._namespace.copy()
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
                iter_tree_node = self._process_sub_context(
                    ast.Module(body=[]),
                    namespace
                )._process_connected_tokens(iter_tokens)

            # TODO (V1): Handle when generator.target is ast.Tuple
            targ_tokens = utils.recursively_tokenize_node(generator.target, [])
            targ_str = utils.stringify_tokenized_nodes(targ_tokens)

            if iter_tree_node: # QUESTION: Needed? If so, might be needed elsewhere too
                namespace[targ_str] = iter_tree_node

            child_ifs.extend(generator.ifs)

        self._process_sub_context(
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

    # TODO (V1): Handle globals, ifexps

    ## Public Methods ##

    def trees(self, flattened=False):
        trees = {} if flattened else []
        for tree in self._hierarchies:
            if flattened:
                trees[tree.id] = tree.paths()
            else:
                trees.append(tree.to_dict(debug=True))

        return trees
