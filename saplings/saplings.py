# Standard Library
import ast
from collections import defaultdict
from copy import copy

# Local Modules
import saplings.utilities as utils
import saplings.tokenization as tkn
from saplings.entities import ObjectNode, Function, Class, ClassInstance
# import utilities as utils
# import tokenization as tkn
# from entities import ObjectNode, Function, Class, ClassInstance


##########
# SAPLINGS
##########


class Saplings(ast.NodeVisitor):
    def __init__(self, tree, object_hierarchies=[], namespace={}, track_modules=True):
        """
        Extracts object hierarchies for imported modules in a program, given its
        AST.

        Parameters
        ----------
        tree : ast.AST
            the AST representation of the program to be analyzed
        object_hierarchies : {list, optional}
            root nodes of existing object hierarchies
        namespace : {dict, optional}
            mapping of identifiers to ObjectNodes/Functions/Classes/ClassInstances
        """

        self._object_hierarchies = [obj_node for obj_node in object_hierarchies]
        self._track_modules = track_modules

        # Maps active identifiers to namespace entities (e.g. ObjectNodes,
        # Functions, Classes, and ClassInstances)
        self._namespace = {name: entity for name, entity in namespace.items()}

        # Keeps track of functions defined in the current scope
        self._functions = set()

        # Namespace entity produced by the first evaluated return statement in
        # the AST
        self._return_value = None

        # True when a Return, Continue, or Break node is hit –– stops traversal
        # of tree
        self._is_traversal_halted = False

        self.visit(tree)
        self._process_uncalled_functions()

    ## Overloaded Methods ##

    def visit(self, node):
        """
        Overloaded AST visitor. Behaves the same as ast.NodeVisitor.visit except
        it only traverses a subtree when self._is_traversal_halted == False.

        Parameters
        ----------
        node : ast.AST
            root node of tree/subtree to be traversed

        Returns
        -------
        output (if any) of overloaded node visitor functions
        """

        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)

        if not self._is_traversal_halted:
            return visitor(node)

    ## Helpers ##

    def _process_uncalled_functions(self):
        """
        Processes uncalled functions. If a function is defined but never called,
        we process the function body after the traversal is over and in the
        state of the namespace in which it was defined.
        """

        # Don't process closures, as they may be called in a different scope
        returns_closure = isinstance(self._return_value, Function)
        if returns_closure and self._return_value in self._functions:
            self._functions.remove(self._return_value)

        while any(not f.called for f in self._functions):
            for function in self._functions.copy():
                if function.called:
                    self._functions.remove(function)
                    continue

                self._process_function(function, function.init_namespace)

    def _process_subtree_in_new_scope(self, tree, namespace):
        """
        Used to process a subtree in a different scope/namespace from the
        current scope. Can act as a "sandbox," where changes to the namespace
        don't affect the containing scope.

        Parameters
        ----------
        tree : ast.AST
            root node of the subtree to process
        namespace : dict
            namespace within which the subtree should be processed

        Returns
        -------
        Saplings
            instance of a Saplings object
        """

        return Saplings(tree, self._object_hierarchies, namespace, self._track_modules)

    def _process_node(self, node):
        """
        Processes an AST node. Processing involves tokenizing the node and then
        feeding it into _process_attribute_chain.

        Parameters
        ----------
        node : ast.AST
            node to be processed

        Returns
        -------
        list
            tokenized node
        {ObjectNode, Function, Class, ClassInstance, None}
            namespace entity produced by the node / attribute chain
        dict
            dictionary containing the class instance context (if any) the
            node was processed in
        """

        tokenized_node = tkn.recursively_tokenize_node(node, [])
        entity, instance = self._process_attribute_chain(tokenized_node)

        return tokenized_node, entity, instance

    def _break_and_process_nested_chains(self, tokens, current_entity, current_instance):
        """
        TODO
        """

        for token in tokens:
            if not isinstance(token, tkn.CallToken):
                continue

            for arg_token in token:
                self._process_attribute_chain(arg_token.arg_val)

        current_entity = None
        current_instance["entity"] = None
        current_instance["init_index"] = 0

    def _process_function_call(self, function, arguments):
        """
        TODO
        """

        if function.is_closure:
            func_namespace = {
                **self._namespace.copy(),
                **function.init_namespace
            }
        else:
            func_namespace = self._namespace.copy()

        return_value, _ = self._process_function(
            function,
            func_namespace,
            arguments
        ) # TODO (V2): Handle tuple returns (blocked by data structure handling)

        if isinstance(return_value, ObjectNode):
            return_value.increment_count()

        return return_value

    def _bind_entity_to_arguments(self, entity, arguments):
        """
        TODO
        """

        hidden_arg = tkn.ArgToken([tkn.NameToken("")])
        self._namespace[""] = entity

        return [hidden_arg] + arguments

    def _process_method_call(self, function, arguments, class_instance=None):
        """
        TODO
        """

        if class_instance: # Method is called by instance
            if function.method_type == "class":
                arguments = self._bind_entity_to_arguments(
                    class_instance.class_entity,
                    arguments
                )
            elif function.method_type == "instance":
                arguments = self._bind_entity_to_arguments(
                    class_instance,
                    arguments
                )
        else: # Method is called by class
            arguments = self._bind_entity_to_arguments(
                function.containing_class,
                arguments
            )

        return_value = self._process_function_call(function, arguments)

        if "" in self._namespace:
            del self._namespace[""]

        return return_value

    ## Processors ##

    def _process_module(self, module, standard_import=False):
        """
        Takes a module and searches the set of object hierarchies for a
        matching root node. If no match is found, new root nodes are generated
        and appended to the set.

        Parameters
        ----------
        module : string
            identifier for a module, sometimes a period-separated string of
            sub-modules
        standard_import : bool
            indicates whether the module was imported normally or the result of
            `from X import Y`

        Returns
        -------
        ObjectNode
            terminal object hierarchy node for the module
        """

        sub_modules = module.split('.') # For module.submodule1.submodule2...
        root_module = sub_modules[0]
        term_node = None

        for root in self._object_hierarchies:
            matching_module = utils.find_matching_node(root, root_module)

            if matching_module:
                term_node = matching_module
                break

        if not term_node:
            root_node = ObjectNode(root_module, order=-1)
            if standard_import:
                self._namespace[root_module] = root_node

            term_node = root_node
            self._object_hierarchies.append(term_node)

        for index in range(len(sub_modules[1:])):
            sub_module = sub_modules[index + 1]
            sub_module_alias = '.'.join([root_module] + sub_modules[1:index + 2])

            matching_sub_module = utils.find_matching_node(term_node, sub_module)

            if matching_sub_module:
                term_node = matching_sub_module
            else:
                new_sub_module = ObjectNode(sub_module)
                if standard_import:
                    self._namespace[sub_module_alias] = new_sub_module

                term_node.add_child(new_sub_module)
                term_node = new_sub_module

        return term_node

    def _process_default_args(self, arg_names, defaults, namespace):
        """
        Takes arguments from a user-defined function's signature and processes
        their default values.

        Parameters
        ----------
        arg_names : list
            list of argument names (strings)
        defaults : list
            list of default values for the args (ast.AST nodes)
        namespace : dict
            namespace in which the function with the defaults was defined

        Returns
        -------
        dict
            map of argument names to their default value's namespace entity
        list
            list of argument names whose defaults had no corresponding
            namespace entity
        """

        num_args, num_defaults = len(arg_names), len(defaults)
        arg_to_default_node, null_defaults = {}, []
        for index, default in enumerate(defaults):
            if not default: # Only kw_defaults can be None
                continue

            arg_name_index = index + (num_args - num_defaults)
            arg_name = arg_names[arg_name_index]

            tokenized_default = tkn.recursively_tokenize_node(default, [])
            default_node, _ = self._process_subtree_in_new_scope(
                ast.Module(body=[]),
                namespace
            )._process_attribute_chain(tokenized_default)

            if not default_node:
                null_defaults.append(arg_name)
                continue

            arg_to_default_node[arg_name] = default_node

        return arg_to_default_node, null_defaults

    def _process_function(self, function, namespace, arguments=[]):
        """
        Processes the arguments and body of a user-defined function. If the
        function is recursive, the recursive calls are not processed (otherwise
        this would throw `Saplings` into an infinite loop). If the function
        returns a closure, that function is added to the list of functions in
        the current scope.

        Parameters
        ----------
        function : Function
            function that's being called
        namespace : dict
            namespace within which the function should be processed
        arguments : list, optional
            arguments passed into the function when called (as a list of
            ArgTokens)

        Returns
        -------
        {ObjectNode, Function, Class, ClassInstance, None}
            namespace entity corresponding to the return value of the function;
            None if the function has no return value or returns something we
            don't care about (i.e. something that isn't tracked in the
            namespace)
        """

        parameters = function.def_node.args

        pos_params = [a.arg for a in parameters.args]
        kw_params = [a.arg for a in parameters.kwonlyargs]

        # Namespace entities corresponding to default values
        default_entities, null_defaults = self._process_default_args(
            pos_params,
            parameters.defaults,
            namespace
        )
        kw_default_entities, null_kw_defaults = self._process_default_args(
            kw_params,
            parameters.kw_defaults,
            namespace
        )

        # Update namespace with default values
        namespace = {**namespace, **default_entities, **kw_default_entities}
        for null_arg_name in null_defaults + null_kw_defaults:
            if null_arg_name in namespace:
                del namespace[null_arg_name]
                utils.delete_sub_aliases(null_arg_name, namespace)

        for index, argument in enumerate(arguments):
            if argument.arg_name == '': # Positional argument
                if index < len(pos_params):
                    arg_name = pos_params[index]
                else: # *arg
                    self._process_attribute_chain(argument.arg_val)
                    continue
            elif argument.arg_name is not None: # Keyword argument
                arg_name = argument.arg_name
                if arg_name not in pos_params + kw_params: # **kwargs
                    self._process_attribute_chain(argument.arg_val)
                    continue
            else: # **kwargs
                self._process_attribute_chain(argument.arg_val)
                continue

            arg_entity, _ = self._process_attribute_chain(argument.arg_val)
            if not arg_entity:
                if arg_name in namespace:
                    del namespace[arg_name]
                    utils.delete_sub_aliases(arg_name, namespace)

                continue

            utils.delete_sub_aliases(arg_name, namespace)
            namespace[arg_name] = arg_entity

        # TODO (V2): Handle star args and **kwargs (blocked by data structure
        # handling)
        if parameters.vararg and parameters.vararg.arg in namespace:
            del namespace[parameters.vararg.arg]
            utils.delete_sub_aliases(parameters.vararg.arg, namespace)
        if parameters.kwarg and parameters.kwarg.arg in namespace:
            del namespace[parameters.kwarg.arg]
            utils.delete_sub_aliases(parameters.kwarg.arg, namespace)

        # Handles recursive functions by deleting all names of the function node
        for name, node in list(namespace.items()):
            if node == function:
                del namespace[name]

        # Processes function body
        func_saplings = self._process_subtree_in_new_scope(
            ast.Module(body=function.def_node.body),
            namespace
        )
        function.called = True
        return_value = func_saplings._return_value

        # If the function returns a closure then treat it like a function
        # defined in the current scope by adding it to self._functions
        if isinstance(return_value, Function):
            return_value.is_closure = True
            # TODO (V1): What if the function is defined in an outer scope but
            # returned in this scope? Then it's not a closure. Handle these.

            if not return_value.called:
                self._functions.add(return_value)
        elif isinstance(return_value, Class):
            for name, entity in return_value.init_instance_namespace.items():
                if isinstance(entity, Function):
                    entity.is_closure = True
                    if not entity.called:
                        self._functions.add(entity)

        return return_value, func_saplings

    def _process_assignment(self, target, val_entity):
        """
        Handles variable assignments and aliasing. There are three types of
        assignments that we consider:
            1. An identifier for an active namespace entity being reassigned
               to another active entity
            2. An identifier for an active namespace entity being reassigned
               to a non-entity (i.e. AST node)
            3. An identifier for a non-entity being assigned to an active
               namespace entity
        For any one of these, the current namespace is modified.

        Parameters
        ----------
        target : ast.AST
            node representing the left-hand-side of the assignment
        val_entity : {ObjectNode, Function, Class, ClassInstance, None}
            namespace entity corresponding to the right-hand-side of the
            assignment
        """

        tokenized_target, targ_entity, instance = self._process_node(target)

        # TODO (V2): Handle assignments to data structures. For an assignment
        # like foo = [bar(i) for i in range(10)], foo.__index__() should be an
        # alias for bar().

        # TODO (V2): Handle assignments to class variables that propagate to
        # class instances (e.g. MyClass.variable = ...; my_instance.variable.foo())
        if instance["entity"]:
            namespace = instance["entity"].namespace
            targ_str = tkn.stringify_tokenized_nodes(tokenized_target[instance["init_index"] + 1:])
        else:
            namespace = self._namespace
            targ_str = tkn.stringify_tokenized_nodes(tokenized_target)

        # Type I: Known entity reassigned to other known entity (E2 = E1)
        if targ_entity and val_entity:
            namespace[targ_str] = val_entity
            utils.delete_sub_aliases(targ_str, namespace)
        # Type II: Known entity reassigned to non-entity (E1 = NE1)
        elif targ_entity and not val_entity:
            del namespace[targ_str]
            utils.delete_sub_aliases(targ_str, namespace)
        # Type III: Non-entity assigned to known entity (NE1 = E1)
        elif not targ_entity and val_entity:
            namespace[targ_str] = val_entity

    def _process_attribute_chain(self, attribute_chain):
        """
        Master function for processing attribute chains. An attribute chain is a
        list of `Name` and `Call` tokens such that each `Name` token is an
        (n + 1)th-order attribute of the nearest prior `Name` token, where n is
        the number of `Call` tokens separating the two. For example:

        ```python
        module.bar(my.var, my_func).lorem[0] + ipsum
        ```

        This expression is an `ast.BinOp` node which, when passed into
        `recursively_tokenize_node`, produces the following attribute chain:

        ```python
        [
            NameToken("module"),
            NameToken("bar"),
            CallToken([
                ArgToken([NameToken("my"), NameToken("var")]),
                ArgToken([NameToken("my_func")])
            ]),
            NameToken("lorem")
            NameToken("__index__"),
            CallToken([
                ArgToken([ast.Num(0)])
            ]),
            NameToken("__add__"),
            CallToken([
                ArgToken([NameToken("ipsum")])
            ])
        ]
        ```

        The attribute chain is given to this function as `attribute_chain`,
        which checks the namespace for entities referenced in the chain ...

        TODO

        Parameters
        ----------
        tokens : list
            list of `NameToken`s and `CallToken`s

        Returns
        -------
        {ObjectNode, Function, Class, ClassInstance}, context_dict
        """

        current_entity = None
        current_instance = {"entity": None, "init_index": 0}
        for index, token in enumerate(attribute_chain):
            if index and not current_entity:
                self._break_and_process_nested_chains(
                    attribute_chain[index:],
                    current_entity,
                    current_instance
                )
                break

            if isinstance(token, tkn.CallToken):
                if isinstance(current_entity, Function):
                    if current_instance["entity"]:
                        # Process call of function from instance of a
                        # user-defined class
                        current_entity = self._process_method_call(
                            current_entity,
                            token.args,
                            current_instance["entity"]
                        )
                    else:
                        if current_entity.method_type == "class":
                            # Process call of function from user-defined class
                            current_entity = self._process_method_call(
                                current_entity,
                                token.args
                            )
                        else:
                            # Process call of user-defined function that is
                            # either unbound to a class or a static method
                            current_entity = self._process_function_call(
                                current_entity,
                                token.args
                            )

                    if isinstance(current_entity, ClassInstance):
                        current_instance["entity"] = current_entity
                        current_instance["init_index"] = index

                    continue
                elif isinstance(current_entity, Class):
                    # Process instantiation of user-defined class

                    init_namespace = current_entity.init_instance_namespace
                    class_instance = ClassInstance(
                        current_entity,
                        init_namespace.copy()
                    )

                    if "__init__" in init_namespace:
                        constructor = init_namespace["__init__"]
                        if isinstance(constructor, Function):
                            self._process_method_call(
                                constructor,
                                token.args,
                                class_instance
                            )
                        else:
                            # If __init__ is not callable, class cannot be
                            # instantiated
                            self._break_and_process_nested_chains(
                                attribute_chain[index + 1:],
                                current_entity,
                                current_instance
                            )
                            break

                            # BUG: __init__ may be a lambda function or an
                            # ObjectNode (e.g. __init__ = module.imported_init)
                    else:
                        # BUG: If __init__ is defined in the base class then
                        # it's a black box and may make unknown changes to the
                        # instance namespace
                        pass

                    current_entity = class_instance
                    current_instance["entity"] = class_instance
                    current_instance["init_index"] = index

                    continue
                elif isinstance(current_entity, ClassInstance):
                    # Process call of instance of user-defined class
                    if "__call__" in current_entity.namespace:
                        call_entity = current_entity.namespace["__call__"]

                        # BUG: __call__ may be a lambda function or an
                        # ObjectNode (e.g. __call__ = module.imported_call)
                        if isinstance(call_entity, Function):
                            current_entity = self._process_method_call(
                                call_entity,
                                token.args,
                                current_entity
                            )

                            if isinstance(current_entity, ClassInstance):
                                current_instance["entity"] = current_entity
                                current_instance["init_index"] = index

                            continue

                    self._break_and_process_nested_chains(
                        attribute_chain[index + 1:],
                        current_entity,
                        current_instance
                    )
                    break

                    # BUG: If __call__ is defined in the base class then
                    # breaking could produce false negatives
                else:
                    for arg_token in token:
                        self._process_attribute_chain(arg_token.arg_val)
            elif not isinstance(token, tkn.NameToken): # token is ast.AST node
                self.visit(token)

                # TODO (V1): Handle IfExps and Lambdas (e.g. (lambda x: x.attr)(module.foo))

                self._break_and_process_nested_chains(
                    attribute_chain[index + 1:],
                    current_entity,
                    current_instance
                )
                break

            if current_instance["entity"]:
                namespace = current_instance["entity"].namespace
                token_seq = attribute_chain[current_instance["init_index"] + 1:index + 1]
            else:
                namespace = self._namespace
                token_seq = attribute_chain[:index + 1]

            token_str = tkn.stringify_tokenized_nodes(token_seq)
            if token_str in namespace:
                current_entity = namespace[token_str]
                if isinstance(current_entity, ClassInstance):
                    current_instance["entity"] = current_entity
                    current_instance["init_index"] = index
                elif isinstance(current_entity, ObjectNode):
                    current_entity.increment_count()
            elif isinstance(current_entity, ObjectNode):
                # Base node exists –– create and append its child
                current_entity = current_entity.add_child(ObjectNode(str(token)))
                namespace[token_str] = current_entity
            else:
                current_entity = None

        last_token_is_instance = current_instance["init_index"] == len(attribute_chain) - 1
        if isinstance(current_entity, ClassInstance) and last_token_is_instance:
            current_instance = {"entity": None, "init_index": 0}

        return current_entity, current_instance

    ## Aliasing Handlers ##

    def visit_Import(self, node):
        """
        TODO
        """

        if not self._track_modules:
            return

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
        """
        TODO
        """

        if not self._track_modules:
            return

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
                if alias.name == child.name:
                    child_exists = True
                    self._namespace[alias_id] = child

                    break

            if not child_exists:
                new_child = ObjectNode(alias.name)
                self._namespace[alias_id] = new_child

                module_node.add_child(new_child)

    def visit_Assign(self, node):
        """
        TODO
        """

        if isinstance(node.value, ast.Tuple):
            values = []
            for value in node.value.elts:
                _, val_entity, _ = self._process_node(value)
                values.append(val_entity)
        else:
            _, values, _ = self._process_node(node.value)

        targets = node.targets if hasattr(node, "targets") else (node.target,)
        for target in targets: # Multiple assignment (e.g. a = b = ...)
            if isinstance(target, ast.Tuple): # Unpacking (e.g. a, b = ...)
                for index, elt in enumerate(target.elts):
                    if isinstance(values, list):
                        self._process_assignment(elt, values[index])
                    else:
                        self._process_assignment(elt, values)
            elif isinstance(values, list):
                for value in values:
                    self._process_assignment(target, value)
            else:
                self._process_assignment(target, values)

    def visit_AnnAssign(self, node):
        self.visit_Assign(node)

    def visit_AugAssign(self, node):
        """
        TODO
        """

        target = node.target
        value = ast.BinOp(left=copy(target), op=node.op, right=node.value)
        _, val_entity, _ = self._process_node(value)
        self._process_assignment(target, val_entity)

    def visit_Delete(self, node):
        """
        TODO
        """

        for target in node.targets:
            target_str = utils.stringify_node(target)
            utils.delete_sub_aliases(target_str, self._namespace)

    ## Function and Class Handlers ##

    def visit_FunctionDef(self, node):
        """
        Handles user-defined functions. When a user-defined function is called,
        it can return a module construct (i.e. a reference to a d-tree node).
        But as Python is dynamically typed, we don't know the return type until
        the function is called. Thus, we only traverse the function body and
        process those nodes when it's called and we know the types of the
        inputs. And if a user-defined function is never called, we process it at
        the end of the AST traversal and in the namespace it was defined in.

        This processing is done by self._process_function. All this
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

        # NOTE: namespace is only used if the function is never called or if its
        # a closure
        function = Function(
            node,
            self._namespace.copy(),
            is_closure=False,
            called=False
        )
        self._namespace[node.name] = function
        self._functions.add(function)

        if node.decorator_list:
            decorator_call_node = utils.create_decorator_call_node(
                node.decorator_list,
                ast.Name(node.name)
            )
            _, entity, _ = self._process_node(decorator_call_node)

            if not entity:
                return function

            self._namespace[node.name] = entity
            if isinstance(entity, Function):
                self._functions.add(entity)

            return entity

        return function

    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node)

    def visit_Lambda(self, node):
        """
        TODO
        """

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

        self._process_subtree_in_new_scope(node.body, namespace)

        # TODO (V2): Handle assignments to lambdas and lambda function calls

    def visit_Return(self, node):
        """
        TODO
        """

        if node.value:
            _, self._return_value, _ = self._process_node(node.value)
            # BUG: What about instance, returned by _process_node?

        self._is_traversal_halted = True

    def visit_ClassDef(self, node):
        """
        TODO
        """

        for base_node in node.bases: # TODO (V2): Handle inheritance
            self.visit(ast.Call(func=base_node, args=[], keywords=[]))

        # TODO (V2): Handle metaclasses

        methods, nested_classes, static_variables = [], [], []
        stripped_body = [] # ;)
        for n in node.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in n.decorator_list:
                    if not hasattr(decorator, "id"):
                        continue

                    if decorator.id == "staticmethod":
                        n.method_type = "static"
                        break
                    elif decorator.id == "classmethod":
                        n.method_type = "class"
                        break
                else:
                    n.method_type = "instance"

                methods.append(n)
                continue
            elif isinstance(n, ast.ClassDef):
                nested_classes.append(n)
                continue
            elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                # BUG: Static variables can be defined without an assignment.
                # For example:
                    # class foo(object):
                    #     for x in range(10):
                    #         continue
                # foo.x is valid.

                targets = [n.target] if not isinstance(n, ast.Assign) else n.targets
                for target in targets:
                    if isinstance(target, ast.Tuple):
                        for element in target.elts:
                            static_variables.append(utils.stringify_node(element))
                    else:
                        static_variables.append(utils.stringify_node(target))

            stripped_body.append(n)

        class_level_namespace = self._process_subtree_in_new_scope(
            ast.Module(body=stripped_body),
            self._namespace.copy()
        )._namespace

        class_entity = Class(node, self._namespace.copy())
        self._namespace[node.name] = class_entity

        static_variable_map = {}
        for name, n in class_level_namespace.items():
            if name in static_variables:
                self._namespace['.'.join((node.name, name))] = n
                static_variable_map[name] = n

        def create_callable_attribute_map(callables):
            callable_map = {}
            for callable in callables:
                callable_name = callable.name

                # Handles callables that are accessed by the enclosing class
                adjusted_name = '.'.join((node.name, callable_name))
                callable.name = adjusted_name

                if isinstance(callable, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    entity = self.visit_FunctionDef(callable)
                elif isinstance(callable, ast.ClassDef):
                    entity = self.visit_ClassDef(callable)

                callable.name = callable_name

                if isinstance(entity, Function):
                    entity.method_type = callable.method_type
                    entity.containing_class = class_entity

                callable_map[callable_name] = entity

            return callable_map

        method_map = create_callable_attribute_map(methods)
        nested_class_map = create_callable_attribute_map(nested_classes)

        # Everything here is an attribute of `self`
        class_entity.init_instance_namespace = {
            **static_variable_map,
            **method_map,
            **nested_class_map
        }

        return class_entity

    ## Control Flow Handlers ##

    def visit_If(self, node):
        """
        Namespace changes in the first `If` block persist into the parent
        context, but changes made in `Elif` or `Else` blocks do not.

        TODO

        Parameters
        ----------
        node : ast.If
            test ...
            body ...
            orelse ...
        """

        self.visit(node.test)

        sub_namespaces = []
        for if_body in [ast.Module(body=node.body)] + node.orelse:
            sub_namespace = self._process_subtree_in_new_scope(
                tree=if_body,
                namespace=self._namespace.copy()
            )._namespace
            sub_namespaces.append(sub_namespace)

        for sub_namespace in sub_namespaces:
            utils.diff_and_clean_namespaces(self._namespace, sub_namespace)

    def visit_For(self, node):
        """
        TODO

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
            value=ast.Call(
                func=ast.Attribute(
                    value=node.iter,
                    attr="__iter__",
                    ctx=ast.Load()
                ),
                args=[],
                keywords=[]
            )
        )
        self.visit(ast.Module(body=[target_assignment] + node.body))

        if not self._is_traversal_halted:
            self.visit(ast.Module(body=node.orelse))

        # If loop is broken by anything other than a return statement, then we
        # don't want to halt the traversal outside of the loop
        if not self._return_value:
            self._is_traversal_halted = False

    def visit_AsyncFor(self, node):
        self.visit_For(node)

    def visit_While(self, node):
        """
        TODO
        """

        self.visit(ast.Module(body=node.body))

        if not self._is_traversal_halted:
            self.visit(ast.Module(body=node.orelse))

        # If loop is broken by anything other than a return statement, then we
        # don't want to halt the traversal outside of the loop
        if not self._return_value:
            self._is_traversal_halted = False

    def visit_Try(self, node):
        """
        TODO
        """

        try_namespace = self._process_subtree_in_new_scope(
            tree=ast.Module(body=node.body + node.orelse),
            namespace=self._namespace.copy()
        )._namespace

        sub_namespaces = [try_namespace]
        for except_handler_node in node.handlers:
            except_namespace = self.visit_ExceptHandler(except_handler_node)
            sub_namespaces.append(except_namespace)

        for sub_namespace in sub_namespaces:
            utils.diff_and_clean_namespaces(self._namespace, sub_namespace)

        # node.finalbody is executed no matter what
        self.visit(ast.Module(body=node.finalbody))

    def visit_ExceptHandler(self, node):
        """
        TODO
        """

        body_to_process = node.body
        if node.type and node.name:
            exception_alias_assign_node = ast.Assign(
                targets=[ast.Name(id=node.name, ctx=ast.Store())],
                value=node.type
            )
            body_to_process.insert(0, exception_alias_assign_node)
        elif node.type:
            self.visit(node.type)

        except_namespace = self._process_subtree_in_new_scope(
            ast.Module(body=body_to_process),
            self._namespace.copy()
        )._namespace
        return except_namespace

    def visit_withitem(self, node):
        """
        TODO
        """

        # TODO (V1): Add call nodes for .__enter__() and .__exit__()

        if node.optional_vars:
            assign_node = ast.Assign(
                targets=[node.optional_vars],
                value=node.context_expr
            )
            self.visit(assign_node)
        else:
            self.visit(node.context_expr)

    def visit_Continue(self, node):
        self._is_traversal_halted = True

    def visit_Break(self, node):
        self._is_traversal_halted = True

    ## Core Handlers ##

    @utils.attribute_chain_handler
    def visit_Name(self, node):
        pass

    @utils.attribute_chain_handler
    def visit_Attribute(self, node):
        pass

    @utils.attribute_chain_handler
    def visit_Call(self, node):
        pass

    @utils.attribute_chain_handler
    def visit_Subscript(self, node):
        pass

    @utils.attribute_chain_handler
    def visit_BinOp(self, node):
        pass

    @utils.attribute_chain_handler
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
        """
        TODO
        """

        comprehension_body = []
        for generator in generators:
            iter_node = ast.Assign(
                target=generator.target,
                value=ast.Subscript(
                    value=generator.iter,
                    slice=ast.Index(value=ast.NameConstant(None)),
                    ctx=ast.Load()
                )
            )
            comprehension_body.append(iter_node)
            comprehension_body.extend(generator.ifs)

        self._process_subtree_in_new_scope(
            ast.Module(body=comprehension_body + elts),
            self._namespace.copy()
        )

    def visit_ListComp(self, node):
        self._comprehension_helper([node.elt], node.generators)

    def visit_SetComp(self, node):
        return self.visit_ListComp(node)

    def visit_GeneratorExp(self, node):
        return self.visit_ListComp(node)

    def visit_DictComp(self, node):
        self._comprehension_helper([node.key, node.value], node.generators)

    ## Public Methods ##

    def get_trees(self):
        """
        TODO
        """

        trees = []
        for root_node in self._object_hierarchies:
            utils.consolidate_call_nodes(root_node)
            trees.append(root_node)

        return trees
