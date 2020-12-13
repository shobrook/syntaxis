# Standard Library
import ast
from collections import defaultdict
from copy import copy

# TODO (V1): Write/Update docstrings
    # Use vocabulary like "containing scope" instead of parent scope, and
    # "objects with local scope die as soon as the function exits, unless
    # they're returned"

    # Explain how "attribute chains" and "object flow" are key to understanding
    # how this all works


####################
# NAMESPACE ENTITIES
####################


class ObjectNode(object):
    def __init__(self, name, is_callable=False, order=0, children=[]):
        """
        Module tree node constructor. Each node represents a feature in a
        package's API. A feature is defined as an object, function, or variable
        that can only be used by importing the package.

        @param id: original identifier for the node.
        @param children: connected sub-nodes.
        """

        self.name = str(name)
        self.is_callable = is_callable
        self.order = order
        self.children = []

        self.count = 1

        for child in children:
            self.add_child(child)

    def __repr__(self):
        return self.name

    def __str__(self):
        return f"{self.name} ({'C' if self.is_callable else 'NC'}, {self.order})"

    def __eq__(self, node):
        if isinstance(node, type(self)):
            return self.name == node.name

        return False

    def __ne__(self, node):
        return not self.__eq__(node)

    ## Instance Methods ##

    def increment_count(self):
        self.count += 1

    def add_child(self, node):
        for child in self.children:
            if child == node: # Child already exists
                # child.increment_count()
                return child

        self.children.append(node)
        return node

    def breadth_first(self):
        node_queue = [self]
        while node_queue:
            node = node_queue.pop(0)
            yield node
            for child in node.children:
                node_queue.append(child)


class Function(object):
    def __init__(self, def_node, init_namespace, is_closure=False, called=False, method_type=None, containing_class=None):
        self.def_node = def_node
        self.init_namespace = init_namespace
        self.is_closure = is_closure
        self.called = called
        self.method_type = method_type
        self.containing_class = containing_class # TEMP


class Class(object):
    def __init__(self, def_node, init_namespace, is_closure=False):
        self.def_node = def_node
        self.init_namespace = init_namespace
        self.is_closure = is_closure


class ClassInstance(object):
    def __init__(self, class_entity, namespace):
        self.class_entity = class_entity
        self.namespace = namespace


########
# TOKENS
########


class NameToken(object):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


class ArgToken(object):
    def __init__(self, arg_val, arg_name=''):
        self.arg_val, self.arg_name = arg_val, arg_name

    def __iter__(self):
        yield from self.arg_val


class CallToken(object):
    def __init__(self, args):
        self.args = args

    def __iter__(self):
        yield from self.args

    def __repr__(self):
        return "()"


######################
# TOKENIZATION HELPERS
######################


BIN_OPS_TO_FUNCS = {
    "Add": "__add__",
    "Sub": "__sub__",
    "Mult": "__mul__",
    "Div": "__truediv__",
    "FloorDiv": "__floordiv__",
    "Mod": "__mod__",
    "Pow": "__pow__",
    "LShift": "__lshift__",
    "RShift": "__rshift__",
    "BitOr": "__or__",
    "BitXor": "__xor__",
    "BitAnd": "__and__",
    "MatMult": "__matmul__"
}
COMPARE_OPS_TO_FUNCS = {
    "Eq": "__eq__",
    "NotEq": "__ne__",
    "Lt": "__lt__",
    "LtE": "__le__",
    "Gt": "__gt__",
    "GtE": "__ge__",
    "Is": "__eq__",
    "IsNot": "__ne__",
    "In": "__contains__",
    "NotIn": "__contains__"
}


def tokenize_slice(slice):
    if isinstance(slice, ast.Index): # e.g. x[1]
        yield recursively_tokenize_node(slice.value, [])
    elif isinstance(slice, ast.Slice): # e.g. x[1:2]
        for partial_slice in (slice.lower, slice.upper, slice.step):
            if not partial_slice:
                continue

            yield recursively_tokenize_node(partial_slice, [])


def recursively_tokenize_node(node, tokens):
    """
    Takes a node representing an identifier or function call and recursively
    unpacks it into its constituent tokens. A "function call" includes
    subscripts (e.g. my_var[1:4] => my_var.__index__(1, 4)), binary operations
    (e.g. my_var + 10 => my_var.__add__(10)), comparisons (e.g. my_var > 10 =>
    my_var.__gt__(10)), and ... .

    Each token in this list is a child of the previous token. The "base" token
    are NameTokens. These are object references.
    """

    if isinstance(node, ast.Name):
        tokens.append(NameToken(node.id))
        return tokens[::-1]
    elif isinstance(node, ast.Call):
        tokenized_args = []

        for arg in node.args:
            arg = ArgToken(
                arg_val=recursively_tokenize_node(arg, []),
                arg_name=''
            )
            tokenized_args.append(arg)

        for keyword in node.keywords:
            arg = ArgToken(
                arg_val=recursively_tokenize_node(keyword.value, []),
                arg_name=keyword.arg
            )
            tokenized_args.append(arg)

        tokens.append(CallToken(tokenized_args))
        return recursively_tokenize_node(node.func, tokens)
    elif isinstance(node, ast.Attribute):
        tokens.append(NameToken(node.attr))
        return recursively_tokenize_node(node.value, tokens)
    elif isinstance(node, ast.Subscript):
        slice = node.slice
        slice_tokens = []
        if isinstance(slice, ast.ExtSlice): # e.g. x[1:2, 3]
            for dim_slice in slice.dims:
                slice_tokens.extend(tokenize_slice(dim_slice))
        else:
            slice_tokens.extend(tokenize_slice(slice))

        arg_tokens = CallToken([ArgToken(token) for token in slice_tokens])
        subscript_name = NameToken("__index__")
        tokens.extend([arg_tokens, subscript_name])

        return recursively_tokenize_node(node.value, tokens)
    elif isinstance(node, ast.BinOp):
        op_args = CallToken([ArgToken(
            arg_val=recursively_tokenize_node(node.right, []),
        )])
        op_name = NameToken(BIN_OPS_TO_FUNCS[type(node.op).__name__])
        tokens.extend([op_args, op_name])

        return recursively_tokenize_node(node.left, tokens)
    elif isinstance(node, ast.Compare):
        operator = node.ops[0]
        comparator = node.comparators[0]

        if node.ops[1:] and node.comparators[1:]:
            new_compare_node = ast.Compare(
                left=comparator,
                ops=node.ops[1:],
                comparators=node.comparators[1:]
            )
            op_args = CallToken([ArgToken(
                arg_val=recursively_tokenize_node(new_compare_node, [])
            )])
        else:
            op_args = CallToken([ArgToken(
                arg_val=recursively_tokenize_node(comparator, [])
            )])

        op_name = NameToken(COMPARE_OPS_TO_FUNCS[type(operator).__name__])
        tokens.extend([op_args, op_name])

        return recursively_tokenize_node(node.left, tokens)
    else:
        return [node]


def stringify_tokenized_nodes(tokens):
    stringified_tokens = ''
    for index, token in enumerate(tokens):
        if index and isinstance(token, NameToken):
            stringified_tokens += '.' + str(token)
        else:
            stringified_tokens += str(token)

    return stringified_tokens


###############
# MISC. HELPERS
###############


def attribute_chain_handler(func):
    def wrapper(self, node):
        tokens = recursively_tokenize_node(node, [])
        self._process_attribute_chain(tokens)

    return wrapper


def delete_sub_aliases(targ_str, namespace):
    """
    If `my_var` is an alias, then `my_var()` and `my_var.attr` are
    considered sub-aliases. This function takes aliases that have been
    deleted or reassigned and deletes all of its sub-aliases.

    Parameters
    ----------
    targ_str : string
        string representation of the target node in the assignment
    """

    for alias in list(namespace.keys()):
        for sub_alias_signifier in ('(', '.'):
            if alias.startswith(targ_str + sub_alias_signifier):
                del namespace[alias]
                break


def consolidate_call_nodes(node, parent=None):
    for child in node.children:
        consolidate_call_nodes(child, node)

    if node.name == "()":
        parent.is_callable = True
        parent.children.remove(node)
        for child in node.children:
            child.order += 1
            parent.children.append(child)


##########
# SAPLINGS
##########


class Saplings(ast.NodeVisitor):
    def __init__(self, tree, object_hierarchies=[], namespace={}):
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

        self._object_hierarchies = object_hierarchies

        # Maps active identifiers to ObjectNodes, Functions, Classes, and
        # ClassInstances
        self._namespace = namespace

        # Keeps track of functions defined in the current scope
        self._functions = set()

        # ObjectNode, Function, Class, or ClassInstance produced by the first
        # evaluated return statement in the AST
        self._return_value = None

        # True when a Return, Continue, or Break node is hit –– stops traversal
        # of subtree
        self._is_traversal_halted = False

        self.visit(tree)
        self._process_uncalled_functions()

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

    def visit(self, node):
        """
        Overloaded AST visitor. Behaves the same as ast.NodeVisitor.visit except
        it only traverses a subtree when self._is_traversal_halted == False.
        """

        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)

        if not self._is_traversal_halted:
            return visitor(node)

    ## Helpers ##

    def _process_subtree_in_new_scope(self, tree, namespace):
        """
        Used to process a subtree in a different scope/namespace from the
        current scope.

        Parameters
        ----------
        tree : ast.AST
            root node of the subtree to process
        namespace : dict
            namespace within which the subtree should be processed
        """

        return Saplings(tree, self._object_hierarchies, namespace)

    ## Processors ##

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
        ObjectNode
            reference to the terminal d-tree node for the module
        """

        # TODO (V2): Check if you can refactor this function to use existing
        # code

        sub_modules = module.split('.')  # For module.submodule1.submodule2...
        root_module = sub_modules[0]
        term_node = None

        def find_matching_node(subtree, name):
            for node in subtree.breadth_first():
                if node.name == name:
                    return node

            return None

        for root in self._object_hierarchies:
            matching_module = find_matching_node(root, root_module)

            if matching_module:
                term_node = matching_module
                break

        if not term_node:
            root_node = ObjectNode(root_module, order=-1)
            if standard_import:
                self._namespace[root_module] = root_node

            term_node = root_node
            self._object_hierarchies.append(term_node)

        for idx in range(len(sub_modules[1:])):
            sub_module = sub_modules[idx + 1]
            sub_module_alias = '.'.join([root_module] + sub_modules[1:idx + 2])

            matching_sub_module = find_matching_node(term_node, sub_module)

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

            tokenized_default = recursively_tokenize_node(default, [])
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

        # TODO (V1): Handle single-star args

        # Starred nodes can now appear in args, and kwargs is replaced by keyword
        # nodes in keywords for which arg is None.


        # args, posonlyargs and kwonlyargs are lists of arg nodes.
        # vararg and kwarg are single arg nodes, referring to the *args, **kwargs parameters.
        # kw_defaults is a list of default values for keyword-only arguments. If one is None, the corresponding argument is required.
        # defaults is a list of default values for arguments that can be passed positionally. If there are fewer defaults, they correspond to the last n arguments.

        # TODO: Check if arguments is empty to skip all this BS

        parameters = function.def_node.args

        arg_names = [a.arg for a in parameters.args]
        kwonlyarg_names = [a.arg for a in parameters.kwonlyargs]

        # Object nodes corresponding to default values
        default_nodes, null_defaults = self._process_default_args(
            arg_names,
            parameters.defaults,
            namespace
        )
        kw_default_nodes, null_kw_defaults = self._process_default_args(
            kwonlyarg_names,
            parameters.kw_defaults,
            namespace
        )

        # Consolidates arg names
        if parameters.vararg:
            arg_names.append(parameters.vararg.arg)
        arg_names.extend(kwonlyarg_names)
        if parameters.kwarg:
            arg_names.append(parameters.kwarg.arg)

        # Update namespace with default values
        namespace = {**namespace, **default_nodes, **kw_default_nodes}
        for null_arg_name in null_defaults + null_kw_defaults:
            if null_arg_name in namespace:
                del namespace[null_arg_name]
                delete_sub_aliases(null_arg_name, namespace)

        # Iterates through arguments in order they were passed into function
        for index, argument in enumerate(arguments):
            if not argument.arg_name:
                arg_name = arg_names[index]
            else:
                arg_name = argument.arg_name

            arg_node, _ = self._process_attribute_chain(argument.arg_val)
            if not arg_node:
                if arg_name in namespace:
                    del namespace[arg_name]
                    delete_sub_aliases(arg_name, namespace)

                continue

            delete_sub_aliases(arg_name, namespace)
            namespace[arg_name] = arg_node

        # TODO (V1): Simply create ast.Assign objects for each default argument
        # and prepend them to function.def_node.body –– let the Saplings instance
        # below do the rest.

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
            if not return_value.called:
                self._functions.add(return_value)

        return return_value, func_saplings

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

        tokenized_target = recursively_tokenize_node(target, tokens=[])
        targ_node, instance = self._process_attribute_chain(tokenized_target)

        tokenized_value = recursively_tokenize_node(value, tokens=[])
        val_node, _ = self._process_attribute_chain(tokenized_value)

        # TODO (V2): Handle assignments to data structures. For an assignment
        # like foo = [bar(i) for i in range(10)], foo.__index__() should be an
        # alias for bar().

        # TODO (V2): Handle assignments to class variables that propagate to
        # class instances (e.g. MyClass.variable = ...; my_instance.variable.foo())
        if instance["entity"]:
            namespace = instance["entity"].namespace
            targ_str = stringify_tokenized_nodes(tokenized_target[instance["init_index"] + 1:])
        else:
            namespace = self._namespace
            targ_str = stringify_tokenized_nodes(tokenized_target)

        # Type I: Known node reassigned to other known node (K2 = K1)
        if targ_node and val_node:
            namespace[targ_str] = val_node
            delete_sub_aliases(targ_str, namespace)
        # Type II: Known node reassigned to unknown node (K1 = U1)
        elif targ_node and not val_node:
            del namespace[targ_str]
            delete_sub_aliases(targ_str, namespace)
        # Type III: Unknown node assigned to known node (U1 = K1)
        elif not targ_node and val_node:
            namespace[targ_str] = val_node

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

        Parameters
        ----------
        tokens : list
            list of `NameToken`s and `CallToken`s

        Returns
        -------
        {ObjectNode, Function, Class, ClassInstance}, context_dict
        """

        def break_and_process_nested_chains(tokens, current_entity, current_instance):
            for token in tokens:
                if not isinstance(token, CallToken):
                    continue

                for arg_token in token:
                    self._process_attribute_chain(arg_token.arg_val)

            current_entity = None
            current_instance["entity"] = None
            current_instance["init_index"] = 0

        def process_function_call(function, arguments):
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
            ) # TODO (V2): Handle tuple returns
            return return_value

        def bind_entity_to_arguments(entity, arguments):
            hidden_arg = ArgToken([NameToken("")])
            self._namespace[""] = entity

            return [hidden_arg] + arguments

        def process_method_call(function, arguments, class_instance=None):
            if class_instance: # Method is called by instance
                if function.method_type == "class":
                    arguments = bind_entity_to_arguments(
                        class_instance.class_entity,
                        arguments
                    )
                elif function.method_type == "instance":
                    arguments = bind_entity_to_arguments(
                        class_instance,
                        arguments
                    )
            else: # Method is called by class
                arguments = bind_entity_to_arguments(
                    function.containing_class,
                    arguments
                )

            return_value = process_function_call(function, arguments)

            if "" in self._namespace:
                del self._namespace[""]

            return return_value

        current_entity = None
        current_instance = {"entity": None, "init_index": 0}
        for index, token in enumerate(attribute_chain):
            if index and not current_entity:
                break_and_process_nested_chains(
                    attribute_chain[index:],
                    current_entity,
                    current_instance
                )
                break

            if isinstance(token, CallToken):
                if isinstance(current_entity, Function):
                    if current_instance["entity"]:
                        # Process call of function from instance of a
                        # user-defined class
                        current_entity = process_method_call(
                            current_entity,
                            token.args,
                            current_instance["entity"]
                        )
                    else:
                        if current_entity.method_type == "class":
                            # Process call of function from user-defined class
                            current_entity = process_method_call(
                                current_entity,
                                token.args
                            )
                        else:
                            # Process call of user-defined function that is
                            # either unbound to a class or a static method
                            current_entity = process_function_call(
                                current_entity,
                                token.args
                            )

                    if isinstance(current_entity, ClassInstance):
                        current_instance["entity"] = current_entity
                        current_instance["init_index"] = index

                    continue
                elif isinstance(current_entity, Class):
                    # Process instantiation of user-defined class

                    # TODO (V1): Handle classes defined inside a class
                    # TODO (V2) Handle functions that return a class (i.e. class closures)

                    init_namespace = current_entity.init_namespace
                    class_instance = ClassInstance(
                        current_entity,
                        init_namespace.copy()
                    )

                    if "__init__" in init_namespace:
                        constructor = init_namespace["__init__"]
                        if isinstance(constructor, Function):
                            process_method_call(
                                constructor,
                                token.args,
                                class_instance
                            )
                        else:
                            # If __init__ is not callable, class cannot be
                            # instantiated
                            break_and_process_nested_chains(
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
                            current_entity = process_method_call(
                                call_entity,
                                token.args,
                                current_entity
                            )

                            if isinstance(current_entity, ClassInstance):
                                current_instance["entity"] = current_entity
                                current_instance["init_index"] = index

                            continue

                    break_and_process_nested_chains(
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
            elif not isinstance(token, NameToken): # token is ast.AST node
                self.visit(token)
                # TODO (V2): Handle the following: (lambda x: x.attr)(module.foo)
                break_and_process_nested_chains(
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

            token_str = stringify_tokenized_nodes(token_seq)
            if token_str in namespace:
                current_entity = namespace[token_str]
                if isinstance(current_entity, ClassInstance):
                    current_instance["entity"] = current_entity
                    current_instance["init_index"] = index
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
                if alias.name == child.name:
                    child_exists = True
                    self._namespace[alias_id] = child

                    break

            if not child_exists:
                new_child = ObjectNode(alias.name)
                self._namespace[alias_id] = new_child

                module_node.add_child(new_child)

    def visit_Assign(self, node):
        values = node.value
        targets = node.targets if hasattr(node, "targets") else (node.target,)
        for target in targets: # Multiple assignment (e.g. a = b = ...)
            if isinstance(target, ast.Tuple): # Unpacking (e.g. a, b = ...)
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
            target_tokens = recursively_tokenize_node(target, [])
            target_str = stringify_tokenized_nodes(target_tokens)

            delete_sub_aliases(target_str, self._namespace)

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

        # TODO (V2): Handle decorators

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

        return function

    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node)

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

        self._process_subtree_in_new_scope(node.body, namespace)

        # TODO (V2): Handle assignments to lambdas and lambda function calls

    def visit_Return(self, node):
        if node.value:
            tokenized_node = recursively_tokenize_node(node.value, [])
            self._return_value, _ = self._process_attribute_chain(tokenized_node)

        self._is_traversal_halted = True

    def visit_ClassDef(self, node):
        """
        """

        for base_node in node.bases: # TODO (V2): Handle inheritance (?)
            self.visit(base_node)

        # TODO (V2): Handle metaclasses
        # TODO (V2): Handle decorators

        def stringify_target(target):
            tokens = recursively_tokenize_node(target, tokens=[])
            targ_str = stringify_tokenized_nodes(tokens)

            return targ_str

        methods, static_variables = [], []
        body_without_methods = []
        for n in node.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in n.decorator_list:
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
            elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                # BUG: Static variables can be defined without an assignment.
                # For example:
                    # class foo(object):
                    #     for x in range(10):
                    #         pass
                # foo.x is valid.

                targets = [n.target] if not isinstance(n, ast.Assign) else n.targets
                for target in targets:
                    if isinstance(target, ast.Tuple):
                        for element in target.elts:
                            static_variables.append(stringify_target(element))
                    else:
                        static_variables.append(stringify_target(target))

            body_without_methods.append(n)

        class_level_namespace = self._process_subtree_in_new_scope(
            ast.Module(body=body_without_methods),
            self._namespace.copy()
        )._namespace

        class_entity = Class(node, {}, is_closure=False)
        self._namespace[node.name] = class_entity

        static_variable_map = {}
        for name, n in class_level_namespace.items():
            if name in static_variables:
                self._namespace['.'.join((node.name, name))] = n
                static_variable_map[name] = n

        method_map = {}
        for method in methods:
            method_name = method.name

            # Handles methods that are accessed by the enclosing class
            adjusted_name = '.'.join((node.name, method_name))
            method.name = adjusted_name
            function = self.visit_FunctionDef(method)
            method.name = method_name

            function.method_type = method.method_type
            function.containing_class = class_entity
            method_map[method_name] = function

        # Everything here is an attribute of `self`
        class_entity.init_namespace = {**static_variable_map, **method_map}

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

        # If node is processed last so the Else nodes are processed with an
        # unaltered namespace
        for else_node in node.orelse:
            self._process_subtree_in_new_scope(
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

        if not self._is_traversal_halted:
            self.visit(ast.Module(body=node.orelse))

        # If loop is broken by anything other than a return statement, then we
        # don't want to halt the traversal outside of the loop
        if not self._return_value:
            self._is_traversal_halted = False

    def visit_AsyncFor(self, node):
        self.visit_For(node)

    def visit_While(self, node):
        self.visit(ast.Module(body=node.body))

        if not self._is_traversal_halted:
            self.visit(ast.Module(body=node.orelse))

        # If loop is broken by anything other than a return statement, then we
        # don't want to halt the traversal outside of the loop
        if not self._return_value:
            self._is_traversal_halted = False

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
            tokenized_exception = recursively_tokenize_node(node.type, [])
            exception_node, _ = self._process_attribute_chain(tokenized_exception)

            if node.name: # except A as B
                tokenized_alias = recursively_tokenize_node(node.name, [])
                alias_str = stringify_tokenized_nodes(tokenized_alias)

                if not exception_node:
                    del namespace[alias_str]
                else:
                    namespace[alias_str] = exception_node

        body_saplings_obj = self._process_subtree_in_new_scope(
            ast.Module(body=node.body),
            namespace
        )

    def visit_withitem(self, node):
        assign_node = ast.Assign(
            targets=[node.optional_vars],
            value=node.context_expr
        )
        self.visit(assign_node)

    def visit_Continue(self, node):
        self._is_traversal_halted = True

    def visit_Break(self, node):
        self._is_traversal_halted = True

    ## Core Handlers ##

    @attribute_chain_handler
    def visit_Name(self, node):
        pass

    @attribute_chain_handler
    def visit_Attribute(self, node):
        pass

    @attribute_chain_handler
    def visit_Call(self, node):
        pass

    @attribute_chain_handler
    def visit_Subscript(self, node):
        pass

    @attribute_chain_handler
    def visit_BinOp(self, node):
        pass

    @attribute_chain_handler
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

    ## Miscellaneous ##

    # TODO (V1): Handle ifexps

    ## Public Methods ##

    def get_trees(self):
        trees = []
        for root_node in self._object_hierarchies:
            consolidate_call_nodes(root_node)
            trees.append(root_node)

        return trees
