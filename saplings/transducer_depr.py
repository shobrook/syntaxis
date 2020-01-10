#########
# GLOBALS
#########


# Standard Library
import ast
from collections import defaultdict

# Local
import utilities as utils


######
# MAIN
######


# [] Inside funcs, block searching parent contexts for aliases equivalent to parameter names (unless it's self)
# [] Debug the frequency analysis
# [] Handle globals and nonlocals

# Handling Functions #
# - Save defined function names and their return types in a field (search field when processing tokens)
# - Is processing the input types of functions as simple as adding the parameter name as an alias for the input node,
#   then calling self.visit() on the function node? You could only modify the input node and it's children, cuz everything
#   else in the function body has already been processed.

class Transducer(ast.NodeVisitor):
    def __init__(self, tree, forest=[], namespace={}):
        # QUESTION: Add a `conservative` parameter?

        self._node_lookup_table = dict(namespace)
        self.conditional_assignments = []

        self.forest = forest  # Holds root nodes of output trees
        self.visit(tree)  # Begins traversal

    ## Private Helpers ##

    def _recursively_process_tokens(self, tokens, no_increment=False, is_data_struct=False):
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

        # BUG: Doesn't handle nested data structures
        def handle_data_structs(content, type):
            if type == "comprehension":
                for sub_token in content:
                    if sub_token[1] == "target":
                        continue  # TODO: Block search for targets when processing elts

                    self._recursively_process_tokens(sub_token[0])

                return True  # TEMP: Return an alias map, or something
            elif type == "array":
                alias_map = {}
                for idx, elmt in enumerate(content):
                    elmt_nodes = self._recursively_process_tokens(elmt)
                    if elmt_nodes:
                        alias_map[str(idx)] = elmt_nodes

                return alias_map
            elif type == "hashmap":
                alias_map = {}
                for key, val in content:
                    val_nodes = self._recursively_process_tokens(val)
                    if len(key) == 1 and key[0][1] in ("str", "num"):
                        alias_map[key[0][0]] = val_nodes
                    else:
                        self._recursively_process_tokens(key)

                return alias_map

            return None

        # QUESTION: How to handle when a data struct. gets manipulated
        # (appended to, popped from, etc.)?

        # if is_data_struct:
        #     content, type = tokens[0]
        #     return handle_data_structs(content, type)

        node_stack = []

        # Flattens nested tokens
        flattened_tokens = []
        # QUESTION: Does this need to be in a separate loop?
        for idx, token in enumerate(tokens):
            content, type = token

            if type == "call":
                for sub_tokens in content:
                    self._recursively_process_tokens(sub_tokens)

                content = "()"
            elif type == "subscript":
                if len(content) == 1 and content[0][1] in ("str", "num"):
                    content = '[' + content[0][0] + ']'
                else:
                    self._recursively_process_tokens(content)
                    content = "[]"
            else:
                nodes_in_structs = handle_data_structs(content, type)
                if nodes_in_structs:
                    content = ''

            flattened_tokens.append((content, type))

        # This list of tokens should have the property of: if the current token
        # is a node, then the next token is a child of that node
        for idx, token in enumerate(flattened_tokens):
            content, type = token

            # BUG: Can't handle something like [1,2,3, ...][0].foo()
            if type in ("comprehension", "array", "hashmap"):
                break

            token_id = utils.stringify_tokenized_nodes(
                flattened_tokens[:idx + 1])
            if token_id in self._node_lookup_table:
                node = self._node_lookup_table[token_id]

                if not no_increment:
                    node.increment_count()

                node_stack.append(node)
            elif node_stack:  # Base token exists; create its child
                node = node_stack[-1].add_child(utils.Node(content))
                self._node_lookup_table[token_id] = node
                node_stack.append(node)
            else:  # Base token doesn't exist; abort processing
                break

        return node_stack

    def _process_assignment(self, target, value):
        """
        @param target: AST node on the left-hand-side of the assignment.
        @param value: tokenized AST node on the right-hand-side of the assignment.
        """

        is_data_struct = len(value) == 1 and value[0][1] in (
            "comprehension", "array", "hashmap")
        tokenized_target = utils.recursively_tokenize_node(target, [])

        targ_matches = self._recursively_process_tokens(tokenized_target)  # LHS
        val_matches = self._recursively_process_tokens(
            value, is_data_struct=is_data_struct)  # RHS

        # BUG: This is stringifying the non-flattened list of tokens
        alias = utils.stringify_tokenized_nodes(tokenized_target)

        # TODO: When you delete an alias after reassignment, delete `alias`
        # subscripts too. i.e. x = [1,2,3] then x is reassigned, and then nodes
        # with x[0] and x[1] as aliases won't get deleted but need to be.

        # TODO: Handle aliasing for is_data_struct case

        # Type I: Known node reassigned to known node (K2 = K1)
        if targ_matches and val_matches:
            targ_node, val_node = targ_matches[-1], val_matches[-1]

            self._node_lookup_table[alias] = val_node
            self.conditional_assignments.append(alias)  # TODO: Double-check
        # Type II: Known node reassigned to unknown node (K1 = U1)
        elif targ_matches and not val_matches:
            targ_node = targ_matches[-1]

            del self._node_lookup_table[alias]
            self.conditional_assignments.append(alias)  # TODO: Double-check
        # Type III: Unknown node assigned to known node (U1 = K1)
        elif not targ_matches and val_matches:
            val_node = val_matches[-1]

            self._node_lookup_table[alias] = val_node

    def _process_module(self, module, alias_origin_module=True):
        """
        Takes the identifier for a module, sometimes a period-separated string
        of sub-modules, and searches the API forest for a matching module. If no
        match is found, new module nodes are generated and appended to
        self.forest.

        @param module: identifier for the module.
        @param alias_root: flag for whether a newly created module node should
        be aliased.

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

    ## Context Handlers ##

    # @utils.context_handler
    def visit_ClassDef(self, node):
        self.generic_visit(node)

    # @utils.context_handler
    def visit_FunctionDef(self, node):
        # NOTE: A function will be evaluated differently depending on when its
        # called.

        # ALGORITHM:
        # When a FunctionDef is hit, add the node to a dictionary A =
        # {"func_name": {"node": Node, "namespace": {...}}} and skip the traversal.
        # When a Call is hit, check if the function name is in A.
            # If it is, create a new Transducer with the current namespace passed in.
            # Then, mark the function as "called."
        # If a function is redefined, remove it from A.
            # If it isn't marked as "called", first add to the uncalled list and
            # then replace with new function in A.
        # At the end of the traversal, create Transducers for each uncalled
        # function.

        # NOTE: Reassignments inside Functions don't have effects on the parent
        # context (unless global is used).

        # NOTE: When a function is evaluated following a call, its namespace needs
        # to be adjusted to map parameter names to the call arguments.

        # NOTE: When an uncalled function is evaluated, its namespace needs to be
        # adjusted for the function's parameter names (remove them from namespace).

        # NOTE: Functions can have multiple aliases and be assigned to things:
            # def foo():
            #    pass
            # bar = foo
            # bar()

        # QUESTION: What about recursive functions? What about closures?

        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    ## Control Flow Handlers ##

    # @utils.conditional_handler
    def visit_If(self, node):
        try:
            node_name = type(node.test).__name__
            custom_visitor = getattr(self, ''.join(["visit_", node_name]))
            custom_visitor(node.test)
        except AttributeError:
            self.generic_visit(node.test)

        body_transducer = Transducer(
            tree=ast.Module(body=node.body),
            forest=self.forest,
            namespace=self._node_lookup_table
        )

        # Any Type I/II assignment (K2 = K1, K = U) should apply to child
        # contexts and K2/K should be deleted from the parent context
        # QUESTION: Run before or after visiting node.orelse?
        for alias in body_transducer.conditional_assignments:
            # QUESTION: What about the following scenario:
            # import x
            # x.foo()
            # if True:
            #    y = x.foo
            #    x.foo = None
            #    x.foo = y
            # x.foo().please() # This won't register. The alias was changed in
            # the If context, but then the change was reverted in some way.
                # Can you just look at the resulting lookup table from the
                # body_transducer?
            del self._node_lookup_table[alias]

        # TODO: Check that nodes in node.orelse don't also have an orelse
        # property
        for else_node in node.orelse:
            self.visit_If(else_node)

        # TODO: Check that nested Ifs work
        # TODO: Check that Elifs and Elses work

        # TODO: Handle Functions/Classes that are defined inside an If body

        # QUESTION: What about:
            # import foo
            # for x in y:
            #    if True:
            #        continue
            #    z = foo()
        # We can't know if `z = foo()` is ever evaluated.

    # @utils.conditional_handler
    def visit_Try(self, node):
        body_transducer = Transducer(
            tree=ast.Module(body=node.body),
            forest=self.forest,
            namespace=self._node_lookup_table
        )

        for alias in body_transducer.conditional_assignments:
            del self._node_lookup_table[alias]

        for except_handler_node in node.handlers:
            # TODO: Handle `name` assignments (i.e. except module.Exception as e)
            self.visit_Try(except_handler_node)

        # QUESTION: How to handle node.orelse and node.finalbody?
            # node.finalbody is where all final assignments are made, and should
            # persist into the parent context

    # @utils.conditional_handler
    def visit_While(self, node):
        try:
            node_name = type(node.test).__name__
            custom_visitor = getattr(self, ''.join(["visit_", node_name]))
            custom_visitor(node.test)
        except AttributeError:
            self.generic_visit(node.test)

        body_transducer = Transducer(
            tree=ast.Module(body=node.body),
            forest=self.forest,
            namespace=self._node_lookup_table
        )

        for alias in body_transducer.conditional_assignments:
            del self._node_lookup_table[alias]

    def visit_For(self, node):
        # TODO: Fix this up

        def del_aliases(target):
            """
            Deletes alias for target node in all contexts.
            """

            tokens = utils.recursively_tokenize_node(target, [])
            alias = utils.stringify_tokenized_nodes(tokens)
            nodes = self._recursively_process_tokens(tokens, no_increment=True)

            if not nodes:
                return

            nodes[-1].del_alias(curr_context, alias)

        if isinstance(node.target, ast.Name):
            del_aliases(node.target)
        elif isinstance(node.target, ast.Tuple) or isinstance(node.target, ast.List):
            for elt in node.target.elts:
                del_aliases(elt)

        tokens = utils.recursively_tokenize_node(node.iter, [])
        self._recursively_process_tokens(tokens)

        body_nodes = [node.iter] + node.body + node.orelse
        utils.visit_body_nodes(self, body_nodes)

        # QUESTION: How to handle node.orelse? Nodes in orelse are executed if
        # the loop finishes normally, rather than with a break statement

    def visit_AsyncFor(self, node):
        self.visit_For(node)

    def visit_withitem(self, node):
        # TODO: Fix this up

        def get_nodes(n):
            tokens = utils.recursively_tokenize_node(n, [])
            alias = utils.stringify_tokenized_nodes(tokens)
            nodes = self._recursively_process_tokens(tokens)

            if not nodes:
                return

            return nodes[-1], alias

        if node.optional_vars:
            value_node, value_alias = get_nodes(node.context_expr)
            target_node, target_alias = get_nodes(
                node.optional_vars)  # Name, Tuple, or List

            self._node_lookup_table[target_alias] = value_node

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
        if isinstance(node.value, ast.Tuple):
            def node_tokenizer(
                elt): return utils.recursively_tokenize_node(elt, [])
            values = tuple(map(node_tokenizer, node.value.elts))
        else:
            values = utils.recursively_tokenize_node(node.value, [])

        targets = node.targets if hasattr(node, "targets") else (node.target)
        for target in targets:
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

    def visit_AnnAssign(self, node):
        self.visit_Assign(node)

    # def visit_Name(self, node):
    #     if isinstance(node.ctx, ast.Del):
    #         pass  # TODO: Delete alias from tree
    #     elif isinstance(node.ctx, ast.Load):
    #         pass  # TODO: Increment count of node (beware of double counting)
    #
    #     return

    # TODO: Fix double-counting for all the default_handler visitors

    @utils.default_handler
    def visit_Call(self, node):
        return

    @utils.default_handler
    def visit_Attribute(self, node):
        # You could try searching up the node.parent.parent... path to find
        # out if attribute is inside a call node. If it is, let the call visiter
        # take care of it. If it isn't, then keep doing what you're doing.

        # Also possible that the best way to deal with this is by just having
        # one ast.Load visitor. Look into this more, i.e. what gets covered by
        # ast.Load.

        # Or ast.Expression?

        return

    @utils.default_handler
    def visit_Subscript(self, node):
        return

    @utils.default_handler
    def visit_Dict(self, node):
        return

    @utils.default_handler
    def visit_List(self, node):
        return

    @utils.default_handler
    def visit_Tuple(self, node):
        return

    @utils.default_handler
    def visit_Set(self, node):
        return

    @utils.default_handler
    def visit_Lambda(self, node):
        return

    ## Public Methods ##

    def trees(self, flattened=False):
        trees = {} if flattened else []
        for tree in self.forest:
            if flattened:
                trees[tree.id] = {}
                paths = tree.flatten()
                for path in paths[1:]:
                    print(path)
                    trees[tree.id]['.'.join(path[0])] = path[1]
            else:
                trees.append(tree.to_dict(debug=True))

        return trees
