#########
# GLOBALS
#########


import ast
import utils


######
# MAIN
######


# [] Handle For and With contexts
# [X] Handle assignments in Try/Except/If/Else/While contexts
# [] Handle comprehensions and generator expressions (ignore assignments, they're too hard)
# [] Handle List/Dict/Set/Tuple assignments
# [] Infer input and return (or yield) types of user-defined functions (and classes)
# [] Inside funcs, block searching parent contexts for aliases equivalent to parameter names (unless it's self)
# [] Get rid of the type/type/type/... schema for searching nodes
# [] Debug the frequency analysis

# Store processed key/val pairs in a field that the assign handler can access
    # Only care about first-level dicts/arrays, not nested in other nodes

# Can't handle appends/dels b/c you don't know the prior size of the object

# BUG: Can't handle [1,2,3, ...][0].foo()

class APIForest(ast.NodeVisitor):
    def __init__(self, tree):
        # IDEA: The entire context/aliasing system can be refactored such that
        # the object holds a mapping from contexts to alive and dead nodes. This
        # could make search more efficient and at the very least make this code
        # easier to read.

        self.tree = tree

        # OPTIMIZE: Turns AST into doubly-linked AST
        for node in ast.walk(self.tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node

        self._context_stack = ["global"] # Keeps track of current context
        self.dependency_trees = [] # Holds root nodes of API usage trees

        self._keyvals = [] # Stores processed key/val pairs for access by the assignment handler
        self._in_conditional = False # Flag for whether inside conditional block or not

        self._context_to_string = lambda: '.'.join(self._context_stack)

        self.visit(self.tree)

    ## Utilities ##

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
            hash_nested_content = lambda: hash(str(adjunctive_content)) if adjunctive_content else hash(str(content)) # hash() or id()?
            if type == "args":
                for sub_tokens in content:
                    adjunctive_content.append(self._recursively_process_tokens(sub_tokens))
                content = hash_nested_content()
            elif type == "subscript":
                adjunctive_content = self._recursively_process_tokens(content)
                content = hash_nested_content()
            # elif type == "hashmap":
            #     pass # TODO
            # elif type == "array":
            #     pass # TODO

            flattened_tokens.append((content, type))

        for idx, token in enumerate(flattened_tokens):
            content, type = token
            token_id = utils.stringify_tokenized_nodes(flattened_tokens[:idx + 1])

            type_pattern = adjusted_type = "implicit"
            if type == "args":
                adjusted_id = "call"
            elif type == "subscript":
                adjusted_id = "sub"
            else:
                adjusted_id = content
                adjusted_type = "instance"
                type_pattern = "instance/implicit/module"

            if not idx: # Beginning of iteration; find base token
                for root in self.dependency_trees:
                    matching_node = utils.find_matching_node(
                        subtree=root,
                        id=token_id,
                        type_pattern="instance/module/implicit",
                        context=curr_context
                    )

                    if matching_node: # Found match for base token, pushing to stack
                        matching_node.increment_count()
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
                    matching_node.increment_count()
                    node_stack.append(matching_node)
                else: # No child node found, creating one
                    child_node = utils.Node(
                        id=adjusted_id,
                        type=adjusted_type,
                        context=curr_context,
                        alias=token_id
                    )

                    node_stack[-1].add_child(child_node)
                    node_stack.append(child_node)
            else: break # Base token doesn't exist, abort processing

        return node_stack

    def _process_assignment(self, target, value):
        """
        @param target:
        @param value:
        """

        curr_context = self._context_to_string()
        tokenized_target = utils.recursively_tokenize_node(target, [])

        targ_matches = self._recursively_process_tokens(tokenized_target) # LHS
        val_matches = self._recursively_process_tokens(value) # RHS

        alias = utils.stringify_tokenized_nodes(tokenized_target)
        add_alias = lambda node: node.add_alias(curr_context, alias)
        del_alias = lambda node: node.del_alias(curr_context, alias)

        def handle_conditional_assignments(nodes=[]):
            """
            """

            if not self._in_conditional:
                return

            outer_context = '.'.join(self._context_stack[:-1])
            for node in nodes:
                node.del_alias(outer_context, alias)

        if targ_matches and val_matches: # Known node reassigned to known node (K2 = K1)
            targ_node, val_node = targ_matches[-1], val_matches[-1]

            add_alias(val_node)
            del_alias(targ_node)
            handle_conditional_assignments([targ_node, val_node])
        elif targ_matches and not val_matches: # Known node reassigned to unknown node (U1 = K1)
            targ_node = targ_matches[-1]

            del_alias(targ_node)
            handle_conditional_assignments([targ_node])
        elif not targ_matches and val_matches: # Unknown node assigned to known node (K2 = U1)
            val_node = val_matches[-1]

            add_alias(val_node)
            handle_conditional_assignments([val_node])

    def _process_module(self, module, context, alias_root=True):
        """
        Takes the identifier for a module, sometimes a period-separated string
        of sub-modules, and searches the list of parse trees for a matching
        module. If no match is found, new module nodes are generated and
        appended to self.dependency_trees.

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

        for root in self.dependency_trees:
            matching_module = utils.find_matching_node(
                subtree=root,
                id=root_module,
                type_pattern="module"
            )

            if matching_module:
                term_node = matching_module
                break

        if not term_node:
            term_node = utils.Node(
                id=root_module,
                type="module",
                context=context,
                alias=root_module if alias_root else '' # For `from X import Y`
            )
            self.dependency_trees.append(term_node)

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
                new_sub_module = utils.Node(
                    id=sub_module,
                    type="instance",
                    context=context,
                    alias=sub_module_alias if alias_root else '' # For `from X.Y import Z`
                )
                term_node.add_child(new_sub_module)
                term_node = new_sub_module

        return term_node

    ## Overloaded Methods ##

    # @utils.context_manager
    # def visit_Global(self, node):
    #     # IDEA: Save pre-state of context_stack, set to ["global"],
    #     # then set back to pre-state
    #     return

    # @utils.context_manager
    # def visit_Nonlocal(self, node):
    #     return

    @utils.context_manager
    def visit_ClassDef(self, node):
        pass

    @utils.context_manager
    def visit_FunctionDef(self, node):
        pass

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    # @utils.context_manager(False)
    # def visit_Lambda(self, node):
    #     return

    @utils.context_manager
    def visit_If(self, node):
        pass # TODO: Handle else/elif manually

    @utils.context_manager
    def visit_Try(self, node):
        pass # TODO: Handle ExceptHandler manually

    @utils.context_manager
    def visit_ExceptHandler(self, node):
        pass

    @utils.context_manager
    def visit_While(self, node):
        pass

    # TODO: Ignore relative imports and * imports

    @utils.visitor(True)
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

    @utils.visitor(True)
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
                module_node.add_child(utils.Node(
                    id=alias.name,
                    type="instance",
                    context=curr_context,
                    alias=alias_id
                ))

    @utils.visitor(False)
    def visit_Assign(self, node):
        curr_context = self._context_to_string()

        if isinstance(node.value, ast.Tuple):
            node_tokenizer = lambda elt: utils.recursively_tokenize_node(elt, [])
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

    @utils.visitor(False) # QUESTION: Needed when visit_Assign is already decorated?
    def visit_AnnAssign(self, node):
        self.visit_Assign(node)

    @utils.visitor(False)
    @utils.default_handler
    def visit_Call(self, node):
        tokens = utils.recursively_tokenize_node(node, [])
        print("Tokens:", tokens)
        nodes = self._recursively_process_tokens(tokens)

    @utils.visitor(False)
    @utils.default_handler
    def visit_Attribute(self, node):
        # You could try searching up the node.parent.parent... path to find
        # out if attribute is inside a call node. If it is, let the call visiter
        # take care of it. If it isn't, then keep doing what you're doing.

        # Also possible that the best way to deal with this is by just having
        # one ast.Load visitor. Look into this more, i.e. what gets covered by
        # ast.Load.

        return

    @utils.visitor(False)
    @utils.default_handler
    def visit_Subscript(self, node):
        return

    @utils.visitor(True)
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Del):
            pass  # TODO: Delete alias from tree
        elif isinstance(node.ctx, ast.Load):
            pass  # TODO: Increment count of node (beware of double counting)

    @utils.visitor(False)
    @utils.default_handler
    def visit_Dict(self, node):
        return

    @utils.visitor(False)
    @utils.default_handler
    def visit_List(self, node):
        return

    @utils.visitor(False)
    @utils.default_handler
    def visit_Tuple(self, node):
        return

    @utils.visitor(False)
    @utils.default_handler
    def visit_Set(self, node):
        return

    @utils.visitor(True)
    def visit_comprehension(self, node):
        return
