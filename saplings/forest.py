#########
# GLOBALS
#########


import ast
from collections import defaultdict

# Local modules
import utils


##########
# REDESIGN
##########


# Input is an AST.
# ast.NodeVisitor performs a depth-first traversal on the AST.
# Output is a forest; each tree represents the subset of an imported package’s
# API that was used in the program. Each node includes a frequency value. Nodes
# are not statically typed.

# The Transducer class should hold the following mapping:
    # {
    #     "context1": {
    #         "alias1": Node(),
    #         ...
    #     },
    #     ...
    # }
# QUESTION: Can a single alias refer to multiple nodes at the same time, in one context? 


######
# MAIN
######


# [] Rename forest.py to transducer.py
# [] Remove "module" type from nodes; actually, remove all types
# [] Infer input and return types of user-defined functions (and classes)
# [] Inside funcs, block searching parent contexts for aliases equivalent to parameter names (unless it's self)
# [] Debug the frequency analysis

# Handling Functions #
# - Save defined function names and their return types in a field (search field when processing tokens)
# - Is processing the input types of functions as simple as adding the parameter name as an alias for the input node,
#   then calling self.visit() on the function node? You could only modify the input node and it's children, cuz everything
#   else in the function body has already been processed.

class Transducer(ast.NodeVisitor):
    def __init__(self, input_tree):
        # IDEA: The entire context/aliasing system can be refactored such that
        # a single dict holds a mapping from contexts to alive and dead nodes.
        # This could make search more efficient and at the very least make this
        # code easier to read.

        self.tree = input_tree
        self.forest = [] # Holds root nodes of output trees

        # # OPTIMIZE: Turns AST into doubly-linked tree
        # for node in ast.walk(self.tree):
        #     for child in ast.iter_child_nodes(node):
        #         child.parent = node

        self._context_stack = ["global"] # Keeps track of current context
        self._context_to_string = lambda: '.'.join(self._context_stack)

        # QUESTION: Is this needed?
        self._in_conditional = False # Flag for whether inside conditional block or not
        # Holds functions that clear aliases manipulated inside a conditional block
        self._conditional_handlers = defaultdict(lambda: [])

        self.visit(self.tree)

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
                        continue # TODO: Block search for targets when processing elts

                    self._recursively_process_tokens(sub_token[0])

                return True # TEMP: Return an alias map, or something
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
        curr_context = self._context_to_string()

        # Flattens nested tokens
        flattened_tokens = []
        for idx, token in enumerate(tokens): # QUESTION: Does this need to be in a separate loop?
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

        for idx, token in enumerate(flattened_tokens):
            content, type = token

            # BUG: Can't handle something like [1,2,3, ...][0].foo()
            if type in ("comprehension", "array", "hashmap"):
                break

            token_id = utils.stringify_tokenized_nodes(flattened_tokens[:idx + 1])
            if not idx: # Beginning of iteration; find base token
                for root in self.forest:
                    matching_node = utils.find_matching_node(
                        subtree=root,
                        id=token_id,
                        type_pattern="instance/module",
                        context=curr_context
                    )

                    if matching_node: # Found match for base token, pushing to stack
                        if not no_increment:
                            matching_node.increment_count()

                        node_stack.append(matching_node)
                        break
            elif node_stack: # Stack exists, continue pushing to it
                matching_node = utils.find_matching_node(
                    subtree=node_stack[-1],
                    id=token_id,
                    type_pattern="instance",
                    context=curr_context
                )

                if matching_node: # Found child node
                    if not no_increment:
                        matching_node.increment_count()
                    node_stack.append(matching_node)
                else: # No child node found, creating one
                    child_node = utils.Node(
                        id=content,
                        type="instance",
                        context=curr_context,
                        alias=token_id
                    )

                    # If in_conditional, and not an assignment, then you wanna get rid of the if contexts...

                    node_stack[-1].add_child(child_node)
                    node_stack.append(child_node)
            else: # Base token doesn't exist, abort processing
                break

        return node_stack

    def _process_assignment(self, target, value):
        """
        @param target: AST node on the left-hand-side of the assignment.
        @param value: tokenized AST node on the right-hand-side of the assignment.
        """

        is_data_struct = len(value) == 1 and value[0][1] in ("comprehension", "array", "hashmap")

        curr_context = self._context_to_string()
        parent_context = '.'.join(self._context_stack[:-1])
        tokenized_target = utils.recursively_tokenize_node(target, [])

        targ_matches = self._recursively_process_tokens(tokenized_target) # LHS
        val_matches = self._recursively_process_tokens(value, is_data_struct=is_data_struct) # RHS

        alias = utils.stringify_tokenized_nodes(tokenized_target) # BUG: This is stringifying the non-flattened list of tokens
        add_alias = lambda node: node.add_alias(curr_context, alias)
        del_alias = lambda node: node.del_alias(curr_context, alias)

        def del_conditional_aliases(nodes=[]):
            # QUESTION: Do you even need an _in_conditional field?
            if not self._in_conditional:
                return

            conditional_handler = lambda node: node.del_alias(parent_context, alias)
            wrapper = lambda: list(map(conditional_handler, nodes))
            self._conditional_handlers[curr_context].append(wrapper)

        # TODO: When you delete an alias after reassignment, delete `alias`
        # subscripts too. i.e. x = [1,2,3] then x is reassigned, and then nodes
        # with x[0] and x[1] as aliases won't get deleted but need to be.

        # TODO: Handle aliasing for is_data_struct case

        if targ_matches and val_matches: # Known node reassigned to known node (K2 = K1)
            targ_node, val_node = targ_matches[-1], val_matches[-1]

            add_alias(val_node)
            del_alias(targ_node)
            del_conditional_aliases([targ_node, val_node])
        elif targ_matches and not val_matches: # Known node reassigned to unknown node (K1 = U1)
            targ_node = targ_matches[-1]

            del_alias(targ_node)
            del_conditional_aliases([targ_node])
        elif not targ_matches and val_matches: # Unknown node assigned to known node (U1 = K1)
            val_node = val_matches[-1]

            add_alias(val_node)
            del_conditional_aliases([val_node])

    def _process_module(self, module, context, alias_root=True):
        """
        Takes the identifier for a module, sometimes a period-separated string
        of sub-modules, and searches the API forest for a matching module. If no
        match is found, new module nodes are generated and appended to
        self.forest.

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

        for root in self.forest:
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
            self.forest.append(term_node)

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

    ## Context Handlers ##

    # @utils.context_handler
    # def visit_Global(self, node):
    #     # IDEA: Save pre-state of context_stack, set to ["global"],
    #     # then set back to pre-state
    #     return

    # @utils.context_handler
    # def visit_Nonlocal(self, node):
    #     return

    # TODO: Handle function/class reassignments (and declarations within conditionals)

    @utils.context_handler
    def visit_ClassDef(self, node):
        pass

    @utils.context_handler
    def visit_FunctionDef(self, node):
        pass

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    ## Control Flow Handlers ##

    @utils.conditional_handler
    def visit_If(self, node):
        contexts = []
        for else_node in node.orelse:
            else_context = ''.join(["If", str(else_node.lineno)])
            contexts.append('.'.join(self._context_stack + [else_context]))
            self.visit_If(else_node)

        return contexts

    @utils.conditional_handler
    def visit_Try(self, node):
        contexts = []
        for except_handler in node.handlers:
            except_context = ''.join(["ExceptHandler", str(except_handler.lineno)])
            contexts.append('.'.join(self._context_stack + [except_context]))
            self.visit_ExceptHandler(except_handler)

        return contexts

        # QUESTION: How to handle node.orelse and node.finalbody?
            # node.finalbody is where all final assignments are made, and should
            # persist into the parent context

    @utils.conditional_handler
    def visit_ExceptHandler(self, node):
        # TODO: Handle `name` assignments (i.e. except Exception as e)
        return []

    @utils.conditional_handler
    def visit_While(self, node):
        return []

    def visit_For(self, node):
        curr_context = self._context_to_string()

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
        def get_nodes(n):
            tokens = utils.recursively_tokenize_node(n, [])
            alias = utils.stringify_tokenized_nodes(tokens)
            nodes = self._recursively_process_tokens(tokens)

            if not nodes:
                return

            return nodes[-1], alias

        if node.optional_vars:
            curr_context = self._context_to_string()

            value_node, value_alias = get_nodes(node.context_expr)
            target_node, target_alias = get_nodes(node.optional_vars) # Name, Tuple, or List

            value_node.add_alias(curr_context, target_alias)
            target_node.del_alias(curr_context, target_alias)

    ## Aliasing Handlers ##

    def visit_Import(self, node):
        curr_context = self._context_to_string()
        for module in node.names:
            if module.name.startswith('.'): # Ignores relative imports
                continue

            alias = module.asname if module.asname else module.name
            module_leaf_node = self._process_module(
                module=module.name,
                context=curr_context,
                alias_root=not bool(module.asname)
            )

            module_leaf_node.add_alias(curr_context, alias)

    def visit_ImportFrom(self, node):
        if node.level: # Ignores relative imports
            return

        curr_context = self._context_to_string()
        module_node = self._process_module(
            module=node.module,
            context=curr_context,
            alias_root=False
        )

        for alias in node.names:
            if alias.name == '*': # Ignore star imports
                continue

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

    def visit_Assign(self, node):
        curr_context = self._context_to_string()

        if isinstance(node.value, ast.Tuple):
            node_tokenizer = lambda elt: utils.recursively_tokenize_node(elt, [])
            values = tuple(map(node_tokenizer, node.value.elts))
        else:
            values = utils.recursively_tokenize_node(node.value, [])

        if curr_context == "global.FunctionDef#scrape_files277":
            print(values)

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

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Del):
            pass  # TODO: Delete alias from tree
        elif isinstance(node.ctx, ast.Load):
            pass  # TODO: Increment count of node (beware of double counting)

        return

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
