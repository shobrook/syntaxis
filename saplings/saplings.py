#########
# GLOBALS
#########


import ast
import json
from collections import defaultdict


#################
# IDENTIFIER NODE
#################


class IdentifierNode(object):
    """
    Node object used in constructing the parse tree.

    @param id: name of the node.
    @param type: type of node, function call (call) or non-function call (name).
    @param aliases: all known aliases of the node, organized by context.
    @param children: list of child nodes.
    """

    def __init__(self, id, type="name", aliases=[], children=[]):
        self._id, self._type = id, type
        self._aliases, self._children = defaultdict(lambda: set()), []
        self._count = 1

        for alias in aliases:
            self.add_aliases({alias[0]}, alias[1])

        for child in children:
            self.add_child(child)

    def __repr__(self):
        return self._id

    def __iter__(self):
        return iter(self._children)

    def __eq__(self, node):
        if isinstance(node, IdentifierNode):
            return self._id == node.id and self._type == node.type

        return False

    def __ne__(self, node):
        if isinstance(node, IdentifierNode):
            return not self.__eq__(node)

        return True

    ## Instance Methods ##

    def add_aliases(self, aliases, context):
        # BUG: set() cast shouldn't be necessary
        self._aliases[context] = set(self._aliases[context])
        self._aliases[context] |= aliases

    def delete_alias(self, alias, context):
        self._aliases[context] = [
            id for id in self._aliases[context] if id != alias]

    def add_child(self, node):
        # QUESTION: Why did I do this?
        for child in self._children:
            if child == node:  # Child node already exists
                for ctx, aliases in node.aliases.items():
                    child.add_aliases(aliases, ctx)

                return

        self._children.append(node)

    def increment_count(self):
        self._count += 1

    def depth_first(self):
        yield self
        for node in self:
            yield from node.depth_first()

    def breadth_first(self):
        queue = [self]
        while queue:
            node = queue.pop(0)
            yield node
            for child in node.children:
                queue.append(child)

    def flatten(self):
        paths = [([self._id], self._count)]

        for node in self._children:
            useable_paths = node.flatten()
            for path in useable_paths:
                paths.append(([self._id] + path[0], node.count))

        return paths

    def to_dict(self, debug=True):
        children = [child.to_dict(debug) for child in self._children]
        if not debug:
            initial_dict = {
                "name": self._id,
                "attributes": {
                    "type": self._type,
                    "count": self._count
                }
            }

            if children:
                return {**initial_dict, **{"children": children}}

            return initial_dict

        initial_dict = {
            "id": self._id,
            "type": self._type,
            "count": self._count,
            "aliases": {ctx: aliases for ctx, aliases in dict(self._aliases).items() if aliases}
        }
        if children:
            return {**initial_dict, **{"children": children}}

        return initial_dict

    ## Properties ##

    @property
    def id(self):
        return self._id

    @property
    def type(self):
        return self._type

    @property
    def count(self):
        return self._count

    @property
    def aliases(self):
        return self._aliases

    @property
    def children(self):
        return self._children


#######
# ROOTS
#######


# Context management [DONE, untested]
# Handling subscripts
# Incorrect frequencies/over-counting
# Duplicate nodes [DONE, untested]
# Factoring in unique call arguments
# Tracking parameter and return types for user-defined functions and classes
# Filtering out built-in function calls (is this possible w/o knowing types?)

# IDEA: Tracking type outputs from functions and classes:
    # Treat function and class definitions as aliases for nodes. For instance,
        # from bs4 import BeautifulSoup
        # def x(y):
        #   return y.find_all()
        # x(BeautifulSoup(...))
    # "x" would be an alias for "find_all".
    # With classes, you'd have to look at __init__()s and inheritance to create
    # aliases.


class Roots(ast.NodeVisitor):
    """
    Generates and stores parse trees for every imported package in a Python
    program.

    @param tree: AST representation of the program.
    """

    def __init__(self, tree):
        self._context_stack = ["global"]  # Keeps track of current context
        self._forest = []  # Holds the root nodes of Package Trees

        self.visit(tree)

    ## Helper Methods ##

    def _update_context(self, context, subtree):
        """
        Wrapper method around generic_visit that updates the context stack before
        traversing a subtree, and pops from the stack when the traversal is finished.

        @param context: name of new context.
        @subtree: root node of subtree to traverse.
        """

        self._context_stack.append(context)
        self.generic_visit(subtree)
        self._context_stack.pop()

    def _context_to_string(self):
        """
        Joins the context stack into a period-separated string.

        @return: period-separated string representing the current context.
        """

        return '.'.join([ctx for ctx in self._context_stack if ctx != "assign"])

    @staticmethod
    def _recursive_node_tokenizer(node, tokens):
        """
        Takes an AST node and recursively unpacks it into it's constituent tokens.
        For example, if the input node is "x.y.z()", this algorithm will return
        [('x', "name"), ('y', "name"), ('z', "call")].

        @param node: AST node of type ast.Call, ast.Attribute, ast.Name, ast.Subscript,
        ast.ListComp, ast.DictComp, or ast.GeneratorExp.
        @param tokens: token accumulator.

        @return: list of tokens (tuples in the form (name, type)).
        """

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                tokens.append((node.func.id, "call"))
                return tokens[::-1]
            elif isinstance(node.func, ast.Attribute):
                tokens.append((node.func.attr, "call"))
                return Roots._recursive_node_tokenizer(
                    node=node.func.value,
                    tokens=tokens
                )
        elif isinstance(node, ast.Attribute):
            tokens.append((node.attr, "name"))
            return Roots._recursive_node_tokenizer(
                node=node.value,
                tokens=tokens
            )
        elif isinstance(node, ast.Name):
            tokens.append((node.id, "name"))
            return tokens[::-1]
        elif isinstance(node, ast.Subscript):
            slice = ''
            if isinstance(node.slice, ast.Index):
                if isinstance(node.slice.value, ast.Str):  # a["b"]
                    slice = ''.join(['"', node.slice.value.s, '"'])
                elif isinstance(node.slice.value, ast.Num):  # a[1]
                    slice = str(node.slice.value.n)
                else:  # a[Object()]
                    pass  # TODO
            elif isinstance(node.slice, ast.Slice):  # a[1:2]
                pass  # TODO
            elif isinstance(node.slice, ast.ExtSlice):  # a[1:2, 3]
                pass  # TODO

            tokens.append((''.join(['[', slice, ']']), "sub"))
            return Roots._recursive_node_tokenizer(
                node=node.value,
                tokens=tokens
            )
        elif isinstance(node, ast.ListComp):
            return None  # TODO
        elif isinstance(node, ast.DictComp):
            return None  # TODO
        elif isinstance(node, ast.GeneratorExp):
            return None  # TODO
        else:
            return None

    @staticmethod
    def _tokenized_node_to_string(tokenized_node):
        """
        # TODO: Write docstring
        """

        if isinstance(tokenized_node, tuple):
            return ''  # TODO

        token_stack = []
        for idx in range(len(tokenized_node)):
            token_id = tokenized_node[idx][0]
            token_type = tokenized_node[idx][1]

            not_last_token = idx < len(tokenized_node) - 1
            if not_last_token and tokenized_node[idx + 1][1] == "sub":
                for token in tokenized_node[idx + 1:]:
                    if token[1] == "sub":
                        token_id += token[0]

                    break

            if token_type == "sub":
                continue
            elif token_type == "name":
                # BUG: What about x[1]()?
                if not_last_token and tokenized_node[idx + 1][1] in ("name", "call"):
                    token_id += token_id + '.'
            elif token_type == "call":  # BUG: Doesn't factor in func args
                token_id += "()"

            token_stack.append(token_id)

        return ''.join(token_stack)

    @staticmethod
    def _assign_node_tokenizer(node):
        """
        # TODO: Write docstring
        """

        if isinstance(node, ast.Tuple):  # a, b, ... || c, d, ...
            return tuple(Roots._recursive_node_tokenizer(elt, [])
                         for elt in node.elts)

        return Roots._recursive_node_tokenizer(node, [])

    @staticmethod
    def _find_matching_node(subtree, id, type, ctx):
        """
        # TODO: Write docstring
        """

        if subtree:
            for node in subtree.breadth_first():
                # direct_alias = False
                # if id == node.id and type == node.type:
                #     direct_alias = True

                for alias in node.aliases[ctx]:
                    if id == alias:
                        return node

                ctx_path = ctx.split('.')
                for idx in range(1, len(ctx_path)):
                    for alias in node.aliases['.'.join(ctx_path[:-idx])]:
                        if id == alias:
                            return node

        return None

    def _process_tokenized_node(self, tokenized_node):
        """
        Takes a list of tokens (either call, name, or subscript nodes) and searches
        for an equivalent node path in the package forest. Once the tail of the known
        path has been reached, the algorithm create additional nodes and adds them
        to the path.

        @param tokenized_node: list of token tuples, (name, type).

        @return: list of nodes representing node paths.
        """

        # BUG: Edge case
        # An assignment is visited, and the target, x.y.z, is stored as an alias
        # for it's value (i.e. stored as "x.y.z"). But when you search for nodes,
        # you break apart the attributes and go step by step, thereby missing
        # concatenated aliases. Growing window as possible solution?

        node_stack = []
        curr_ctx = self._context_to_string()

        if tokenized_node:
            # TODO: Handle tuple assignments
            if isinstance(tokenized_node, tuple):
                pass
            else:  # [("var_name", "type"), ("func_name", "type"), ...]
                for idx in range(len(tokenized_node)):  # Iterate through node chain
                    not_last_token = idx < len(tokenized_node) - 1
                    token_name = tokenized_node[idx][0]
                    token_type = tokenized_node[idx][1]

                    # Merges all subscript tokens into one (i.e. name + slice1 + slice2 + ...)
                    if not_last_token and tokenized_node[idx + 1][1] == "sub":
                        for next_token in tokenized_node[idx + 1:]:
                            if next_token[1] == "sub":
                                token_name += next_token[0]

                            break

                    if not node_stack and idx == 0:  # Beginning of iteration
                        for pkg_tree in self._forest:
                            matched_node = Roots._find_matching_node(
                                subtree=pkg_tree,
                                id=token_name,
                                type=token_type,
                                ctx=curr_ctx
                            )
                            if matched_node:  # Found base node, initializing chain
                                node_stack.append(matched_node)
                                break
                    elif node_stack:  # Chain exists, continue adding to it
                        matched_node = Roots._find_matching_node(
                            subtree=node_stack[-1],
                            id=token_name,
                            type=token_type,
                            ctx=curr_ctx
                        )
                        if matched_node:  # Found child node
                            if not not_last_token:  # Increment tail of the chain
                                matched_node.increment_count()
                            node_stack.append(matched_node)
                        else:  # No child node found, creating one
                            child_node = IdentifierNode(
                                id=token_name,
                                type=token_type,
                                aliases=[(token_name, curr_ctx)]
                            )
                            node_stack[-1].add_child(child_node)
                            node_stack.append(child_node)
                    else:  # Base token doesn't exist, abort processing
                        break

        return node_stack

    ## Context Managers ##

    def visit_Global(self, node):
        """
        # TODO: Write docstring
        """

        self.generic_visit(node)  # TODO

    def visit_Nonlocal(self, node):
        """
        # TODO: Write docstring
        """

        self.generic_visit(node)  # TODO

    def visit_ClassDef(self, node):
        """
        # TODO: Write docstring
        """

        self._update_context('-'.join(["class", node.name]), node)

    def visit_FunctionDef(self, node):
        """
        # TODO: Write docstring
        """

        self._update_context('-'.join(["function", node.name]), node)

    def visit_AsyncFunctionDef(self, node):
        """
        # TODO: Write docstring
        """

        self._update_context('-'.join(["async_function", node.name]), node)

    def visit_Lambda(self, node):
        """
        # TODO: Write docstring
        """

        self._update_context("lambda", node)

    ## Visitors ##

    # TODO: Handle pkg.subpkg.subpkg ... cases
    # TODO: Create two nodes for each imported thing (call and name types)

    def visit_Import(self, node):
        """
        # TODO: Write docstring
        """

        for alias in node.names:
            alias_id = alias.asname if alias.asname else alias.name
            curr_ctx = self._context_to_string()
            module_exists = False

            for module_node in self._forest:
                if alias.name == module_node.id:  # Module tree already in forest
                    module_exists = True
                    alias_exists = any(
                        module_alias == alias_id for module_alias in module_node.aliases[curr_ctx])

                    module_node.increment_count()

                    if not alias_exists:  # Alias doesn't exist
                        module_node.add_aliases(
                            aliases={alias_id},
                            context=curr_ctx
                        )

                    break

            if not module_exists:  # Module tree doesn't exist
                self._forest.append(IdentifierNode(
                    id=alias.name,
                    aliases=[(alias_id, curr_ctx)]
                ))

        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """
        # TODO: Write docstring
        """

        module_exists = False
        curr_ctx = self._context_to_string()

        for module_node in self._forest:
            if node.module == module_node.id:  # Module tree already in forest
                module_exists = True

                for alias in node.names:
                    alias_exists = False
                    alias_id = alias.asname if alias.asname else alias.name

                    for child in module_node.children:
                        if alias.name == child.id:  # Child already exists
                            alias_exists = any(
                                child_alias == alias_id for child_alias in child.aliases[curr_ctx])

                            child.increment_count()

                            if not alias_exists:  # Alias doesn't exist
                                child.add_aliases(
                                    aliases={alias_id},
                                    context=curr_ctx
                                )
                                alias_exists = True

                            break

                    if not alias_exists:  # Module feature doesn't exist
                        module_node.add_child(IdentifierNode(
                            id=alias.name,
                            aliases=[(alias_id, curr_ctx)]
                        ))

                module_node.increment_count()
                break

        if not module_exists:  # Module doesn't exist
            module_node = IdentifierNode(id=node.module)

            for alias in node.names:
                alias_id = alias.asname if alias.asname else alias.name
                module_node.add_child(IdentifierNode(
                    id=alias.name,
                    aliases=[(alias_id, curr_ctx)]
                ))

            self._forest.append(module_node)

        self.generic_visit(node)

    def visit_Assign(self, node):
        """
        # TODO: Write docstring
        """

        targets = [Roots._assign_node_tokenizer(
            t) for t in node.targets]
        values = [Roots._assign_node_tokenizer(node.value)]

        curr_ctx = self._context_to_string()
        targ_val_map = list(zip(targets, values))
        for idx in range(len(targ_val_map)):
            target, value = targ_val_map[idx][0], targ_val_map[idx][1]

            targ_token_matches = self._process_tokenized_node(target)
            val_token_matches = self._process_tokenized_node(value)

            if targ_token_matches and val_token_matches:  # Known node reassigned to known node
                targ_tail, val_tail = targ_token_matches[-1], val_token_matches[-1]

                val_tail.add_aliases(
                    aliases={Roots._tokenized_node_to_string(target)},
                    context=curr_ctx,
                )
                targ_tail.delete_alias(
                    alias=Roots._tokenized_node_to_string(target),
                    context=curr_ctx
                )
            elif targ_token_matches and not val_token_matches:  # Known node reassigned to unknown node
                targ_tail = targ_token_matches[-1]
                targ_tail.delete_alias(
                    alias=Roots._tokenized_node_to_string(target),
                    context=curr_ctx
                )
            elif not targ_token_matches and val_token_matches:  # Unknown node assigned to known node
                val_tail = val_token_matches[-1]
                val_tail.add_aliases(
                    aliases={Roots._tokenized_node_to_string(target)},
                    context=curr_ctx
                )

        # IDEA: When, say, BeautifulSoup is first imported, we assume that it's
        # of both types (call and name) and add two children to bs4 node. Then
        # after the entire file is processed, we check to see if both nodes have
        # any children or a count > 0. If one doesn't, then it gets deleted.

        self._update_context("assign", node)

    def visit_Call(self, node):
        """
        # TODO: Write docstring
        """

        # NOTE: If _context_stack[:-1] == "assign", don't increment whatever you
        # find unless it doesn't exist yet

        self._process_tokenized_node(Roots._recursive_node_tokenizer(node, []))
        self.generic_visit(node)

    def visit_Attribute(self, node):
        """
        # TODO: Write docstring
        """

        ids = Roots._recursive_node_tokenizer(node, [])
        self._process_tokenized_node(ids)
        self.generic_visit(node)

    def visit_Name(self, node):
        """
        # TODO: Write docstring
        """

        if isinstance(node.ctx, ast.Del):
            pass  # TODO: Delete alias from tree
        elif isinstance(node.ctx, ast.Load):
            pass  # TODO: Increment count of node

        self.generic_visit(node)

    # def visit_Subscript(self, node):
    #     self.generic_visit(node)

    # TODO: Handle loop/comp contexts

    ## Public Methods ##

    def to_dict(self, debug=True):
        """
        # TODO: Write docstring
        """

        return [pkg_node.to_dict(debug) for pkg_node in self._forest]

    def flatten(self):
        """
        # TODO: Write docstring
        """

        flattened_tree = {}
        for pkg_node in self._forest:
            flattened_tree[pkg_node.id] = {}
            for path in pkg_node.flatten()[1:]:
                flattened_tree[pkg_node.id]['.'.join(path[0])] = path[1]

        return flattened_tree


###########
# HARVESTER
###########


class Harvester(ast.NodeVisitor):  # ast.NodeTransformer
    """
    # TODO: Write docstring
    """
    
    def __init__(self, tree):
        self._ast_root = tree

    ## Overloaded Methods ##

    def generic_visit(self, node):
        """
        # TODO: Write docstring
        """

        if any(isinstance(node, n_type) for n_type in self.__skip):
            pass
        elif not self.__nodes or any(isinstance(node, n_type) for n_type in self.__nodes):
            self.__retrieved.append(node)
            self.__freq_map[type(node).__name__] += 1
            # node = ast.copy_location(self.__transformer(node), node)

        ast.NodeVisitor.generic_visit(self, node)

    ## Private Methods ##

    def __reset_fields(self):
        """
        # TODO: Write docstring
        """

        self.__nodes, self.__skip = [], []
        self.__retrieved = []
        self.__freq_map = defaultdict(lambda: 0, {})
        self.__transformer = lambda node: node

    ## Public Methods ##

    def find(self, nodes=[], skip=[]):  # TODO: Add an `all` parameter
        """
        @param nodes: list of nodes to retrieve
        @param skip: list of subtrees to skip in the traversal

        Both parameters are optional, and by default find() will return a list of
        all nodes contained in the AST.

        @return: list of matching AST nodes
        """

        self.__reset_fields()
        self.__nodes, self.__skip = nodes, skip
        self.visit(self._ast_root)

        return self.__retrieved

    def get_freq_map(self, nodes=[], skip=[]):
        """
        @param nodes: list of node classes to analyze
        @param skip: list of subtrees to skip in the traversal

        Both parameters are optional, and by default get_freq_map() will return
        a dictionary containing all node types in the tree and their frequencies.

        @return: dictionary mapping node types to their frequency of occurence
        in the AST
        """

        self.__reset_fields()
        self.__nodes, self.__skip = nodes, skip
        self.visit(self._ast_root)

        return dict(self.__freq_map)

    def transform(nodes=[], transformer=lambda node: node, skip=[]):
        """
        # TODO: Write docstring
        """

        self.__reset_fields()
        self.__transformer = transformer
        self.__nodes, self.__skip = nodes, skip
        self.visit(self._ast_root)

        return self._ast_root

    def get_halstead(metric_name):
        """
        # TODO: Write docstring
        """

        pass

    def get_type(nodes):
        """
        # TODO: Write docstring
        """

        pass

    def get_roots():
        """
        # TODO: Write docstring
        """

        return Roots(self._ast_root)

    ## Properties ##

    @property
    def _root(self):
        return self._ast_root
