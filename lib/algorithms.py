#########
# GLOBALS
#########


import os
import ast
import yaml

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
SYNTAX = yaml.load(open('/'.join([CURR_DIR, "syntax.yml"]), 'r'))["Python"]


#########
# HELPERS
#########


class AliasNode(object):
    def __init__(self, id, contexts={}):
        self._id, self._contexts = id, contexts

        # QUESTION: Include name and asname, where both are other alias nodes (like a doubly linked list)?
        # IDEA: For each context, include the full path (i.e. "global.funcion.class.function")
        # to account for things like function currying


    def add_context(self, context):
        self._contexts.add(context)


    def remove_context(self, context):
        self._contexts.remove(context)


    def to_dict(self):
        return {
            "id": self._id,
            "contexts": self._contexts
        }


    @property
    def id(self):
        return self._id


    @property
    def contexts(self):
        return self._contexts


class IdentifierNode(object):
    def __init__(self, id, aliases=[], children=[]):
        self._id, self._count = id, 1
        self._aliases, self._children = aliases, []
        if children != []:
            for child in children:
                self.add_child(child)


    def add_alias(self, alias_node):
        self._aliases.append(alias_node)


    def add_child(self, identifier_node):
        assert isinstance(identifier_node, IdentifierNode)
        self._children.append(identifier_node)


    def increment_count(self):
        self._count += 1


    def to_dict(self):
        return {
            "id": self._id,
            "count": self._count,
            "aliases": [alias.to_dict() for alias in self._aliases],
            "children": [child.to_dict() for child in self._children]
        }


    @property
    def id(self):
        return self._id


    @property
    def count(self):
        return self._count


    @property
    def aliases(self):
        return self._aliases


    @property
    def children(self):
        return self._children


######
# MAIN
######


class AnalyzedAST(ast.NodeVisitor):
    def __init__(self, file):
        self._tree = ast.parse(open(file, 'r').read())
        self._context_stack = ["global"] # Keeps track of current context

        self._module_forest = []
        self._built_in_funcs = {id: 0 for id in SYNTAX["built-ins"]}
        self._literals = {id: 0 for id in SYNTAX["literals"]}
        self._binary_ops = {id: 0 for id in SYNTAX["binary operations"]}
        self._subscripts = {id: 0 for id in SYNTAX["subscripts"]}
        self._comprehensions = {id: 0 for id in SYNTAX["comprehensions"]}
        self._statements = {id: 0 for id in SYNTAX["statements"]}
        self._control_flow = {id: 0 for id in SYNTAX["control flow"]}
        self._func_and_class_defs = {id: 0 for id in SYNTAX["definitions"]}
        self._async_ops = {id: 0 for id in SYNTAX["asynchronous"]}

        self.visit(self._tree)


    ### Helpers ##


    def generic_visit(self, node):
        node_type = type(node).__name__

        # Built-in Functions
        if isinstance(node, ast.Call):
            if hasattr(node.func, "id"):
                func_id = node.func.id
            elif hasattr(node.func, "name"):
                func_id = node.func.name
            else:
                func_id = ''

            if func_id in SYNTAX["built-ins"]:
                self._built_in_funcs[func_id] += 1

        # Other Syntax Features
        if node_type in SYNTAX["literals"]:
            self._literals[node_type] += 1
        elif node_type in SYNTAX["binary operations"]:
            self._binary_ops[node_type] += 1
        elif node_type in SYNTAX["subscripts"]:
            self._subscripts[node_type] += 1
        elif node_type in SYNTAX["comprehensions"]:
            self._comprehensions[node_type] += 1
        elif node_type in SYNTAX["statements"]:
            self._statements[node_type] += 1
        elif node_type in SYNTAX["control flow"]:
            self._control_flow[node_type] += 1
        elif node_type in SYNTAX["definitions"]:
            self._func_and_class_defs[node_type] += 1
        elif node_type in SYNTAX["asynchronous"]:
            self._async_ops[node_type] += 1

        ast.NodeVisitor.generic_visit(self, node)


    def _update_context(self, context_name, node):
        self._context_stack.append(context_name)
        self.generic_visit(node)
        self._context_stack.pop()


    def _recursive_alias_search(self, node, target_id):
        for alias in node.aliases:
            if alias.id == target_id: # and self._context_stack[-1] in alias.contexts
                return node

        for child in node.children:
            return self._recursive_alias_search(child, target_id)


    ## Context Managers ##


    def visit_Global(self, node):
        self.generic_visit(node) # TODO


    def visit_Nonlocal(self, node):
        self.generic_visit(node) # TODO


    def visit_ClassDef(self, node):
        self._update_context("class", node)


    def visit_FunctionDef(self, node):
        self._update_context("function", node)


    def visit_AsyncFunctionDef(self, node):
        self._update_context("function", node)


    def visit_Lambda(self, node):
        self._update_context("function", node)


    ## Import Handlers ##


    def visit_Import(self, node):
        for alias in node.names:
            alias_id = alias.asname if alias.asname else alias.name
            current_context = self._context_stack[-1]
            module_exists = False

            for module_node in self._module_forest:
                if alias.name == module_node.id: # Module tree already in forest
                    module_exists = True
                    alias_exists = False

                    for module_alias in module_node.aliases:
                        if module_alias.id == alias_id: # Alias already exists
                            alias_exists = True
                            if current_context in module_alias.contexts: # Alias already exists in current context
                                break
                            else: # Alias exists but not in current context
                                module_alias.add_context(current_context)

                    module_node.increment_count()
                    if not alias_exists: # Alias doesn't exist
                        module_node.add_alias(AliasNode(id=alias_id, contexts={current_context}))

                    break

            if not module_exists: # Module tree doesn't exist
                self._module_forest.append(IdentifierNode(id=alias.name, aliases=[AliasNode(id=alias_id, contexts={current_context})]))

        self.generic_visit(node)


    def visit_ImportFrom(self, node):
        module_exists = False
        current_context = self._context_stack[-1]

        for module_node in self._module_forest:
            if node.module == module_node.id: # Module tree already in forest
                module_exists = True

                for alias in node.names:
                    alias_exists = False
                    alias_id = alias.asname if alias.asname else alias.name

                    for child in module_node.children:
                        if alias.name == child.id: # Module feature already exists
                            for child_alias in child.aliases:
                                if child_alias.id == alias_id: # Alias already exists
                                    alias_exists = True
                                    if current_context in child_alias.contexts: # Alias exists in current context
                                        break
                                    else:
                                        child_alias.add_context(current_context)

                            child.increment_count()
                            if not alias_exists: # Alias doesn't exist
                                child.add_alias(AliasNode(id=alias_id, contexts={current_context}))
                                alias_exists = True

                            break

                    if not alias_exists: # Module feature doesn't exist
                        module_node.add_child(IdentifierNode(id=alias.name, aliases=[AliasNode(id=alias_id, contexts={current_context})]))

                module_node.increment_count()
                break

        if not module_exists: # Module doesn't exist
            module_node = IdentifierNode(id=node.module)

            for alias in node.names:
                alias_id = alias.asname if alias.asname else alias.name
                module_node.add_child(IdentifierNode(id=alias.name, aliases=[AliasNode(id=alias_id, contexts={current_context})]))

            self._module_forest.append(module_node)

        self.generic_visit(node)


    ## Alias Trackers ##


    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            for module_node in self._module_forest:
                func_node = self._recursive_alias_search(module_node, node.func.id)
                if func_node != None:
                    func_node.increment_count()
        elif isinstance(node.func, ast.Attribute):
            for module_node in self._module_forest:
                if isinstance(node.func.value, ast.Name):
                    func_node = self._recursive_alias_search(module_node, node.func.value.id)
                    if func_node != None:
                        attr_node = self._recursive_alias_search(module_node, node.func.attr)
                        if attr_node != None:
                            attr_node.increment_count()
                        else:
                            func_node.add_child(IdentifierNode(id=node.func.attr, aliases=[AliasNode(id=node.func.attr, contexts={self._context_stack[-1]})]))

        self.generic_visit(node)


    def visit_Assign(self, node):
        self.generic_visit(node) # TODO


    ## Public ##


    @property
    def syntax_features(self):
        return {
            "literals": self._literals,
            "binary_ops": self._binary_ops,
            "subscripts": self._subscripts,
            "comprehensions": self._comprehensions,
            "statements": self._statements,
            "control_flow": self._control_flow,
            "func_and_class_defs": self._func_and_class_defs,
            "async_ops": self._async_ops
        }


    @property
    def std_lib_features(self):
        return {
            "built_in_funcs": self._built_in_funcs
        } # TODO: Add more


    @property
    def technologies(self):
        return [module_node.to_dict() for module_node in self._module_forest]
