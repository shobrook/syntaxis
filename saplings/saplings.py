#########
# GLOBALS
#########


import ast
from typing import List, Dict
from collections import defaultdict


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


##############
# PACKAGE TREE
##############


class PackageTree(ast.NodeVisitor):
    def __init__(self, tree):
        self._context_stack = ["global"] # Keeps track of current context
        self._module_forest = []

        self.visit(tree)

    ## Helper Methods ##

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

    ## Import Visitors ##

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

    ## Other Visitors ##

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
                            func_node.add_child(IdentifierNode(
                                id=node.func.attr,
                                aliases=[AliasNode(
                                    id=node.func.attr,
                                    contexts={self._context_stack[-1]}
                                )]
                            ))

        self.generic_visit(node)

    def visit_Assign(self, node):
        self.generic_visit(node) # TODO

    ## Public Methods ##

    def to_dict(self):
        return [module_node.to_dict() for module_node in self._module_forest]


###########
# HARVESTER
###########


class Harvester(ast.NodeVisitor): # ast.NodeTransformer
    def __init__(self, tree):
        self._root = tree

    ## Overloaded Methods ##

    def generic_visit(self, node):
        if any(isinstance(node, n_type) for n_type in self.__skip):
            pass
        elif not self.__nodes or any(isinstance(node, n_type) for n_type in self.__nodes):
            self.__retrieved.append(node)
            self.__freq_map[type(node).__name__] += 1
            node = ast.copy_location(self.__transformer(node), node)

        ast.NodeVisitor.generic_visit(self, node)

    ## Private Methods ##

    def __reset_fields(self):
        self.__nodes, self.__skip = [], []
        self.__retrieved = []
        self.__freq_map = defaultdict(lambda: 0, {})
        self.__transformer = lambda node: node

    ## Public Methods ##

    def find(self, nodes=[], skip=[]): # TODO: Add an `all` parameter
        """
        @param nodes: list of nodes to retrieve
        @param skip: list of subtrees to skip in the traversal

        Both parameters are optional, and by default find() will return a list of
        all nodes contained in the AST.

        @return: list of matching AST nodes
        """

        self.__reset_fields()
        self.__nodes, self.__skip = nodes, skip
        self.visit(self._root)

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
        self.visit(self._root)

        return dict(self.__freq_map)

    def transform(nodes=[], transformer=lambda node: node, skip=[]):
        self.__reset_fields()
        self.__transformer = transformer
        self.__nodes, self.__skip = nodes, skip
        self.visit(self._root)

        return self._root

    def get_halstead(metric_name) -> float:
        pass

    def get_type(nodes):
        pass

    def get_pkg_tree(pkg_names=[]):
        pass

    ## Properties ##

    @property
    def _root(self):
        return self._root


#########
# TESTING
#########


if __name__ == "__main__":
    my_file = open("py_ast.py", 'r').read()
    my_ast = ast.parse(my_file)

    #my_harvester = Harvester(my_ast)
    my_pkg_forest = PackageTree(my_ast)

    #print(my_harvester.find(nodes=[ast.For]))
    #print(my_harvester.get_freq_map(nodes=[ast.Call, ast.For]))

    print(my_pkg_forest.to_dict())
