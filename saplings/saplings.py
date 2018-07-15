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
        pass


###########
# HARVESTER
###########


class Harvester(ast.NodeVisitor): # ast.NodeVisitor
    def __init__(self, tree):
        self._root = tree
        self.__context_stack = ["global"] # Keeps track of current context

    ## Overloaded Methods ##

    def generic_visit(self, node):
        if any(isinstance(node, n_type) for n_type in self.__skip):
            pass
        elif not self.__nodes or any(isinstance(node, n_type) for n_type in self.__nodes):
            self.__retrieved.append(node)
            self.__freq_map[type(node).__name__] += 1

        ast.NodeVisitor.generic_visit(self, node)

    ## Private Methods ##

    def __reset_fields(self):
        self.__context_stack = ["global"]
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
        pass

    def get_halstead(metric_name) -> float:
        pass

    def get_type(nodes):
        pass

    def get_pkg_tree(pkg_names=[]):
        pass


#########
# TESTING
#########


if __name__ == "__main__":
    my_file = open("harvester.py", 'r').read()
    my_ast = ast.parse(my_file)
    my_harvester = Harvester(my_ast)

    print(my_harvester.find(nodes=[ast.For]))
    print(my_harvester.get_freq_map(nodes=[ast.Call, ast.For]))
