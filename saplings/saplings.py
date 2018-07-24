#########
# GLOBALS
#########


import ast
from collections import defaultdict


###########
# HARVESTER
###########


class Harvester(ast.NodeVisitor): # ast.NodeTransformer
    def __init__(self, tree):
        self._ast_root = tree

    ## Overloaded Methods ##

    def generic_visit(self, node):
        if any(isinstance(node, n_type) for n_type in self.__skip):
            pass
        elif not self.__nodes or any(isinstance(node, n_type) for n_type in self.__nodes):
            self.__retrieved.append(node)
            self.__freq_map[type(node).__name__] += 1
            #node = ast.copy_location(self.__transformer(node), node)

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
        self.__reset_fields()
        self.__transformer = transformer
        self.__nodes, self.__skip = nodes, skip
        self.visit(self._ast_root)

        return self._ast_root

    def get_halstead(metric_name) -> float:
        pass

    def get_type(nodes):
        pass

    def get_pkg_tree(pkg_names=[]):
        pass

    ## Properties ##

    @property
    def _root(self):
        return self._ast_root
