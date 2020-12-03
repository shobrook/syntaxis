# Standard Library
import ast
from collections import defaultdict


###########
# RENDERING
###########


HORIZ_EDGE = "+--"
VERT_EDGE = "|"
INDENT = "    "

def render_tree(node, level=0):
    if not level:
        pre = ""
    else:
        pre = " " +  INDENT * (level - 1) + HORIZ_EDGE + " "

    tree_repr = [(pre, node)]
    for i, child in enumerate(node.children):
        has_children = bool(child.children)
        has_sibling = i < len(node.children) - 1

        subtree_repr = render_tree(child, level + 1)
        for j, (pre, n) in enumerate(subtree_repr):
            if not j:
                continue

            if has_children and has_sibling: # Adds vertical edge
                insert_index = level * len(INDENT) + 1
                pre = pre[:insert_index] + VERT_EDGE + pre[insert_index + 1:]

            subtree_repr[j] = (pre, n)

        tree_repr += subtree_repr

    return tree_repr


######################
# TREE/GRAPH UTILITIES
######################


class ObjectNode(object):
    def __init__(self, id, children=[]):
        """
        Module tree node constructor. Each node represents a feature in a
        package's API. A feature is defined as an object, function, or variable
        that can only be used by importing the package.

        @param id: original identifier for the node.
        @param children: connected sub-nodes.
        """

        self.id = str(id)
        self.children = []
        self.count = 1

        for child in children:
            self.add_child(child)

    def __repr__(self):
        return self.id

    def __iter__(self):
        return iter(self.children)

    def __eq__(self, node):
        if isinstance(node, type(self)):
            return self.id == node.id

        return False

    def __ne__(self, node):
        return not self.__eq__(node)

    ## Instance Methods ##

    def increment_count(self):
        self.count += 1

    def add_child(self, node):
        for child in self.children:
            if child == node: # Child already exists
                child.increment_count()
                return child

        self.children.append(node)
        return node

    def depth_first(self):
        yield self
        for node in self:
            yield from node.depth_first()

    def breadth_first(self):
        node_queue = [self]
        while node_queue:
            node = node_queue.pop(0)
            yield node
            for child in node.children:
                node_queue.append(child)

    def paths(self):
        if not self.children:
            return [[self.id]]

        paths = []
        for child in self.children:
            for path in child.paths():
                paths.append([self.id] + path)

        return paths

    def to_dict(self, debug=False):
        default = {
            "id": self.id,
            "count": self.count
        }

        children = [child.to_dict(debug) for child in self.children]
        if children:
            return {**default, **{"children": children}}

        return default
