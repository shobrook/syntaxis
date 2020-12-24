# Standard Library
import ast
from collections import defaultdict

HORIZ_EDGE = "+--"
VERT_EDGE = "|"
INDENT = "    "


def render_tree(node, level=0):
    if not level:
        pre = ""
    else:
        pre = " " + INDENT * (level - 1) + HORIZ_EDGE + " "

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


def dictify_tree(node):
    d = {node.name: {
        "is_callable": node.is_callable,
        "order": node.order,
        "frequency": node.frequency,
        "children": []
    }}
    for child in node.children:
        d[node.name]["children"].append(dictify_tree(child))

    return d
