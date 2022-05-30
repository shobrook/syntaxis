# Standard Library
import ast

# Local Modules
import saplings.tokenization as tkn
from saplings.entities import ObjectNode
# import tokenization as tkn
# from entities import ObjectNode


def find_matching_node(subtree, name):
    for node in subtree.breadth_first():
        if node.name == name:
            return node

    return None


def attribute_chain_handler(func):
    def wrapper(self, node):
        self._process_node(node)

    return wrapper


def delete_sub_aliases(targ_str, namespace):
    """
    If `my_var` is an alias, then `my_var()` and `my_var.attr` are
    considered sub-aliases. This function takes aliases that have been
    deleted or reassigned and deletes all of its sub-aliases.

    Parameters
    ----------
    targ_str : string
        string representation of the target node in the assignment
    """

    for alias in list(namespace.keys()):
        for sub_alias_signifier in ('(', '.'):
            if alias.startswith(targ_str + sub_alias_signifier):
                del namespace[alias]
                break


def create_decorator_call_node(decorator_list, args):
    if not decorator_list:
        return args

    decorator = decorator_list.pop()
    return create_decorator_call_node(
        decorator_list,
        ast.Call(func=decorator, args=[args], keywords=[])
    )


def consolidate_call_nodes(node, parent=None):
    for child in node.children:
        consolidate_call_nodes(child, node)

    if node.name == "()":
        parent.is_callable = True
        parent.frequency += node.frequency
        parent.children.remove(node)
        for child in node.children:
            child.order += 1
            parent.children.append(child)


def stringify_node(node):
    tokens = tkn.recursively_tokenize_node(node, [])
    node_str = tkn.stringify_tokenized_nodes(tokens)

    return node_str


def diff_and_clean_namespaces(namespace, other_namespace):
    diff = []
    for name, entity in namespace.items():
        if name not in other_namespace:
            diff.append(name)
        elif other_namespace[name] != entity:
            diff.append(name)

    for name in diff:
        if name not in namespace:
            continue

        del namespace[name]
        delete_sub_aliases(name, namespace)


def init_namespace(object_ids):
    object_hierarchies = []
    init_namespace = {}
    for object_id in object_ids:
        object_node = ObjectNode(object_id, order=-1, frequency=0)
        init_namespace[object_id] = object_node
        object_hierarchies.append(object_node)

    return {
        "object_hierarchies": object_hierarchies,
        "namespace": init_namespace
    }
