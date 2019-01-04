#########
# GLOBALS
#########


import ast
import json
import math
from collections import defaultdict
import utils
from forest import APIForest
from metrics import ProgramMetrics


######
# MAIN
######


class Saplings(object):
    def __init__(self, tree):
        """
        @param tree: either the path to a Python file or an already parsed AST
        for a Python file.
        """

        if isinstance(tree, ast.Module): # Already parsed Python AST
            self.tree = tree
        elif isinstance(tree, str): # Path to Python file
            self.tree = ast.parse(open(tree, 'r').read())
        else:
            raise Exception # TODO: Create custom exception

        self._api_forest = APIForest(self.tree).dependency_trees
        self._program_metrics = ProgramMetrics(self.tree)

    ## Public Methods ##

    def find(self, nodes=[]):
        """
        Both parameters are optional, and by default find() will return a list
        of all nodes contained in the AST.

        @param nodes: list of node types to retrieve.

        @return: list of matching AST nodes.
        """

        pass

    def get_freq_map(self, nodes=[]):
        """
        Both parameters are optional, and by default get_freq_map() will return
        a dictionary containing all node types in the tree and their
        frequencies.

        @param nodes: list of node types to analyze.

        @return: dictionary mapping node types to their frequency of occurence
        in the AST.
        """

        pass

    def get_halstead_metrics(self):
        halstead_metrics = self._program_metrics.halstead_metrics
        get_report = lambda metrics: {
            "operands": metrics.operands,
            "operators": metrics.operators,
            "distinct_operands": metrics.distinct_operands(),
            "distinct_operators": metrics.distinct_operators(),
            "vocabulary": metrics.vocabulary,
            "length": metrics.length,
            "calculated_length": metrics.calculated_length,
            "volume": metrics.volume,
            "difficulty": metrics.difficulty,
            "effort": metrics.effort,
            "time": metrics.time,
            "bugs": metrics.bugs
        }

        return {**get_report(halstead_metrics), **{
            "function_level_metrics": {
                func_metrics.context: get_report(func_metrics)
            for func_metrics in halstead_metrics.function_metrics}
        }}

    def get_cyclomatic_complexity(self):
        pass

    def get_method_couplings(self):
        pass

    def get_method_to_loc_map(self):
        pass

    def get_api_forest(self, flattened=False):
        dep_trees = {} if flattened else []
        for tree in self._api_forest:
            if flattened:
                dep_trees[tree.id] = {}
                for path in tree.flatten()[1:]:
                    dep_trees[tree.id]['.'.join(path[0])] = path[1]
            else:
                # NOTE: 'True' for debugging purposes
                dep_trees.append(tree.to_dict(True))

        return dep_trees

# TODO: Add NetworkX viz functionality

if __name__ == "__main__":
    import sys
    sys.path = sys.path[1:] + ['']

    with open("./output.json", 'w') as output:
        # saplings = Saplings("./test.py")
        saplings = Saplings("./cases.py")

        from pprint import pprint
        # pprint(saplings._program_metrics.method_deps)
        # pprint(saplings.get_halstead_metrics())

        output.write(json.dumps(saplings.get_api_forest()))


########################
# FLOATING CODE SNIPPETS
########################


# def export_json(tree):
#     return json.dumps(
#         DictExportVisitor().visit(tree),
#         sort_keys=True,
#         separators=(',', ':')
#     )
#
# class DictExportVisitor:
#     def visit(self, node):
#         method_name = "visit_" + type(node).__name__
#         method = getattr(self, method_name, self.default_visit)
#
#         return method(node)
#
#     def default_visit(self, node):
#         node_type = type(node).__name__
#         args = {"node": node_type}
#
#         # Visit fields
#         for field in node._fields:
#             method_name = "visit_field_" + node_type + '_' + field
#             method = getattr(self, method_name, self.default_visit_field)
#
#             args[field] = method(getattr(node, field))
#
#         # Visit attributes
#         for attr in node._attributes:
#             method_name = "visit_attribute_" + node_type + '_' + attr
#             method = getattr(self, method_name, self.default_visit_field)
#
#             args[attr] = method(getattr(node, attr, None))
#
#         return args
#
#     def default_visit_field(self, field):
#         if isinstance(field, ast.AST):
#             return self.visit(field)
#         elif isinstance(field, list) or isinstance(field, tuple):
#             return [self.visit(f) for f in field]
#
#         return field
#
#     # Special visitors
#
#     def visit_str(self, field):
#         return str(field)
#
#     def visit_Bytes(self, field):
#         return str(field.s)
#
#     def visit_NoneType(self, field):
#         return None
#
#     def visit_field_NameConstant_value(self, field):
#         return str(field)
#
#     def visit_field_Num_n(self, field):
#         if isinstance(field, int):
#             return {"node": "int", "n": field, "n_str": str(field)}
#         elif isinstance(field, float):
#             return {"node": "float", "n": field}
#         elif isinstance(field, complex):
#             return {"node": "complex", "n": field.real, "i": field.imag}
#
# with open("./ast.json", 'w') as ast_viz:
#     file_ast = ast.parse(open("./test.py").read())
#     ast_viz.write(export_json(file_ast))
