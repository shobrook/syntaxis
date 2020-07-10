#########
# GLOBALS
#########


import ast
import json
import math
from collections import defaultdict
import utils
from forest import Transducer
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

        self._forest = Transducer(self.tree).forest
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

    def get_freq_map(self, nodes=[], built_ins=False, skip=[]):
        """
        Both parameters are optional, and by default get_freq_map() will return
        a dictionary containing all node types in the tree and their
        frequencies.

        @param nodes: list of node types to analyze.

        @return: dictionary mapping node types to their frequency of occurence
        in the AST.
        """

        if built_ins:
            func_map = defaultdict(lambda: 0)
            for node in nodes:
                func_id = ''
                if hasattr(node.func, "id"):
                    func_id = node.func.id
                elif hasattr(node.func, "name"):
                    func_id = node.func.name

                if func_id in self.built_in_func_names:
                    func_map[func_id] += 1

            return dict(func_map)

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
        for tree in self._forest:
            if flattened:
                dep_trees[tree.id] = {}
                for path in tree.flatten()[1:]:
                    dep_trees[tree.id]['.'.join(path[0])] = path[1]
            else:
                # NOTE: 'True' for debugging purposes
                dep_trees.append(tree.to_dict(True))

        return dep_trees

    @property
    def literals(self):
        return [
            ast.Bytes,
            ast.Num,
            ast.Str,
            ast.JoinedStr,
            ast.List,
            ast.Tuple,
            ast.Set,
            ast.Dict,
            ast.Ellipsis,
            ast.NameConstant
        ]

    @property
    def variables(self):
        return [ast.Name, ast.Load, ast.Store, ast.Del, ast.Starred]

    @property
    def unary_ops(self):
        return [ast.UAdd, ast.USub, ast.Not, ast.Invert]

    @property
    def binary_ops(self):
        return [
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
            ast.LShift,
            ast.RShift,
            ast.BitOr,
            ast.BitXor,
            ast.BitAnd,
            ast.MatMult
        ]

    @property
    def boolean_ops(self):
        return [ast.And, ast.Or]

    @property
    def comparisons(self):
        return [
            ast.Compare,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.Is,
            ast.IsNot,
            ast.In,
            ast.NotIn
        ]

    @property
    def misc_expressions(self):
        return [ast.Call, ast.IfExp, ast.Attribute]

    @property
    def subscripts(self):
        return [ast.Subscript, ast.Index, ast.Slice, ast.ExtSlice]

    @property
    def comprehensions(self):
        return [ast.ListComp, ast.SetComp, ast.DictComp]

    @property
    def generators(self):
        return [ast.GeneratorExp, ast.Yield, ast.YieldFrom]

    @property
    def statements(self):
        return [ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete, ast.Pass]

    @property
    def exception_handling(self):
        return [ast.Raise, ast.Assert, ast.Try, ast.ExceptHandler]

    @property
    def imports(self):
        return [ast.Import, ast.ImportFrom, ast.alias]

    @property
    def control_flow(self):
        return [ast.If, ast.For, ast.While, ast.Break, ast.Continue, ast.With]

    @property
    def funcs_and_classes(self):
        return [
            ast.ClassDef,
            ast.FunctionDef,
            ast.Lambda,
            ast.arguments,
            ast.arg,
            ast.Return,
            ast.Global,
            ast.Nonlocal
        ]

    @property
    def async(self):
        return [ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith]

    @property
    def built_in_func_names(self):
        return [
            "abs",
            "all",
            "any",
            "ascii",
            "bin",
            "bool",
            "bytearray",
            "bytes",
            "callable",
            "chr",
            "classmethod",
            "compile",
            "complex",
            "delattr",
            "dict",
            "dir",
            "divmod",
            "enumerate",
            "eval",
            "exec",
            "filter",
            "float",
            "format",
            "frozenset",
            "getattr",
            "globals",
            "hasattr",
            "hash",
            "help",
            "hex",
            "id",
            "input",
            "int",
            "isinstance",
            "issubclass",
            "iter",
            "len",
            "list",
            "locals",
            "map",
            "max",
            "memoryview",
            "min",
            "next",
            "object",
            "oct",
            "open",
            "ord",
            "pow",
            "print",
            "property",
            "range",
            "repr",
            "reversed",
            "round",
            "set",
            "setattr",
            "slice",
            "sorted",
            "staticmethod",
            "str",
            "sum",
            "super",
            "tuple",
            "type",
            "vars",
            "zip",
            "__import__"
        ]

# TODO: Add NetworkX viz functionality

if __name__ == "__main__":
    import sys
    sys.path = sys.path[1:] + ['']

    with open("./output.json", 'w') as output:
        # saplings = Saplings("../cases.py")
        saplings = Saplings("../test.py")

        from pprint import pprint
        # pprint(saplings._program_metrics.method_deps)
        # pprint(saplings.get_halstead_metrics())

        output.write(json.dumps(saplings.get_api_forest()))
