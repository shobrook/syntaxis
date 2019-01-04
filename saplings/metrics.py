"""
Most of this code was ripped from the radon project <LINK> and has been slightly
refactored for clarity.
"""

#########
# GLOBALS
#########


import ast
import math
from collections import defaultdict
import utils


######
# MAIN
######


class HalsteadMetrics(ast.NodeVisitor): # Based on Radon source: [LINK]
    def __init__(self, context=None):
        # TODO: Ensure you're counting from all sources of operators/operands
        self.operators_seen = set()
        self.operands_seen = set()
        self.operators = 0
        self.operands = 0

        self.context = context
        self.function_metrics = []

        self._get_name = lambda obj: obj.__class__.__name__

        self.distinct_operands = lambda: len(self.operands_seen) if self.operands_seen else 1
        self.distinct_operators = lambda: len(self.operators_seen) if self.operators_seen else 1

    ## Overloaded Methods ##

    @utils.dispatch
    def visit_BinOp(self, node):
        return (1, 2, (self._get_name(node.op),), (node.left, node.right))

    @utils.dispatch
    def visit_UnaryOp(self, node):
        return (1, 1, (self._get_name(node.op),), (node.operand,))

    @utils.dispatch
    def visit_BoolOp(self, node):
        return (1, len(node.values), (self._get_name(node.op),), node.values)

    @utils.dispatch
    def visit_AugAssign(self, node):
        return (1, 2, (self._get_name(node.op),), (node.target, node.value))

    @utils.dispatch
    def visit_Compare(self, node):
        return (len(node.ops), len(node.comparators) + 1,
                map(self._get_name, node.ops), node.comparators + [node.left])

    def visit_FunctionDef(self, node):
        func_metrics = HalsteadMetrics(node.name)

        for child in node.body:
            metrics = HalsteadMetrics(node.name)
            metrics.visit(child)

            self.operators += metrics.operators
            self.operands += metrics.operands
            self.operators_seen.update(metrics.operators_seen)
            self.operands_seen.update(metrics.operands_seen)

            func_metrics.operators += metrics.operators
            func_metrics.operands += metrics.operands
            func_metrics.operators_seen.update(metrics.operators_seen)
            func_metrics.operands_seen.update(metrics.operands_seen)

        # Save the visited function visitor for later reference.
        self.function_metrics.append(func_metrics)

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    ## Properties ##

    @property
    def vocabulary(self):
        return self.distinct_operators() + self.distinct_operands()

    @property
    def length(self):
        return self.operators + self.operands

    @property
    def calculated_length(self):
        return self.distinct_operators() * math.log(self.distinct_operators(), 2) + self.distinct_operands() * math.log(self.distinct_operands(), 2)

    @property
    def volume(self):
        return self.length * math.log(self.vocabulary, 2)

    @property
    def difficulty(self):
        return (self.distinct_operators() / 2) * (self.operands / self.distinct_operands())

    @property
    def effort(self):
        return self.difficulty * self.volume

    @property
    def time(self):
        return self.effort / 18

    @property
    def bugs(self):
        return self.volume / 3000

class CyclomaticComplexity(ast.NodeVisitor):
    def __init__(self, to_nethod=False, classname=None, off=True):
        self.off = off
        self.complexity = int(off)
        self.functions = []
        self.classes = []
        self.to_method = to_method
        self.classname = classname
        self._max_line = float('-inf')

    def generic_visit(self, node):
        name = node.__class__.__name__

        if name in ("Try", "TryExcept"):
            self.complexity += len(node.handlers) + len(node.orelse)
        elif name == "BoolOp":
            self.complexity += len(node.values) - 1
        elif name in ("With", "If", "IfExp", "AsyncWith", "Assert"):
            self.complexity += 1
        elif name in ("For", "While", "AsyncFor"):
            self.complexity += bool(node.orelse) + 1
        elif name == "comprehension":
            self.complexity += len(node.ifs) + 1

        super().generic_visit(node)

    def visit_FunctionDef(self, node):
        """
        Function complexity is computed taking into account the following
        factors: no. of decorators, complexity of the function body, and no. of
        closures.
        """

        closures = []
        body_complexity = 1

        for child in node.body:
            visitor = ComplexityVisitor(off=False)
            visitor.visit(child)
            closures.extend(visitor.functions)
            body_complexity += visitor.complexity

        self.functions.append({
            ""
        })

class ProgramMetrics(ast.NodeVisitor):
    def __init__(self, tree):
        self.tree = tree
        self._context_stack = ["global"]

        self.all_nodes = []
        self.freq_map = defaultdict(lambda: 0)
        self.method_to_loc_map = {} # Holds mapping from method names to LOC range

        # self.halstead_metrics = HalsteadMetrics()
        # self.halstead_metrics.visit(self.tree)

        # Functional programming techniques
        self.recursive_funcs = []
        self.partial_applications = [] # Closures?
        self.curried_funcs = []
        self.funcs_with_callbacks = []

        self._context_to_string = lambda: '.'.join(self._context_stack)

        self.visit(self.tree)

    ## Overloaded Methods ##

    def generic_visit(self, node):
        self.all_nodes.append(node)
        self.freq_map[type(node).__name__] += 1

        super().generic_visit(node)

    @utils.context_manager
    def visit_ClassDef(self, node):
        pass

    @utils.context_manager
    def visit_FunctionDef(self, node):
        curr_ctx = self._context_to_string()
        self.method_to_loc_map[curr_ctx] = range(node.lineno, utils.get_max_lineno(node.body[-1]) + 1)

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)
