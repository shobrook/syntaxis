#######
# NOTES
#######


# AST to JSON: https://github.com/fpoli/python-astexport
# AST Parser that's version-agnostic: https://github.com/serge-sans-paille/gast
# AST Unparser: https://github.com/simonpercivall/astunparse
# AST Utilities: https://github.com/mutpy/astmonkey


#########
# GLOBALS
#########


import ast
import json
import math
from collections import defaultdict

import utils


#########
# HELPERS
#########


# [] Handle For/While contexts
# [] Handle assignments in Try/Except and If/Else and With contexts
# [] Handle comprehensions and generator expressions
# [] Handle List/Dict/Set/Tuple assignments
# [] Infer input and return (or yield) types of user-defined functions (and classes)
# [] Inside funcs, block searching parent contexts for aliases equivalent to parameter names (unless it's self)
# [] Get rid of the type/type/type/... schema for searching nodes
# [] Debug the frequency analysis

class APIForest(ast.NodeVisitor):
    def __init__(self, tree):
        # IDEA: The entire context/aliasing system can be refactored such that
        # the object holds a mapping from contexts to alive and dead nodes. This
        # could make search more efficient and at the very least make this code
        # easier to read.

        self.tree = tree

        # OPTIMIZE: Turns AST into doubly-linked AST
        for node in ast.walk(self.tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node

        self._context_stack = ["global"] # Keeps track of current context
        self.dependency_trees = [] # Holds root nodes of API usage trees

        self._call_table = {} # For holding user-defined funcs/classes
        self._comp_table = {} # For holding user-defined comprehensions/generator expressions
        self._temp_aliases = {}

        self._context_to_string = lambda: '.'.join(self._context_stack)

        self.visit(self.tree)

    ## Utilities ##

    def _recursively_process_tokens(self, tokens):
        """
        Takes a list of tokens (types: instance, function, args, or subscript)
        and searches for an equivalent nodes in each parse tree. Once the leaf
        of the path of equivalent nodes has been reached, the algorithm creates
        additional nodes and adds them to the parse tree.

        @param tokens: list of tokenized nodes, each as (token(s), type).

        @return: list of references to parse tree nodes corresponding to the
        tokens.
        """

        node_stack = []
        curr_context = self._context_to_string()

        # Flattens nested tokens
        flattened_tokens = []
        for idx, token in enumerate(tokens):
            content, type = token

            adjunctive_content = []
            if type == "args":
                for sub_tokens in content:
                    adjunctive_content.append(self._recursively_process_tokens(sub_tokens))
                content = id(str(adjunctive_content)) if adjunctive_content else id(str(content))
            elif type == "subscript":
                adjunctive_content = self._recursively_process_tokens(content)
                content = id(str(adjunctive_content)) if adjunctive_content else id(str(content))

            flattened_tokens.append((content, type))

        for idx, token in enumerate(flattened_tokens):
            content, type = token
            token_id = utils.stringify_tokenized_nodes(flattened_tokens[:idx + 1])

            type_pattern = adjusted_type = "implicit"
            if type == "args":
                adjusted_id = "call"
            elif type == "subscript":
                adjusted_id = "sub"
            else:
                adjusted_id = content
                adjusted_type = "instance"
                type_pattern = "instance/implicit/module"

            if not idx: # Beginning of iteration
                for root in self.dependency_trees:
                    matching_node = utils.find_matching_node(
                        subtree=root,
                        id=token_id,
                        type_pattern="instance/module/implicit",
                        context=curr_context
                    )

                    if matching_node: # Found base node, pushing to stack
                        matching_node.increment_count()
                        node_stack.append(matching_node)
                        break
            elif node_stack: # Stack exists, continue pushing to it
                matching_node = utils.find_matching_node(
                    subtree=node_stack[-1],
                    id=token_id,
                    type_pattern=type_pattern,
                    context=curr_context
                )

                if matching_node: # Found child node
                    matching_node.increment_count()
                    node_stack.append(matching_node)
                else: # No child node found, creating one
                    child_node = utils.Node(
                        id=adjusted_id,
                        type=adjusted_type,
                        context=curr_context,
                        alias=token_id
                    )

                    node_stack[-1].add_child(child_node)
                    node_stack.append(child_node)
            else: break # Base token doesn't exist, abort processing

        return node_stack

    def _process_assignment(self, target, value):
        """
        @param target:
        @param value:
        """

        curr_context = self._context_to_string()
        tokenized_target = utils.recursively_tokenize_node(target, [])

        targ_matches = self._recursively_process_tokens(tokenized_target)
        val_matches = self._recursively_process_tokens(value)

        alias = utils.stringify_tokenized_nodes(tokenized_target)
        add_alias = lambda node: node.add_alias(curr_context, alias)
        del_alias = lambda node: node.del_alias(curr_context, alias)

        is_temp_assignment = curr_context in self._temp_aliases
        if is_temp_assignment:
            temp_aliases = self._temp_aliases[curr_context]

        if targ_matches and val_matches: # Known node reassigned to known node
            targ_node, val_node = targ_matches[-1], val_matches[-1]

            add_alias(val_node)
            del_alias(targ_node)
            if is_temp_assignment:
                temp_aliases.append(lambda: add_alias(targ_node))
                temp_aliases.append(lambda: del_alias(val_node))
        elif targ_matches and not val_matches: # Known node reassigned to unknown node
            targ_node = targ_matches[-1]

            del_alias(targ_node)
            if is_temp_assignment:
                temp_aliases.append(lambda: add_alias(targ_node))
        elif not targ_matches and val_matches: # Unknown node assigned to known node
            val_node = val_matches[-1]

            add_alias(val_node)
            if is_temp_assignment:
                temp_aliases.append(lambda: del_alias(val_node))

    def _process_module(self, module, context, alias_root=True):
        """
        Takes the identifier for a module, sometimes a period-separated string
        of sub-modules, and searches the list of parse trees for a matching
        module. If no match is found, new module nodes are generated and
        appended to self.dependency_trees.

        @param module: identifier for the module.
        @param context: context in which the module is imported.
        @param alias_root: flag for whether a newly created module node should
        be aliased.

        @return: reference to the terminal Node object in the list of
        sub-modules.
        """

        sub_modules = module.split('.') # For module.submodule1.submodule2...
        root_module = sub_modules[0]
        term_node = None

        for root in self.dependency_trees:
            matching_module = utils.find_matching_node(
                subtree=root,
                id=root_module,
                type_pattern="module"
            )

            if matching_module:
                term_node = matching_module
                break

        if not term_node:
            term_node = utils.Node(
                id=root_module,
                type="module",
                context=context,
                alias=root_module if alias_root else '' # For `from X import Y`
            )
            self.dependency_trees.append(term_node)

        for idx in range(len(sub_modules[1:])):
            sub_module = sub_modules[idx + 1]
            sub_module_alias = '.'.join([root_module] + sub_modules[1:idx + 2])

            matching_sub_module = utils.find_matching_node(
                subtree=term_node,
                id=sub_module,
                type_pattern="instance",
                context=None # QUESTION: Should this be context?
            )

            if matching_sub_module:
                term_node = matching_sub_module
            else:
                new_sub_module = utils.Node(
                    id=sub_module,
                    type="instance",
                    context=context,
                    alias=sub_module_alias if alias_root else '' # For `from X.Y import Z`
                )
                term_node.add_child(new_sub_module)
                term_node = new_sub_module

        return term_node

    ## Overloaded Methods ##

    # @utils.context_manager
    # def visit_Global(self, node):
    #     # IDEA: Save pre-state of context_stack, set to ["global"],
    #     # then set back to pre-state
    #     return

    # @utils.context_manager
    # def visit_Nonlocal(self, node):
    #     return

    @utils.context_manager
    def visit_ClassDef(self, node):
        pass

    @utils.context_manager
    def visit_FunctionDef(self, node):
        pass

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    # @utils.context_manager(False)
    # def visit_Lambda(self, node):
    #     return

    @utils.context_manager
    def visit_If(self, node):
        pass

    @utils.context_manager
    def visit_Try(self, node):
        pass

    @utils.context_manager
    def visit_ExceptHandler(self, node):
        pass

    @utils.visitor(True)
    def visit_Import(self, node):
        curr_context = self._context_to_string()
        for module in node.names:
            alias = module.asname if module.asname else module.name
            module_leaf_node = self._process_module(
                module=module.name,
                context=curr_context,
                alias_root=not bool(module.asname)
            )

            module_leaf_node.add_alias(curr_context, alias)

    @utils.visitor(True)
    def visit_ImportFrom(self, node):
        curr_context = self._context_to_string()
        module_node = self._process_module(
            module=node.module,
            context=curr_context,
            alias_root=False
        )

        for alias in node.names:
            child_exists = False
            alias_id = alias.asname if alias.asname else alias.name

            for child in module_node.children:
                if alias.name == child.id:
                    child_exists = True
                    if not alias_id in child.aliases[curr_context]:
                        child.add_alias(curr_context, alias_id)

                    break

            if not child_exists:
                module_node.add_child(utils.Node(
                    id=alias.name,
                    type="instance",
                    context=curr_context,
                    alias=alias_id
                ))

    @utils.visitor(visit_children=False) # QUESTION: Should be False?
    def visit_Assign(self, node):
        curr_context = self._context_to_string()

        if isinstance(node.value, ast.Tuple):
            node_tokenizer = lambda elt: utils.recursively_tokenize_node(elt, [])
            values = tuple(map(node_tokenizer, node.value.elts))
        else:
            values = utils.recursively_tokenize_node(node.value, [])

        targets = node.targets if hasattr(node, "targets") else (node.target)
        for target in targets:
            if isinstance(target, ast.Tuple):
                for idx, elt in enumerate(target.elts):
                    if isinstance(values, tuple):
                        self._process_assignment(elt, values[idx])
                    else:
                        self._process_assignment(elt, values)
            elif isinstance(values, tuple):
                for value in values:
                    self._process_assignment(target, value)
            else:
                self._process_assignment(target, values)

    @utils.visitor(visit_children=False)
    def visit_AnnAssign(self, node):
        self.visit_Assign(node)

    @utils.visitor(True)
    def visit_Call(self, node):
        tokens = utils.recursively_tokenize_node(node, [])
        nodes = self._recursively_process_tokens(tokens)

    @utils.visitor(True)
    def visit_Attribute(self, node):
        # You could try searching up the node.parent.parent... path to find
        # out if attribute is inside a call node. If it is, let the call visiter
        # take care of it. If it isn't, then keep doing what you're doing.

        # Also possible that the best way to deal with this is by just having
        # one ast.Load visitor. Look into this more, i.e. what gets covered by
        # ast.Load.

        tokens = utils.recursively_tokenize_node(node, [])
        nodes = self._recursively_process_tokens(tokens)

    @utils.visitor(True)
    def visit_Subscript(self, node):
        pass

    @utils.visitor(True)
    def visit_comprehension(self, node):
        pass

    @utils.visitor(True)
    def visit_Dict(self, node):
        pass

    @utils.visitor(True)
    def visit_List(self, node):
        pass

    @utils.visitor(True)
    def visit_Tuple(self, node):
        pass

    @utils.visitor(True)
    def visit_Set(self, node):
        pass

    @utils.visitor(True)
    def visit_Compare(self, node):
        pass

    @utils.visitor(True)
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Del):
            pass  # TODO: Delete alias from tree
        elif isinstance(node.ctx, ast.Load):
            pass  # TODO: Increment count of node (beware of double counting)

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
    def __init__(self, tree):
        self.tree = tree

class ProgramMetrics(ast.NodeVisitor):
    def __init__(self, tree):
        self.tree = tree

        self.retrieved_nodes = []
        self.freq_map = defaultdict(lambda: 0)
        self.method_to_loc_map = {} # Holds mapping from method names to LOC range

        self.halstead_metrics = HalsteadMetrics()
        self.halstead_metrics.visit(self.tree)

        self.visit(self.tree)

    ## Overloaded Methods ##

    def generic_visit(self, node):
        self.retrieved_nodes.append(node)
        self.freq_map[type(node).__name__] += 1

        super().generic_visit(node)

    @utils.visitor(visit_children=True)
    def visit_FunctionDef(self, node):
        self.method_to_loc_map[node.name] = range(node.lineno, utils.get_max_lineno(node.body[-1]) + 1)


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

if __name__ == "__main__":
    import sys
    sys.path = sys.path[1:] + ['']

    with open("./output.json", 'w') as output:
        # saplings = Saplings("./test.py")
        saplings = Saplings("./cases.py")

        from pprint import pprint
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
