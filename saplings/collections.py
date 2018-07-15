import ast

LITERALS = [
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

VARIABLES = [
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Del,
    ast.Starred
]

UNARY_OPS = [
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.Invert
]

BINARY_OPS = [
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

BOOLEAN_OPS = [
    ast.And,
    ast.Or
]

COMPARISONS = [
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

MISC_EXPRESSIONS = [
    ast.Call,
    ast.IfExp,
    ast.Attribute
]

SUBSCRIPTS = [
    ast.Subscript,
    ast.Index,
    ast.Slice,
    ast.ExtSlice
]

COMPREHENSIONS = [
    ast.ListComp,
    ast.SetComp,
    ast.DictComp
]

GENERATORS = [
    ast.GeneratorExp,
    ast.Yield,
    ast.YielFrom
]

STATEMENTS = [
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.Print,
    ast.Delete,
    ast.Pass
]

EXCEPTION_HANDLING = [
    ast.Raise,
    ast.Assert,
    ast.Try,
    #ast.TryFinally,
    #ast.TryExcept,
    ast.ExceptHandler
]

IMPORTS = [
    ast.Import,
    ast.ImportFrom,
    ast.Alias
]

CONTROL_FLOW = [
    ast.If,
    ast.For,
    ast.While,
    ast.Break,
    ast.Continue,
    ast.With
]

FUNCS_AND_CLASSES = [
    ast.ClassDef,
    ast.FunctionDef,
    ast.Lambda,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.Global,
    ast.Nonlocal
]

ASYNC = [
    ast.AsyncFunctionDef,
    ast.Await,
    ast.AsyncFor,
    ast.AsyncWith
]

BUILT_IN_FUNC_NAMES = [
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
