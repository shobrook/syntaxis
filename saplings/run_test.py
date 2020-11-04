from transducer import Saplings
import ast
from pprint import pprint

my_program = open("test_input.py", "r").read()
program_ast = ast.parse(my_program)
my_saplings = Saplings(program_ast)

pprint(my_saplings.trees())
