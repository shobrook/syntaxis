#---

def func1():
    return np.array([])

func1() # Invalid

import numpy as np

func1() # Valid

#---



#---

# When you create a new subcontext, everything from parent should propogate into
# it

# The contents of a function will be parsed differently depending on when the
# function is called.

# Function redefinition?

# One strategy is to, when a function node is hit, skip traversing it and add it
# to a list A. Then, once the function call is hit, check A to see if the
# function name is in the list; if it is, then parse it (start by propogating
# all aliases in parent context into it.). Mark this function as "called". At
# the end of the traversal, for all the uncalled functions, parse them with the
# aliases defined

# Save the state (alias map) in which the function was defined along with the
# function node.

# Reassignments in functions don't have effects on the parent context
