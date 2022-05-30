class ObjectNode(object):
    """
    Object hierarchy node. Represents an object that's descendant of an imported
    module –– descendant meaning there exists an "attribute chain" between the
    module and the object.
    """

    def __init__(self, name, is_callable=False, order=0, frequency=1, children=[]):
        """
        Parameters
        ----------
        name : str
            name of the object
        is_callable : bool
            indicates whether the object is callable (i.e. has __call__ defined)
        order : int
            indicates the type of connection to the parent node (e.g. `0` is an
            attribute of the parent, `1` is an attribute of the output of the
            parent when called, etc.); `-1` if node is root
        frequency : int
            number of times the object is used in the program
        children : list
            list of child nodes
        """

        self.name = name
        self.is_callable = is_callable
        self.order = order
        self.frequency = frequency
        self.children = []

        for child in children:
            self.add_child(child)

    def __repr__(self):
        return self.name

    def __str__(self):
        return f"{self.name} ({'C' if self.is_callable else 'NC'}, {self.order})"

    def __eq__(self, node):
        if isinstance(node, type(self)):
            return self.name == node.name

        return False

    def __ne__(self, node):
        return not self.__eq__(node)

    ## Instance Methods ##

    def increment_count(self):
        self.frequency += 1

    def add_child(self, node):
        for child in self.children:
            if child == node: # Child already exists
                child.increment_count()
                return child

        self.children.append(node)
        return node

    def breadth_first(self):
        node_queue = [self]
        while node_queue:
            node = node_queue.pop(0)
            yield node
            for child in node.children:
                node_queue.append(child)


class Function(object):
    """
    Represents a user-defined function.
    """

    def __init__(self, def_node, init_namespace, is_closure=False, called=False, method_type=None, containing_class=None):
        """
        Parameters
        ----------
        def_node : ast.FunctionDef
        init_namespace : dict
            namespace in which the function was defined
        is_closure : bool
            indicates whether the function is a closure
        called : bool
            indicates whether the function has been called
        method_type : {string, None}
            if the function was defined inside a class, this indicates the type
            of method it is (e.g. instance, class, or static); None for
            non-methods
        containing_class : {Class, None}
            if the function was defined inside a class, this is the object
            representing that class entity
        """

        self.def_node = def_node
        self.init_namespace = init_namespace
        self.is_closure = is_closure
        self.called = called
        self.method_type = method_type
        self.containing_class = containing_class


class Class(object):
    """
    Represents a user-defined class.
    """

    def __init__(self, def_node, init_namespace, init_instance_namespace={}):
        """
        Parameters
        ----------
        def_node : ast.ClassDef
        init_namespace : dict
            namespace in which the class is defined
        init_instance_namespace : dict
            namespace containing the methods and variables defined inside the
            class; everything in this namespace is an attribute of `self`
        """

        self.def_node = def_node
        self.init_instance_namespace = init_instance_namespace


class ClassInstance(object):
    """
    Represents an instance of a user-defined class.
    """

    def __init__(self, class_entity, namespace):
        """
        Parameters
        ----------
        class_entity : Class
            class entity for which this is an instance of
        namespace : dict
            namespace/state of the instance (everything here is an attribute of
            `self`)
        """

        self.class_entity = class_entity
        self.namespace = namespace
