# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#


class EPNode:
    """
    """

    def __init__(self, function, name=None):
        # init connections of node frame
        self._function = function

    @property
    def function(self):
        return self._function

    @function.setter
    def function(self, function):
        self._function = function

    def use_function(self, args):
        return self._function(*args)


"""
Test
"""
if __name__ == '__main__':

    def myfun(a, b):
        return a + b

    node = EPNode(myfun)

    c = node.use_function((2, 3))
    print(c)
