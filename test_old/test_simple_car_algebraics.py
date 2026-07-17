import unittest
from examples.simple_car_algebraics.simple_car_algebraic import solve_ocp


class TestSimpleCarAlgebraics(unittest.TestCase):

    def test_simple_solving(self):
        solve_ocp()


if __name__ == "__main__":
    unittest.main()
