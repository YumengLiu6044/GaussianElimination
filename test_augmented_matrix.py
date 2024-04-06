import unittest
from augmented_matrix import AugmentedMatrix


class TestAugmentedMatrix(unittest.TestCase):

    def test_augmented_matrix(self):
        self.augmented_matrix = AugmentedMatrix(
            [[1, -2, 1, 0],
             [0, 2, -8, 8],
             [5, 0, -5, 10]]
        )
        expected_solution = [1, 0, -1]
        solution = self.augmented_matrix.solve()
        print('Actual:\n', solution)
        print('Expected:\n', expected_solution)
        self.assertTrue(expected_solution, solution)


if __name__ == '__main__':
    unittest.main()
