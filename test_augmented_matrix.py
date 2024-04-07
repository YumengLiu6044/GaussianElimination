import unittest
from augmented_matrix import *


class TestAugmentedMatrix(unittest.TestCase):

    def test_augmented_matrix(self):
        self.augmented_matrix = AugmentedMatrix(
            [[1, -2, 1, 0],
             [0, 2, -8, 8],
             [5, 0, -5, 10]]
        )
        expected_solution = [1, 0, -1]
        solution = self.augmented_matrix.solve()
        self.assertEqual(expected_solution, solution)

    def test_augmented_matrix2(self):
        self.augmented_matrix = AugmentedMatrix(
            [[1, -3, 0, 5],
             [-1, 1, 5, 2],
             [0, 1, 1, 0]]
        )
        expected_solution = [2, -1, 1]
        solution = self.augmented_matrix.solve()
        self.assertEqual(expected_solution, solution)

    def test_augmented_matrix_empty(self):
        self.augmented_matrix = AugmentedMatrix(
            [[0, 0, 0],
             [0, 0, 0]]
        )
        with self.assertRaises(InfiniteSolutionError):
            self.augmented_matrix.solve()

    def test_no_solution(self):
        self.augmented_matrix = AugmentedMatrix(
            [[1, 2, 3, 4],
             [1, 2, 3, 5],
             [3, 0, 4, 2]]
        )
        with self.assertRaises(NoSolutionError):
            self.augmented_matrix.solve()

    def test_infinite_solution(self):
        self.augmented_matrix = AugmentedMatrix(
            [[1, 2, 3, 4],
             [2, 4, 6, 8],
             [0, 1, 1, 0]])
        with self.assertRaises(InfiniteSolutionError) as e:
            self.augmented_matrix.solve()


if __name__ == '__main__':
    unittest.main()
