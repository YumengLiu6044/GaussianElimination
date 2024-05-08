import unittest
from augmented_matrix import *
from fractions import Fraction


class TestAugmentedMatrix(unittest.TestCase):
    def _to_fraction(self, original):
        return [[Fraction(j).limit_denominator() for j in i] for i in original]

    def check_matrix_equal(self, unsolved, expected, constraint=None):
        if constraint is not None:
            constraint = [Fraction(j).limit_denominator() for j in constraint]
        m = AugmentedMatrix(self._to_fraction(unsolved), constraint)
        m.solve()
        self.assertEqual(m.matrix.tolist(), self._to_fraction(expected))

    def check_determinant(self, matrix, expected_determinant):
        m = AugmentedMatrix(matrix)
        self.assertEqual(m.determinant(), expected_determinant)

    def test_augmented_matrix(self):
        original_matrix = [
            [3, 5, -1, 10],
            [1, 4, 1, 7],
            [9, 0, 2, 1]
        ]

        expected = [
            [1, 0, 0, 0.2],
            [0, 1, 0, 1.8],
            [0, 0, 1, -0.4]
        ]
        self.check_matrix_equal(original_matrix, expected)

    def test_call_reduce_echelon_when_non_echelon(self):
        m = AugmentedMatrix(self._to_fraction([
            [1, 2, 3, 4],
            [0, 1, 2, 2],
            [0, 1, 1, 3]
        ]))
        m.reduce_echelon()
        expected = [
            [1, 0, 0, -1],
            [0, 1, 0, 4],
            [0, 0, 1, -1]
        ]
        self.assertEqual(m.matrix.tolist(), expected)

    def test_augmented_matrix_no_solution(self):
        original = [
            [1, 2, 3],
            [1, 2, 4]
        ]
        with self.assertRaises(NoSolutionError):
            self.check_matrix_equal(original, [])

    def test_augmented_matrix_with_free_variable(self):
        original_matrix = [
            [1, 2, 3],
            [2, 4, 6]
        ]
        expected = [
            [1, 2, 3],
            [0, 0, 0]
        ]
        self.check_matrix_equal(original_matrix, expected)

    def test_augmented_matrix_with_constraint(self):
        original_matrix = [
            [3, 5, -1],
            [1, 4, 1],
            [9, 0, 2]
        ]
        constraint = [10, 7, 1]
        expected = [
            [1, 0, 0, 0.2],
            [0, 1, 0, 1.8],
            [0, 0, 1, -0.4]
        ]
        self.check_matrix_equal(original_matrix, expected, constraint=constraint)

    def test_determinant(self):
        matrix = [
            [2, -8, 6, 8],
            [3, -9, 5, 10],
            [-3, 0, 1, -2],
            [1, -4, 0, 6]
        ]
        self.check_determinant(matrix, -36)

    def test_non_square_matrix_determinant_raise_error(self):
        matrix = [
            [1, 2, 3],
            [4, 5, 6]
        ]
        with self.assertRaises(NoSolutionError):
            self.check_determinant(matrix, 1)


if __name__ == '__main__':
    unittest.main()
