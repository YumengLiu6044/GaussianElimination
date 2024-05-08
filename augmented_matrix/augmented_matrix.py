import numpy
import numpy as np
from fractions import Fraction


class NoSolutionError(Exception):
    pass


class AugmentedMatrix:
    """
    A class that represents an augmented matrix
    """
    def __init__(self, matrix, constraint=None, /, **kwargs):

        self._matrix = np.asarray([[Fraction(j).limit_denominator() for j in i] for i in matrix], **kwargs)

        if constraint is not None:
            constraint = np.asarray([Fraction(i).limit_denominator() for i in constraint], **kwargs)
            self._matrix = np.append(self._matrix, [[i] for i in constraint], axis=1)

        self._swap_factor = 1
        self._mult_factor = 1

    @property
    def matrix(self) -> np.ndarray:
        """
        :return: the augmented matrix
        """
        return self._matrix

    def set_matrix(self, matrix):
        self._matrix = matrix

    def check_valid_solution(self):
        """
        Checks if the matrix is consistent
        :raises NoSolutionError: If the matrix is not consistent
        """
        matrix = self._matrix
        for row in matrix:
            if all(x == 0 for x in row[:-1]):
                if row[-1] != 0:
                    raise NoSolutionError('No solution exists!')

    def check_upper_triangular(self) -> bool:
        """
        Checks if the matrix is in upper triangular form
        :returns True: if the matrix is in upper triangular form
        :returns False: if the matrix is not in upper triangular form
        """
        matrix = self._matrix
        row, column = 0, 0
        while row < (matrix.shape[0] - 1) and column < matrix.shape[1]:
            if all(x == 0 for x in matrix[:, column][row + 1:]):
                row += 1
                column += 1
            else:
                return False

        return True

    def partial_pivot(self) -> np.ndarray:
        """
        Transforms the matrix into upper triangular form by partial pivoting

        :return: the upper triangular form of the matrix
        """
        matrix = self._matrix
        row, column = 0, 0
        while row < (row_count := matrix.shape[0]) and column < matrix.shape[1]:
            # Use the highest absolute value as pivot
            max_row_index = row
            for i in range(row, matrix.shape[0]):
                if abs(matrix[i][column]) > abs(matrix[max_row_index][column]):
                    max_row_index = i

            if row != max_row_index:
                matrix[[row, max_row_index]] = matrix[[max_row_index, row]]
                self._swap_factor *= -1

            pivot = matrix[row][column]
            if pivot != 0:
                for i in range(row + 1, row_count):
                    leading_value = matrix[i][column]
                    if leading_value == 0:
                        continue
                    factor = -1 * pivot / leading_value
                    matrix[i] = factor * matrix[i] + matrix[row]
                    self._mult_factor *= factor

            column += 1
            row = column

        self._matrix = matrix
        return matrix

    def simplify_echelon(self) -> np.ndarray:
        """
        Simplifies the upper triangular form so that the pivots are 1
        """
        matrix = self._matrix
        # Simplify the echelon form
        for row_index in range(matrix.shape[0]):
            pivot = 1
            for element in matrix[row_index]:
                if element != 0:
                    pivot = element
                    break
            matrix[row_index] = matrix[row_index] / pivot

        self._matrix = matrix
        return matrix

    def reduce_echelon(self) -> np.ndarray:
        """
        Transforms the matrix from upper triangular form to reduced echelon form
        :raise: NoSolutionException: if the matrix has no solution
        """
        if not self.check_upper_triangular():
            self.partial_pivot()

        self.simplify_echelon()

        matrix = self._matrix
        for row_index in range(matrix.shape[0] - 1, -1, -1):
            pivot_coord = None
            for element_index in range(matrix.shape[1] - 1):
                if matrix[row_index][element_index] != 0:
                    pivot_coord = [row_index, element_index]
                    break

            if pivot_coord is None:
                continue
            else:
                for i in range(row_index):
                    leading = matrix[i][pivot_coord[1]]
                    if leading == 0:
                        continue

                    pivot = matrix[pivot_coord[0]][pivot_coord[1]]
                    factor = -1 * leading / pivot
                    matrix[i] = factor * matrix[pivot_coord[0]] + matrix[i]

        self._matrix = matrix
        return matrix

    def solve(self, return_type=float) -> np.ndarray:
        """
        Solves the augmented matrix
        :returns self._matrix: the row-reduced matrix
        :raises NoSolutionException: if the matrix has no solution
        """
        self.partial_pivot()
        self.reduce_echelon()
        self.check_valid_solution()
        return self._matrix.astype(return_type)

    def is_square(self) -> bool:
        """
        Checks if the matrix is square (the dimensions are equal and the matrix is 2D)
        :returns True: if the matrix is square
        :returns False: if the matrix is not square
        """
        shape = self._matrix.shape
        return len(shape) == 2 and shape[0] == shape[1]

    def determinant(self):
        """
        Calculates the determinant of the matrix
        :return: a value indicating the determinant of this matrix
        :raises NoSolutionException: if the matrix is not square
        """

        if self.is_square():
            self.partial_pivot()
            return numpy.prod(self._matrix.diagonal(), axis=0) * self._swap_factor / self._mult_factor
        else:
            raise NoSolutionError("The matrix is not square")


__all__ = ['AugmentedMatrix', 'NoSolutionError']
