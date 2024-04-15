import numpy as np

NON_SIMPLIFIED = -1
UPPER_TRIANGULAR = 0
REDUCED_ECHELON = 1


class NoSolutionError(Exception):
    pass


class AugmentedMatrix(np.ndarray):
    def __new__(cls, *args, **kwargs):

        return np.asarray(*args, **kwargs).view(cls)

    def _check_valid_solution(self):
        """
        Checks if the matrix is consistent
        :return:
        """
        for row in self:
            if all(x == 0 for x in row[:-1]):
                if row[-1] != 0:
                    raise NoSolutionError('No solution exists!')

    def partial_pivot(self):
        """
        Transforms the matrix into upper triangular form by partial pivoting
        """
        row, column = 0, 0
        while row < (row_count := self.shape[0]) and column < self.shape[1]:
            # Use the highest absolute value as pivot
            max_row_index = row
            for i in range(row, self.shape[0]):
                if abs(self[i][column]) > abs(self[max_row_index][column]):
                    max_row_index = i

            self[[row, max_row_index]] = self[[max_row_index, row]]

            pivot = self[row][column]
            if pivot != 0:
                for i in range(row + 1, row_count):
                    leading_value = self[i][column]
                    if leading_value == 0:
                        continue
                    factor = -1 * pivot / leading_value
                    self[i] = factor * self[i] + self[row]

            column += 1
            row = column

    def _simplify_echelon(self):
        """
        Simplifies the upper triangular form so that the pivots are 1
        """
        # Simplify the echelon form
        for row_index in range(self.shape[0]):
            pivot = 1
            for element in self[row_index]:
                if element != 0:
                    pivot = element
                    break
            self[row_index] = self[row_index] / pivot

    def _get_state(self):
        row, column = 0, 0
        while row < (self.shape[0] - 1) and column < self.shape[1]:
            if all(x == 0 for x in self[:, column][row+1:]):
                row += 1
                column += 1
            else:
                return NON_SIMPLIFIED

        return UPPER_TRIANGULAR

    def reduce_echelon(self):
        """
        Transforms the matrix from upper triangular form to reduced echelon form
        :raise: NoSolutionException: if the matrix has no solution
        """
        if self._get_state() != UPPER_TRIANGULAR:
            self.partial_pivot()

        self._check_valid_solution()
        self._simplify_echelon()

        for row_index in range(self.shape[0] - 1, -1, -1):
            pivot_coord = None
            for element_index in range(self.shape[1] - 1):
                if self[row_index][element_index] != 0:
                    pivot_coord = [row_index, element_index]
                    break

            if pivot_coord is None:
                continue
            else:
                for i in range(row_index):
                    leading = self[i][pivot_coord[1]]
                    if leading == 0:
                        continue

                    pivot = self[pivot_coord[0]][pivot_coord[1]]
                    factor = -1 * leading / pivot
                    self[i] = factor * self[pivot_coord[0]] + self[i]

    def solve(self):
        """
        Solves the augmented matrix
        :raise: NoSolutionException: if the matrix has no solution
        """
        self.partial_pivot()
        self.reduce_echelon()


__all__ = ['AugmentedMatrix']
