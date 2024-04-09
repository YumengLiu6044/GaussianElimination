import numpy as np


class NoSolutionError(Exception):
    pass


class AugmentedMatrix(np.ndarray):
    def __new__(cls, *args, **kwargs):
        return np.asarray(*args, dtype=np.float64, **kwargs).view(cls)

    def _check_valid_solution(self):
        for row in self:
            if all(x == 0 for x in row[:-1]):
                if row[-1] != 0:
                    raise NoSolutionError('No solution exists!')

    def _swap_max_value_row(self, row, column):
        max_row_index = row
        for i in range(row, self.shape[0]):
            if abs(self[i][column]) > abs(self[max_row_index][column]):
                max_row_index = i

        self[[row, max_row_index]] = self[[max_row_index, row]]

    def _partial_pivot(self):
        row, column = 0, 0
        while row < (row_count := self.shape[0]) and column < self.shape[1]:
            self._swap_max_value_row(row, column)
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
        # Simplify the echelon form
        for row_index in range(self.shape[0]):
            pivot = 1
            for element in self[row_index]:
                if element != 0:
                    pivot = element
                    break
            self[row_index] = self[row_index] / pivot

    def _reduce_echelon(self):
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
        self._partial_pivot()
        self._check_valid_solution()
        self._simplify_echelon()
        self._reduce_echelon()


if __name__ == '__main__':
    m = AugmentedMatrix(
        [[0, 3, -6, 6, 4, 5],
         [3, -7, 8, -5, 8, 9],
         [3, -9, 12, -9, 6, 15]])
    m.solve()
    print(m, '\n')

