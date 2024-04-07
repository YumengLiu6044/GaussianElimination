import numpy as np


class InfiniteSolutionError(Exception):
    pass


class NoSolutionError(Exception):
    pass


class AugmentedMatrix(np.ndarray):
    def __new__(cls, *args, **kwargs):
        return np.asarray(*args, **kwargs).view(cls)

    def check_valid_solution(self):
        if self.shape[0] != self.shape[1] - 1:
            raise InfiniteSolutionError('Infinite solutions exists')

        for row in self:
            if all(x == 0 for x in row[:-1]):
                if row[-1] == 0:
                    raise InfiniteSolutionError('Infinite solutions exist')
                else:
                    raise NoSolutionError('No solutions exist')

    def swap_max_value_row(self, row, column):
        max_row_index = row
        for i in range(row, self.shape[0]):
            if abs(self[i][column]) > abs(self[max_row_index][column]):
                max_row_index = i

        self[[row, max_row_index]] = self[[max_row_index, row]]

    def partial_pivot(self, search_for_max=True):
        row, column = 0, 0
        while row < (row_count := self.shape[0]) and column < self.shape[1]:
            if search_for_max:
                self.swap_max_value_row(row, column)
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

    def backward_substitution(self):
        x = [None for _ in range(self.shape[0])]
        # Simplify the echelon form
        for row_index in range(self.shape[0]):
            pivot = 1
            for element in self[row_index]:
                if element != 0:
                    pivot = element
                    break

            self[row_index] = self[row_index] / pivot

        # Backwards substitution
        for row_index in range(self.shape[0] - 1, -1, -1):
            sum_of_left = sum(self[row_index][row_index + 1:self.shape[0]] * x[row_index + 1:self.shape[0]])
            x[row_index] = self[row_index][-1] - sum_of_left

        return x

    def solve(self):
        self.partial_pivot()
        self.check_valid_solution()
        return self.backward_substitution()
