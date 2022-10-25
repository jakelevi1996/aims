import cvxpy as cp
import numpy as np

def solve(known_values):
    """
    Solve the sudoku specified by known_values. known_values should be a list
    containing 9 elements, each corresponding to a row in the soduku, which is
    itself a list containing 9 elements, each correpsonding to a column and a
    value in that row. The elements in the inner list should be a number from 1
    to 9 if that number is known, or 0 if that value is unknown.
    """
    # Define x (the decision variable), the list of coordinates, and the
    # dictionary mapping coordinates to elements in x. x_from_coord[val, row,
    # col] == 1 implies that the element at position (row, col) has value val
    x = cp.Variable(shape=(9*9*9), boolean=True)
    numbers_1_to_9 = list(range(1, 10))

    coord_list = [
        (val, row, col)
        for val in numbers_1_to_9
        for row in numbers_1_to_9
        for col in numbers_1_to_9
    ]

    x_from_coord = {coord: x[i] for i, coord in enumerate(coord_list)}

    # Initialise list of constraints
    constraints = []

    # Each element can only have one value, EG in the 3rd row and the 5th
    # column, x_from_coord[1, 3, 5] + x_from_coord[2, 3, 5] + ... +
    # x_from_coord[9, 3, 5] == 1
    for row in numbers_1_to_9:
        for col in numbers_1_to_9:
            constraints.append(
                sum(
                    x_from_coord[val, row, col]
                    for val in numbers_1_to_9
                ) == 1
            )

    # Each value can only be in each row once, EG for the value 2 in the 8th
    # row, x_from_coord[2, 8, 1] + x_from_coord[2, 8, 2] + ... +
    # x_from_coord[2, 8, 9] == 1
    for val in numbers_1_to_9:
        for row in numbers_1_to_9:
            constraints.append(
                sum(
                    x_from_coord[val, row, col]
                    for col in numbers_1_to_9
                ) == 1
            )

    # Each value can only be in each column once, EG for the value 7 in the 4th
    # column, x_from_coord[7, 1, 4] + x_from_coord[7, 2, 4] + ... +
    # x_from_coord[7, 9, 4] == 1
    for val in numbers_1_to_9:
        for col in numbers_1_to_9:
            constraints.append(
                sum(
                    x_from_coord[val, row, col]
                    for row in numbers_1_to_9
                ) == 1
            )

    # Each value can only be in each 3x3 cell once, EG for the value 8 in the
    # cell which is 3 down and 2 accross, x_from_coord[8, 7, 4] +
    # x_from_coord[8, 7, 5]
    #+ ... + x_from_coord[8, 9, 6]] == 1
    for val in numbers_1_to_9:
        for cell_row in [0, 1, 2]:
            for cell_col in [0, 1, 2]:
                constraints.append(
                    sum(
                        x_from_coord[
                            val,
                            3 * cell_row + row_in_cell,
                            3 * cell_col + col_in_cell,
                        ]
                        for row_in_cell in [1, 2, 3]
                        for col_in_cell in [1, 2, 3]
                    ) == 1
                )

    # Specify known values as constraints
    for row, row_list in zip(numbers_1_to_9, known_values):
        for col, val in zip(numbers_1_to_9, row_list):
            if val > 0:
                constraints.append(x_from_coord[val, row, col] == 1)

    objective = cp.Maximize(0)
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # Convert solution into a numpy array and return
    s = np.zeros(shape=[9, 9])
    for i in range(len(x.value)):
        if x.value[i] == 1:
            val, row, col = coord_list[i]
            if s[row - 1, col - 1] != 0:
                raise RuntimeError(
                    "Found 2 values %i and %i in row %i and column %i"
                    % (val, s[row - 1, col - 1], row, col)
                )
            s[row - 1, col - 1] = val

    return s
