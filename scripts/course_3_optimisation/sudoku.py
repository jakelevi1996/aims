import cvxpy as cp
import numpy as np

# Define x (the decision variable), the list of coordinates, and the dictionary
# mapping coordinates to elements in x. x[coord_to_ind_dict[val, row, col]] ==
# 1 implies that the element at position (row, col) has value val. In
# subsequent comments, the shorthand x[[val, row, col]] is used for
# x[coord_to_ind_dict[val, row, col]]
x = cp.Variable(shape=(9*9*9), boolean=True)
numbers_1_to_9 = list(range(1, 10))

coord_list = [
    (val, row, col)
    for val in numbers_1_to_9
    for row in numbers_1_to_9
    for col in numbers_1_to_9
]

coord_to_ind_dict = {coord: i for i, coord in enumerate(coord_list)}
# x_from_coord = {coord: x[i] for i, coord in enumerate(coord_list)}

# Initialise list of constraints
constraints = []

# Each element can only have one value, EG in the 3rd row and the 5th column,
# x[[1, 3, 5]] + x[[2, 3, 5] + ... + x[[9, 3, 5]]] == 1
for row in numbers_1_to_9:
    for col in numbers_1_to_9:
        constraints.append(
            sum(
                x[coord_to_ind_dict[val, row, col]]
                for val in numbers_1_to_9
            ) == 1
        )

# Each value can only be in each row once, EG for the value 2 in the 8th row,
# x[[2, 8, 1]] + x[[2, 8, 2]] + ... + x[[2, 8, 9]] == 1
for val in numbers_1_to_9:
    for row in numbers_1_to_9:
        constraints.append(
            sum(
                x[coord_to_ind_dict[val, row, col]]
                for col in numbers_1_to_9
            ) == 1
        )

# Each value can only be in each column once, EG for the value 7 in the 4th
# column, x[[7, 1, 4]] + x[[7, 2, 4]] + ... + x[[7, 9, 4]] == 1
for val in numbers_1_to_9:
    for col in numbers_1_to_9:
        constraints.append(
            sum(
                x[coord_to_ind_dict[val, row, col]]
                for row in numbers_1_to_9
            ) == 1
        )

# Each value can only be in each 3x3 cell once, EG for the value 8 in the cell
# which is 3 down and 2 accross, x[[8, 7, 4]] + x[[8, 7, 5]] + ... + x[[8, 9,
# 6]] == 1
for val in numbers_1_to_9:
    for cell_row in [0, 1, 2]:
        for cell_col in [0, 1, 2]:
            constraints.append(
                sum(
                    x[
                        coord_to_ind_dict[
                            val,
                            3 * cell_row + row_in_cell,
                            3 * cell_col + col_in_cell,
                        ]
                    ]
                    for row_in_cell in [1, 2, 3]
                    for col_in_cell in [1, 2, 3]
                ) == 1
            )

# Specify sudoku as a numpy array
known_row_col_val_list = [
    [1, 2, 1], [1, 8, 8],
    [2, 1, 8], [2, 3, 4], [2, 7, 7], [2, 9, 6],
    [3, 2, 3], [3, 5, 9], [3, 8, 1],
    [4, 4, 7], [4, 6, 8],
    [5, 3, 5], [5, 5, 1], [5, 7, 8],
    [6, 4, 3], [6, 6, 4],
    [7, 2, 4], [7, 5, 2], [7, 8, 6],
    [8, 1, 5], [8, 3, 2], [8, 7, 3], [8, 9, 9],
    [9, 2, 9], [9, 8, 5],
]
for row, col, val in known_row_col_val_list:
    constraints.append(x[coord_to_ind_dict[val, row, col]] == 1)

objective = cp.Maximize(0)
prob = cp.Problem(objective, constraints)
print(prob.solve())
print(x.value)

# Convert solution into a numpy array and print
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

print(s)
