import cvxpy as cp

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
