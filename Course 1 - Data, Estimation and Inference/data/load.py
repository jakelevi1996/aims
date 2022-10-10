import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, "data", "sotonmet.txt")

with open(DATA_PATH) as f:
    data_line_list = f.read().split("\n")

heading_list = data_line_list[0].split(",")
data_table = [line.split(",") for line in data_line_list[1:] if len(line) > 0]

num_columns = len(heading_list)
num_rows = len(data_table)

for row in data_table:
    assert len(row) == num_columns

data_dict = {
    heading: [row[i] for row in data_table]
    for i, heading in enumerate(heading_list)
}

for data_list in data_dict.values():
    assert len(data_list) == num_rows

assert num_columns == 19
assert num_rows == 1258
