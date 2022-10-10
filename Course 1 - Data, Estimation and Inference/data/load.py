import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SOTONMET_PATH = os.path.join(CURRENT_DIR, "sotonmet.txt")

def load_dict(data_path=SOTONMET_PATH):
    with open(data_path) as f:
        data_line_list = f.read().split("\n")

    heading_list = data_line_list[0].split(",")
    data_table = [
        line.split(",")
        for line in data_line_list[1:]
        if len(line) > 0
    ]

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

    return data_dict, num_columns, num_rows
