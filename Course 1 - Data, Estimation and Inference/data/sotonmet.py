import os
import time
import calendar
import numpy as np
import plotting
import data.load

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SOTONMET_PATH = os.path.join(CURRENT_DIR, "sotonmet.txt")
DAYS_PER_SECOND = 1 / (60 * 60 * 24)

class Sotonmet:
    def __init__(self, data_path=SOTONMET_PATH):
        self.n_train = 917
        self.n_truth = 1258

        self._data_dict, num_cols, num_rows = data.load.load_dict(data_path)
        assert num_cols == 19
        assert num_rows == 1258

        t_str_list = self._data_dict["Reading Date and Time (ISO)"]
        y_str_list = self._data_dict["Tide height (m)"]

        t0 = get_timestamp(t_str_list[0])
        t_list = [
            (get_timestamp(t_str) - t0) * DAYS_PER_SECOND
            for t_str in t_str_list
        ]

        has_data_list = [len(y) > 0 for y in y_str_list]

        y_train = [
            float(y_str)
            for y_str, has_data in zip(y_str_list, has_data_list)
            if has_data
        ]
        t_train = [
            t
            for t, has_data in zip(t_list, has_data_list)
            if has_data
        ]

        self.t_train = np.array(t_train)
        self.y_train = np.array(y_train)
        self.t_truth = np.array(t_list)
        self.y_truth = self.get_column_data("True tide height (m)")
        assert self.t_train.size == 917
        assert self.y_train.size == 917
        assert self.t_truth.size == 1258
        assert self.y_truth.size == 1258

    def get_column_data(self, column_name):
        y_str_list = self._data_dict[column_name]
        y_data = [float(y_str) for y_str in y_str_list if len(y_str) > 0]

        return np.array(y_data)

    def get_train_test_plot_lines(self):
        train_data_line = plotting.Line(
            self.t_train,
            self.y_train,
            c="k",
            ls="",
            marker="o",
            alpha=0.5,
            zorder=20,
        )
        test_data_line = plotting.Line(
            self.t_truth,
            self.y_truth,
            c="k",
            ls="",
            marker="x",
            alpha=0.5,
            zorder=20,
        )
        return train_data_line, test_data_line

def get_timestamp(t_str, t_format="%Y-%m-%dT%H:%M:%S"):
    time_struct = time.strptime(t_str, t_format)
    timestamp = calendar.timegm(time_struct)
    return timestamp
