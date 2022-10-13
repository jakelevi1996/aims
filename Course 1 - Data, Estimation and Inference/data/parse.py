import time
import calendar
import numpy as np

DAYS_PER_SECOND = 1 / (60 * 60 * 24)

def parse_dict(data_dict):
    t_str_list = data_dict["Reading Date and Time (ISO)"]
    y_str_list = data_dict["Tide height (m)"]

    t0 = get_timestamp(t_str_list[0])
    t_list = [
        (get_timestamp(t_str) - t0) * DAYS_PER_SECOND
        for t_str in t_str_list
    ]

    has_data_list = [len(y) > 0 for y in y_str_list]

    y_data = [
        float(y_str)
        for y_str, has_data in zip(y_str_list, has_data_list)
        if has_data
    ]
    t_data = [t for t, has_data in zip(t_list, has_data_list) if has_data]

    return np.array(t_list), np.array(t_data), np.array(y_data)

def get_timestamp(t_str, t_format="%Y-%m-%dT%H:%M:%S"):
    time_struct = time.strptime(t_str, t_format)
    timestamp = calendar.timegm(time_struct)
    return timestamp
