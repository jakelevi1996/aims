import time
import calendar
import __init__
import util
import plotting
import load_data

data_dict = load_data.data_dict

t_str_list = data_dict["Reading Date and Time (ISO)"]
y_str_list = data_dict["Tide height (m)"]

t_format = "%Y-%m-%dT%H:%M:%S"
days_per_second = 1 / (60 * 60 * 24)

get_timestamp = lambda t_str: calendar.timegm(time.strptime(t_str, t_format))
t0 = get_timestamp(t_str_list[0])
t_list = [
    (get_timestamp(t_str) - t0) * days_per_second
    for t_str in t_str_list
]

has_data_list = [len(y) > 0 for y in y_str_list]

y_data = [
    float(y_str)
    for y_str, has_data in zip(y_str_list, has_data_list)
    if has_data
]
t_data = [t for t, has_data in zip(t_list, has_data_list) if has_data]
t_pred = [t for t, has_data in zip(t_list, has_data_list) if not has_data]

data_line = plotting.Line(t_data, y_data, c="b", ls="-", marker="o", alpha=0.5)
plotting.plot([data_line], "Tide height (m) vs time (days)")
pred_lines = [plotting.HVLine(v=t, c="r", alpha=0.2) for t in t_pred]
plotting.plot(
    [data_line] + pred_lines,
    "Tide height (m) vs time (days), including missing data points"
)