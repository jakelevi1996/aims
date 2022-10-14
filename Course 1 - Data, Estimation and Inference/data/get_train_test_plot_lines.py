import data.parse
import plotting

def get_train_test_plot_lines(
    t_data,
    y_data,
    t_pred,
    data_dict=None,
    y_truth=None,
):
    if y_truth is None:
        if data_dict is None:
            raise ValueError("Must provide either y_truth or data_dict")
        y_truth = data.parse.parse_column(data_dict, "True tide height (m)")

    train_data_line = plotting.Line(
        t_data,
        y_data,
        c="k",
        ls="",
        marker="o",
        alpha=0.5,
        zorder=20,
    )
    test_data_line = plotting.Line(
        t_pred,
        y_truth,
        c="k",
        ls="",
        marker="x",
        alpha=0.5,
        zorder=20,
    )
    return train_data_line, test_data_line
