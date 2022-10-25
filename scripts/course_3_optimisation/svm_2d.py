import numpy as np
import svm
import __init__
import plotting
import scripts.course_3_optimisation

def make_plot(n=1000, x_offset=6, y_offset=2, norm_penalty=0.1):
    rng = np.random.default_rng(0)
    x = rng.normal(size=[n, 2])
    y = np.where(rng.random(size=n) > 0.5, 1, -1)
    x[y == -1] += np.array([x_offset, y_offset])

    a, b = svm.solve(x, y, norm_penalty)

    x0_min, x1_min = np.min(x, axis=0)
    x0_max, x1_max = np.max(x, axis=0)
    x0_array = np.array([x0_min, x0_max])
    x1_array = (-b - a[0] * x0_array) / a[1]

    marker_kwargs = {"marker": "o", "ls": "", "alpha": 0.5}
    blue_xy = [x[y==1][:, 0], x[y==1][:, 1]]
    red_xy = [x[y==-1][:, 0], x[y==-1][:, 1]]
    x_lim = np.array([-5, 13])
    y_margin = (-b - a[0] * x_lim) / a[1]
    y_margin_blue = (1 -b - a[0] * x_lim) / a[1]
    y_margin_red = (-1 -b - a[0] * x_lim) / a[1]

    filename = plotting.plot(
        plotting.Line(*blue_xy, c="b", **marker_kwargs, label="y = 1"),
        plotting.Line(*red_xy, c="r", **marker_kwargs, label="y = -1"),
        plotting.Line(x_lim, y_margin, c="k", label="Margin"),
        plotting.Line(x_lim, y_margin_blue, c="b", ls="--"),
        plotting.Line(x_lim, y_margin_red, c="r", ls="--"),
        plot_name="SVM 2D, x_offset = %.3f" % x_offset,
        dir_name=scripts.course_3_optimisation.RESULTS_DIR,
        axis_properties=plotting.AxisProperties(xlim=x_lim, ylim=[-5, 6]),
        legend_properties=plotting.LegendProperties(),
    )
    return filename

make_plot()
make_plot(x_offset=10)
make_plot(x_offset=1)

filename_list = []
for x in 5 * (1 + np.sin(np.linspace(0, 2 * np.pi))):
    filename_list.append(make_plot(x_offset=x))

plotting.make_gif(
    *filename_list,
    output_name="2D SVM gif",
    output_dir=scripts.course_3_optimisation.RESULTS_DIR,
    frame_duration_ms=2*1000/50,
)
