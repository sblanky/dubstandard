import matplotlib.pyplot as plt
from pygaps.graphing.calc_graphs import dra_plot

def drafit(
    filteredresult,
    ax=None,
    show=False,
):
    f = filteredresult
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        dra_plot(
            f.log_v, f.log_p_exp,
            minimum=f.i, maximum=f.j,
            slope=f.fit_grad[f.i, f.j],
            intercept=f.fit_intercept[f.i, f.j],
            exp=f.exp,
            ax=ax,
        )

    if show:
        plt.show()

    return fig, ax
