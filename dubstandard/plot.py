import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pygaps.graphing.calc_graphs as pgcc

roqhandles = [
    'all points',
    'chosen points',
    'characteristic'
]

def drafit(
    filteredresult,
    ax=None,
    show=False,
):
    f = filteredresult
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    pgcc.dra_plot(
        f.log_v, f.log_p_exp,
        minimum=f.i, maximum=f.j,
        slope=f.fit_grad[f.i, f.j],
        intercept=f.fit_intercept[f.i, f.j],
        exp=f.exp,
        ax=ax,
    )

    if show:
        plt.show()

    return ax


def roqfit(
    filteredresult,
    ax=None,
    show=False,
):
    f = filteredresult
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    pgcc.roq_plot(
        f.pressure, f.rouq_y,
        f.i, f.j,
        f.characteristic_pressure,
        f.rouq_y[f.characteristic_index],
        ax=ax,
    )
    ax.legend(
        roqhandles,
        frameon=False,
        loc='best'
    )

    if show:
        plt.show()

    return ax


def expandroqfit(
    filteredresult,
    ax=None,
    show=False
):
    f = filteredresult
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    pgcc.roq_plot(
        f.pressure, f.ultrarouq_y,
        f.i, f.j,
        f.characteristic_pressure,
        f.ultrarouq_y[f.characteristic_index],
        ax=ax,
    )
    ax.semilogx()
    ax.set_title('Expanded Rouquerol')
    ax.set_ylabel(r'$\Theta ln{p^0/p}$')
    ax.set_xlabel(r'$p/p^0$')
    ax.invert_xaxis()
    ax.legend(
        roqhandles,
        frameon=False,
        loc='best'
    )

    if show:
        plt.show()

    return ax


def create_standard_plot(
    filteredresult,
    fig=None,
    show=True,
    size=[9.52756, 6.29921],
):
    f = filteredresult
    if fig is None:
        fig = plt.figure(figsize=(size[0], size[1]))
    fig.set_size_inches(size[0], size[1])
    fig.suptitle('')

    gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    roqfit(f, ax=ax)

    ax = fig.add_subplot(gs[0, 1])
    expandroqfit(f, ax=ax)

    ax = fig.add_subplot(gs[1, 1])
    drafit(f, ax=ax)

    plt.tight_layout()
    if show:
        plt.show()
    return fig
