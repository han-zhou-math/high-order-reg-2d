"""Physical panel dimensions shared by the revised manuscript's error plots."""
import re

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullLocator

CELL_WIDTH = 3.25
ROW_HEIGHT = 2.85
PANEL_WIDTH = 2.48
PANEL_HEIGHT = 1.70


def configure():
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8,
        "lines.linewidth": 1.2, "lines.markersize": 4,
        "axes.linewidth": .7, "pdf.fonttype": 42,
    })


def align_panels(fig, axes, columns=2, rows=1, center_last=False):
    """Use identical plotting areas in inches, independent of panel count."""
    width, height = CELL_WIDTH * columns, ROW_HEIGHT * rows
    fig.set_size_inches(width, height)
    for index, ax in enumerate(axes):
        row, column = divmod(index, columns)
        left = CELL_WIDTH * column + .62
        if center_last and index == len(axes) - 1:
            left += CELL_WIDTH / 2
        bottom = ROW_HEIGHT * (rows - row - 1) + .48
        ax.set_position([left / width, bottom / height,
                         PANEL_WIDTH / width, PANEL_HEIGHT / height])
        ax.tick_params(which="major", labelsize=8, length=3, width=.7)
        ax.tick_params(which="minor", length=2, width=.5)
        ax.grid(False, which="both")
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(LogLocator(numticks=7))
        ax.yaxis.set_minor_locator(NullLocator())


def save_panels(fig, output):
    """Export one consistently sized, self-contained PDF per plot.

    Panel letters and grouping belong to the manuscript, not these assets.
    The fixed export box preserves identical axes dimensions in every PDF.
    """
    axes = list(fig.axes)
    for legend in fig.legends:
        legend.set_visible(False)
    for index, ax in enumerate(axes):
        handles = getattr(ax, "_publication_handles", None)
        if handles is None:
            handles = getattr(fig, "_publication_handles", None)
        if handles is None:
            handles, _ = ax.get_legend_handles_labels()
        labels = [handle.get_label() for handle in handles]
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        for line in list(ax.lines) + list(handles):
            if line.get_marker() not in (None, "None", "none", "", " "):
                line.set_markerfacecolor(line.get_color())
        labels = [label.replace("reference slopes", "Slopes")
                  .replace("Node-doubling difference", "Node doubling")
                  .replace("Stokes pressure", "Pressure")
                  .replace("Stokes velocity", "Velocity") for label in labels]
        # Fresh handles avoid inheriting marker placement from an old legend.
        handles = [Line2D([], [], color=h.get_color(), marker=h.get_marker(),
                          markerfacecolor=h.get_color(), markeredgecolor=h.get_color(),
                          markersize=4, linestyle=h.get_linestyle(), linewidth=h.get_linewidth())
                   for h in handles]
        if handles:
            ax.legend(handles, labels,
                      loc=getattr(ax, "_publication_legend_loc", "best"), fontsize=7,
                      ncol=getattr(ax, "_publication_legend_ncol", 2 if len(handles) >= 5 else 1),
                      frameon=True, facecolor="white", edgecolor="0.8",
                      framealpha=1, borderpad=.35, labelspacing=.25,
                      handlelength=1.6, handletextpad=.4, columnspacing=.7, numpoints=1)
        ax.grid(False, which="both", axis="both")
        ax.tick_params(labelleft=True, labelbottom=True)
        xlabel = ax.get_xlabel()
        for prefix in ("Smoothing width ", "Normal distance ", "Grid size "):
            xlabel = xlabel.replace(prefix, "")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Error")
        for loc in ("left", "center", "right"):
            title = re.sub(r"^\([a-z]\)\s*", "", ax.get_title(loc=loc))
            ax.set_title(title, loc=loc)
        fig.canvas.draw()
        bounds = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        crop = Bbox.from_bounds(bounds.x0 - .62, bounds.y0 - .44, CELL_WIDTH, 2.42)
        target = output if len(axes) == 1 else output.with_name(
            f"{output.stem}_{chr(97 + index)}{output.suffix}")
        fig.savefig(target, bbox_inches=crop, metadata={"CreationDate": None})
