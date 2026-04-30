import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plot_grouped_bar(
    data_dict,
    group_names,
    method_names,
    y_label="Performance",
    title=None,
    colors=None,
    figsize=(6.5, 4.2),
    bar_width=0.18,
    group_gap=0.9,
    ylim=None,
    show_values=False,
    value_fmt="{:.2f}",
    err_dict=None,
    edgecolor="black",
    linewidth=0.8,
    legend=True,
    legend_ncol=None,
    save_path=None,
    dpi=300,
):
    """
    Parameters
    ----------
    data_dict : dict
        形如:
        {
            "Method A": [0.71, 0.83, 0.65],
            "Method B": [0.75, 0.81, 0.69],
            "Method C": [0.79, 0.86, 0.73],
        }
        每个 key 是一个方法/模型，每个 value 对应每个 group 的数值

    group_names : list[str]
        x 轴大组名称，例如:
        ["K562", "MCF7", "A549"]

    method_names : list[str]
        柱子的排列顺序，例如:
        ["Method A", "Method B", "Method C"]

    y_label : str
        y 轴标签

    title : str or None
        图标题

    colors : list[str] or None
        每个方法对应一种颜色，长度应与 method_names 一致

    figsize : tuple
        图尺寸（英寸）

    bar_width : float
        单个柱宽

    group_gap : float
        组与组之间的间隔

    ylim : tuple or None
        y 轴范围，例如 (0.5, 0.9)

    show_values : bool
        是否在柱顶显示数值

    value_fmt : str
        数值显示格式

    err_dict : dict or None
        误差条，格式与 data_dict 相同，例如:
        {
            "Method A": [0.02, 0.01, 0.03],
            "Method B": [0.01, 0.02, 0.02],
            "Method C": [0.03, 0.02, 0.01],
        }

    save_path : str or None
        保存路径，例如 "benchmark_barplot.pdf"
    """

    n_groups = len(group_names)
    n_methods = len(method_names)

    if colors is None:
        # 偏论文风格的低饱和配色
        colors = [
            "#4C78A8", "#F58518", "#54A24B", "#E45756",
            "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D"
        ]
    if len(colors) < n_methods:
        raise ValueError("colors 的数量不足，请至少为每个 method 提供一种颜色。")

    # 检查输入
    for m in method_names:
        if m not in data_dict:
            raise ValueError(f"{m} 不在 data_dict 中。")
        if len(data_dict[m]) != n_groups:
            raise ValueError(f"{m} 的数值长度与 group_names 不一致。")
        if err_dict is not None:
            if m not in err_dict:
                raise ValueError(f"{m} 不在 err_dict 中。")
            if len(err_dict[m]) != n_groups:
                raise ValueError(f"{m} 的误差长度与 group_names 不一致。")

    # -------- figure style --------
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=figsize)

    # 计算每个 group 的中心位置
    group_centers = np.arange(n_groups) * (n_methods * bar_width + group_gap)

    # 每个 method 相对 group center 的偏移
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * bar_width

    bars_all = []

    for i, method in enumerate(method_names):
        x = group_centers + offsets[i]
        y = data_dict[method]
        yerr = err_dict[method] if err_dict is not None else None

        bars = ax.bar(
            x,
            y,
            width=bar_width,
            color=colors[i],
            edgecolor=edgecolor,
            linewidth=linewidth,
            yerr=yerr,
            capsize=2.5 if yerr is not None else 0,
            error_kw=dict(
                elinewidth=0.9,
                capthick=0.9,
                ecolor="black"
            ),
            zorder=3,
            label=method
        )
        bars_all.append(bars)

        if show_values:
            for rect, val in zip(bars, y):
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_height(),
                    value_fmt.format(val),
                    ha="center",
                    va="bottom",
                    fontsize=7
                )

    # -------- axes style --------
    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_names)
    ax.set_ylabel(y_label)

    if title is not None:
        ax.set_title(title, pad=8)

    if ylim is not None:
        ax.set_ylim(*ylim)

    # Nature 风格常见处理：去掉 top/right spine
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.tick_params(axis="both", width=1.0, length=4)
    ax.set_axisbelow(True)

    # 很浅的横向参考线
    ax.yaxis.grid(True, linestyle="-", linewidth=0.5, alpha=0.18)
    ax.xaxis.grid(False)

    if legend:
        if legend_ncol is None:
            legend_ncol = min(n_methods, 4)
        ax.legend(
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0,
            ncol=1 if n_methods <= 4 else legend_ncol
        )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", transparent=False)

    return fig, ax
