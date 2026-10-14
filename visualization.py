"""

    NOTE: The code from this file will just affect style and not model performance,
    so I considered acceptable to use AI assistance for writting it faster.

"""

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from pathlib import Path

NEURIPS_RC = {
    # fonts (STIX is metrically close to Times and always available)
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",

    # sizes for dense grids (final width 5.5 in); use 9/8 for single panels
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 6.5,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6.5,

    # thin lines
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2,    "ytick.major.size": 2,
    "xtick.minor.size": 1,    "ytick.minor.size": 1,

    # your structural settings
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "legend.frameon": False,

    # output
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,   # no Type 3 fonts
    "ps.fonttype": 42,
}
FT_COLORS = {
    "last-linear-layer": "#0072B2",
    "lora": "#D55E00",
    "full-model": "#009E73",
}

FT_MARKERS = {"last-linear-layer": "o", "lora": "s", "full-model": "^"}

SIZE_MAP_WIDTH   = {"gpt2": 25, "gpt2-medium": 55, "gpt2-large": 100}

TASK_MAP = {"sst": "Sentiment (SST)", "cfimdb": "Sentiment (CFIMDB)",
            "paraphrase": "Paraphrase", "sonnet_top_p": "Sonnet (Top P)", 
            "sonnet_beam": "Sonnet (Beam Search)"}

MODES_LR = {'last-linear-layer': ('Last linear layer', '1e-3'),
            'lora': ('LoRA', '1e-4'), 
            'full-model': ('Full model', '1e-5')}

MODE_LABELS = {"last-linear-layer": "LLL", "lora": "LoRA", "full-model": "Full"}
SIZE_LABELS = {"gpt2": "small", "gpt2-medium": "medium", "gpt2-large": "large"}

AGG_METRICS = [
    ('mean', 'performance', 'mean'),
    ('std', 'performance', 'std'),
    ('n', 'performance', 'count'),
    ('trainable_params', 'trainable_params', 'first'),
    ('total_params', 'total_params', 'first'),
    ('best_epoch_time_mean', 'best_epoch_time', 'mean'),
    ('best_epoch_time_std', 'best_epoch_time', 'std'),
    ('best_epoch_mean', 'best_epoch', 'mean'),
    ('best_epoch_std', 'best_epoch', 'std'),
    ("time_per_epoch_mean", "time_per_epoch", "mean"),
    ("time_per_epoch_std", "time_per_epoch", "std"),
    ('mem_alloc_mean', 'train_peak_allocated_gb', 'mean'),
    ('mem_alloc_std', 'train_peak_allocated_gb', 'std'),
    ('mem_alloc_max', 'train_peak_allocated_gb', 'max'),
]

# (x label, col in df, log scale)
COST_ROWS = [
    ("Trainable Parameters", "trainable_params", True),
    ("Peak Memory (GB)", "mem_alloc_mean", False),
    ("Time Per Epoch (min)", "time_per_epoch_mean",True),
]

def load_results_to_df(filepath, stop_line=None, start_line=None, comment_prefix='//'):
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):   # lineno starts at 1
            if start_line is not None and lineno <= start_line:
                continue
            if stop_line is not None and lineno > stop_line:
                break                          # stop once we pass the target
            line = line.strip()
            if not line or line.startswith(comment_prefix):
                continue
            records.append(json.loads(line))
    return pd.DataFrame(records)


def get_task(r):
    if pd.notna(r.get('mode')):       return r['mode']
    if pd.notna(r.get('para_train')): return 'paraphrase'
    if pd.notna(r.get('sonnet_train')):
        return f"sonnet_{r.get('generation_method', 'unknown')}"
    return 'unknown'


def aggregate_seeds(dfs, group_cols=('fine_tune_mode', 'model_size', 'task'), 
                    metrics=None, within_seed='max'):
    """
    Aggregate a list of per-seed DataFrames into one row per group.

    Parameters
    ----------
    dfs : list[pd.DataFrame]
        One DataFrame per seed. All must contain `group_cols`.
    metrics : list[tuple[str, str, str]] | None
        Each entry is (output_name, source_column, agg_func).
        - output_name: column name in the result
        - source_column: column to aggregate from the input
        - agg_func: pandas agg name — e.g. 'mean', 'std', 'count',
          'first', 'max', 'min', 'median'
        Example:
            [
                ('mean',            'performance',      'mean'),
                ('std',             'performance',      'std'),
                ('n',               'performance',      'count'),
                ('trainable_params','trainable_params', 'first'),
                ('time_std',        'best_epoch_time',  'std'),
            ]
        If None, uses some default ones.
    group_cols : tuple[str]
        Grouping keys.
    within_seed : str
        How to collapse duplicate rows within a single seed before
        cross-seed aggregation. 'max' is the usual choice for performance.

    Returns
    -------
    pd.DataFrame with columns: list(group_cols) + [output_name for each metric].
    """
    if metrics is None:
        metrics = AGG_METRICS

    # Check just in case
    if not isinstance(dfs, (list, tuple)) or len(dfs) == 0:
        raise TypeError("dfs must be a non-empty list/tuple of DataFrames.")

    needed = set(group_cols) | {src for _, src, _ in metrics}
    for i, d in enumerate(dfs):
        missing = needed - set(d.columns)
        if missing:
            raise KeyError(
                f"DataFrame at index {i} is missing columns: {sorted(missing)}. "
                f"Available: {list(d.columns)}"
            )

    # Tag seeds
    tagged = []
    for i, d in enumerate(dfs):
        d = d.copy()
        d['_seed'] = i
        tagged.append(d)
    all_df = pd.concat(tagged, ignore_index=True)

    # Collapse within seed
    sources = sorted({src for _, src, _ in metrics})
    per_seed = (all_df
                .groupby(list(group_cols) + ['_seed'])[sources]
                .agg(within_seed)
                .reset_index())

    # Aggregate across seeds
    agg_spec = {out: (src, func) for out, src, func in metrics}

    agg = (per_seed
           .groupby(list(group_cols))
           .agg(**{out: (src, func) for out, (src, func) in agg_spec.items()})
           .reset_index())

    return agg


def scatter_plot(df, metric1, metric2, log_scale=False, filepath=None):
    # metric1 = (name, col)  or (name, col, None)
    # metric2 = (name, mean_col, std_col_or_None)

    name1, col1 = metric1[0], metric1[1]

    name2   = metric2[0]
    mean2   = metric2[1]
    std2    = metric2[2] if len(metric2) > 2 else None

    show_std = std2 is not None and std2 in df.columns
    tasks = [t for t in df['task'].unique() if t != 'unknown']
    n = len(tasks)

    fig, axes = plt.subplots(1, n, figsize=(2.3 * n, 2.2), squeeze=False)
    axes = axes.flatten()

    for i, (ax, task) in enumerate(zip(axes, tasks)):
        sub = df[df['task'] == task]

        for ft, g in sub.groupby('fine_tune_mode'):
            color  = FT_COLORS.get(ft, 'gray')
            marker = FT_MARKERS.get(ft, 'o')
            sizes  = [SIZE_MAP_WIDTH.get(m, 40) for m in g['model_size']]

            # Mean scatter
            ax.scatter(
                g[col1], g[mean2],
                s=sizes,
                marker=marker,
                color=color,
                alpha=0.85, edgecolors='black', linewidths=0.3,
                zorder=3,
            )

            # error bars
            if show_std:
                yerr = g[std2].fillna(0).values
                ax.errorbar(
                    g[col1], g[mean2],
                    yerr=yerr,
                    fmt='none',
                    ecolor=color,
                    elinewidth=0.8,
                    capsize=2,
                    alpha=0.7,
                    zorder=2,
                )

        if log_scale:
            ax.set_xscale('log')
        ax.set_xlabel(name1)
        ax.set_ylabel(name2 if i == 0 else '')     # label once, leftmost
        ax.set_title(TASK_MAP[task], fontsize=9, pad=4)

    # Shared legends
    ft_handles = [
        plt.Line2D([], [], linestyle='', marker=FT_MARKERS[k], color=FT_COLORS[k],
                   markersize=6, markeredgecolor='black', markeredgewidth=0.3,
                   label=k)
        for k in FT_MARKERS
    ]
    size_handles = [
        plt.scatter([], [], s=s, color='gray', alpha=0.7,
                    edgecolors='black', linewidths=0.3, label=k)
        for k, s in SIZE_MAP_WIDTH.items()
    ]
    h1 = fig.legend(handles=ft_handles, loc='lower left',
                    bbox_to_anchor=(0.01, -0.05), ncol=3, frameon=False)
    fig.add_artist(h1)
    fig.legend(handles=size_handles, loc='lower right',
               bbox_to_anchor=(0.99, -0.05), ncol=3, frameon=False)

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if filepath:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath)

    plt.show()
    

def performance_table(df=None, mean_std_df=None, return_table=False):

    # Check just in case
    if (df is None) == (mean_std_df is None):
        raise ValueError("Provide exactly one of `df` or `mean_std_df`.")

    # Build lookup from given input
    if df is not None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"`df` must be a DataFrame, got {type(df).__name__}.")
        if 'performance' not in df.columns:
            raise KeyError("`df` is missing a 'performance' column.")

        agg = (df
               .groupby(['fine_tune_mode', 'model_size', 'task'])['performance']
               .max()
               .reset_index()
               .rename(columns={'performance': 'mean'}))
        agg['std'] = None

    else:
        if not isinstance(mean_std_df, pd.DataFrame):
            raise TypeError(f"`mean_std_df` must be a DataFrame, got {type(mean_std_df).__name__}.")
        for col in ('fine_tune_mode', 'model_size', 'task', 'mean', 'std'):
            if col not in mean_std_df.columns:
                raise KeyError(f"`mean_std_df` is missing column '{col}'.")
        agg = mean_std_df

    lookup = {}
    for _, r in agg.iterrows():
        lookup[(r['fine_tune_mode'], r['model_size'], r['task'])] = (r['mean'], r['std'])

    # Format
    def fmt(task_key, mean, std):
        if mean is None or pd.isna(mean):
            return '-'
        scale = 1 if task_key.startswith('sonnet') else 100
        if std is None or pd.isna(std):
            return f"{scale * mean:.1f}"
        return f"{scale * mean:.1f} ± {scale * std:.1f}"

    # Build the display table
    rows = []
    for mode_key in ['last-linear-layer', 'lora', 'full-model']:
        mode_name, lr = MODES_LR[mode_key]
        for i, size_key in enumerate(['gpt2', 'gpt2-medium', 'gpt2-large']):
            row = {
                'Mode (LR)': f"{mode_name} ({lr})" if i == 0 else '',
                'Size':      SIZE_LABELS[size_key],
            }
            for task_key, task_name in TASK_MAP.items():
                mean, std = lookup.get((mode_key, size_key, task_key), (None, None))
                row[task_name] = fmt(task_key, mean, std)
            rows.append(row)

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    if return_table:
        return table


def cost_scatter_grid(df, rows=COST_ROWS, score=("mean", "std"), tasks=None,
                      connect=True, min_n=3, skip=(), width=5.5,
                      row_height=1.45, filepath=None):
    """
    rows   : list of (xlabel, column, log) -> one grid row each
    score  : (mean_col, std_col) of the dev metric (y axis)
    tasks  : task keys to show (one column each); default = all in df
    connect: thin line joining each mode's points from small to large
    min_n  : cells with fewer seeds are drawn hollow
    skip   : set of (task, x_column) panels to leave blank
    width  : figure width in inches (NeurIPS text width = 5.5)
    """
    mean_col, std_col = score
    if tasks is None:
        tasks = [t for t in df["task"].unique() if t != "unknown"]
    modes = [m for m in FT_MARKERS if m in set(df["fine_tune_mode"])]
    order = {s: i for i, s in enumerate(SIZE_LABELS)}
    any_hollow = False

    with plt.rc_context(NEURIPS_RC):
        fig, axes = plt.subplots(
            len(rows), len(tasks), sharex="row", sharey="col", squeeze=False,
            figsize=(width, row_height * len(rows) + 0.45),
        )

        for r, (xlabel, xcol, xlog) in enumerate(rows):
            for c, task in enumerate(tasks):
                ax = axes[r, c]
                if r == 0:
                    ax.set_title(TASK_MAP.get(task, task), pad=8)
                if (task, xcol) in skip:
                    ax.text(0.5, 0.5, "not shown", ha="center", va="center",
                            transform=ax.transAxes, fontsize=6, color="gray")
                    continue

                scale = 1.0 if task.startswith("sonnet") else 100.0  # acc -> %
                sub = df[df["task"] == task]

                for ft in modes:
                    g = sub[sub["fine_tune_mode"] == ft].copy()
                    if g.empty:
                        continue
                    g["_o"] = g["model_size"].map(order)
                    g = g.sort_values("_o")

                    x = g[xcol].to_numpy(float)
                    y = g[mean_col].to_numpy(float) * scale
                    e = g[std_col].fillna(0).to_numpy(float) * scale
                    s = np.array([SIZE_MAP_WIDTH.get(m, 40) for m in g["model_size"]])
                    hollow = (g["n"].to_numpy() < min_n) if "n" in g \
                        else np.zeros(len(g), bool)
                    color = FT_COLORS.get(ft, "gray")
                    marker = FT_MARKERS.get(ft, "o")

                    if connect and len(g) > 1:
                        ax.plot(x, y, color=color, lw=0.6, alpha=0.5, zorder=1)
                    ax.errorbar(x, y, yerr=e, fmt="none", ecolor=color,
                                elinewidth=0.7, capsize=1.5, alpha=0.8, zorder=2)
                    ax.scatter(x[~hollow], y[~hollow], s=s[~hollow], marker=marker,
                               color=color, alpha=0.9, edgecolors="black",
                               linewidths=0.3, zorder=3)
                    if hollow.any():
                        any_hollow = True
                        ax.scatter(x[hollow], y[hollow], s=s[hollow], marker=marker,
                                   facecolors="none", edgecolors=color,
                                   linewidths=0.9, zorder=3)

                if xlog:
                    ax.set_xscale("log")
                    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
                    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
                ax.grid(alpha=0.25, linewidth=0.4)
                ax.spines[["top", "right"]].set_visible(False)
                ax.tick_params(pad=1.5)
                if c == 0:
                    ax.set_ylabel("Dev score")

        # single shared legend: mode (color + marker), then size (marker area)
        mode_h = [Line2D([], [], ls="", marker=FT_MARKERS[m], color=FT_COLORS[m],
                         markersize=4.5, markeredgecolor="black",
                         markeredgewidth=0.3, label=MODE_LABELS.get(m, m))
                  for m in modes]
        size_h = [Line2D([], [], ls="", marker="o", color="gray",
                         markersize=np.sqrt(SIZE_MAP_WIDTH[k]),
                         markeredgecolor="black", markeredgewidth=0.3,
                         label=SIZE_LABELS[k])
                  for k in SIZE_LABELS if k in SIZE_MAP_WIDTH]
        spacer = Line2D([], [], ls="", marker="", label=" ")
        handles = mode_h + [spacer] + size_h
        if any_hollow:
            handles.append(Line2D([], [], ls="", marker="o", markerfacecolor="none",
                                  markeredgecolor="black", markersize=4.5,
                                  label=f"$n<{min_n}$ seeds"))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0),
                   ncol=len(handles), frameon=False, columnspacing=1.0,
                   handletextpad=0.2, borderpad=0)

        # h_pad leaves room between rows for the row labels written below
        fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=2, w_pad=0.4)

        # one x-label per row, centred under that row's panels
        fig_h = fig.get_size_inches()[1]
        for r, (xlabel, _, _) in enumerate(rows):
            p0, p1 = axes[r, 0].get_position(), axes[r, -1].get_position()
            fig.text((p0.x0 + p1.x1) / 2, p0.y0 - 0.17 / fig_h, xlabel,
                    ha="center", va="top")

        # Save figure if a path is given
        if filepath:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(filepath, bbox_inches="tight")

    plt.show()

