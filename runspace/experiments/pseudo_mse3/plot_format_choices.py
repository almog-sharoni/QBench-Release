import argparse
import csv
import math
import os


DEFAULT_PLOT_FILENAME = "format_choices_mse_vs_pseudo_mse3.png"


def _clean_csv_row(row):
    return {
        str(key).strip(): value.strip() if isinstance(value, str) else value
        for key, value in row.items()
    }


def read_mismatch_summary_rows(summary_csv):
    with open(summary_csv, newline="") as handle:
        return [_clean_csv_row(row) for row in csv.DictReader(handle)]


def format_choice_counts_by_bit_width(rows):
    """Collect MSE and pseudo_MSE3 e1/e2 counts from compact summary rows."""
    grouped = {}
    for raw_row in rows:
        row = _clean_csv_row(raw_row)
        bit_width = int(row["bit_width"])
        bits_to_take = int(row["bits_to_take"])
        mse_counts = {
            "e1": int(row["runtime_mse_e1"]),
            "e2": int(row["runtime_mse_e2"]),
        }
        pseudo_counts = {
            "e1": int(row["pseudo_e1"]),
            "e2": int(row["pseudo_e2"]),
        }

        bucket = grouped.setdefault(
            bit_width,
            {
                "mse": None,
                "pseudo": {},
                "fixed_rounding": set(),
                "tie_break": set(),
            },
        )
        if bucket["mse"] is not None and bucket["mse"] != mse_counts:
            raise ValueError(
                f"Inconsistent MSE choice counts for {bit_width}-bit rows"
            )
        existing = bucket["pseudo"].get(bits_to_take)
        if existing is not None and existing != pseudo_counts:
            raise ValueError(
                f"Conflicting pseudo_MSE3 counts for {bit_width}-bit "
                f"bits_to_take={bits_to_take}"
            )

        bucket["mse"] = mse_counts
        bucket["pseudo"][bits_to_take] = pseudo_counts
        bucket["fixed_rounding"].add(row.get("fixed_rounding", "floor"))
        bucket["tie_break"].add(row.get("tie_break", "exp1"))

    return dict(sorted(grouped.items()))


def _bits_to_take_values(grouped):
    return sorted(
        {
            bits_to_take
            for bucket in grouped.values()
            for bits_to_take in bucket["pseudo"]
        }
    )


def _series_label(bits_to_take):
    return f"Pseudo N={bits_to_take}"


def _mode_label(grouped):
    rounding = {
        value
        for bucket in grouped.values()
        for value in bucket["fixed_rounding"]
        if value
    }
    tie_break = {
        value
        for bucket in grouped.values()
        for value in bucket["tie_break"]
        if value
    }
    if len(rounding) != 1 or len(tie_break) != 1:
        return None
    return f"rounding={next(iter(rounding))}, tie={next(iter(tie_break))}"


def plot_format_choice_counts_from_rows(
    rows,
    output_path,
    title="MSE vs Pseudo_MSE3 Format Selections",
    dpi=160,
):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.ticker import StrMethodFormatter
    except Exception as exc:
        print(f"[plot] skipped pseudo_MSE3 format choices ({exc})")
        return None

    grouped = format_choice_counts_by_bit_width(rows)
    if not grouped:
        print("[plot] no mismatch-summary rows; skipped format choices plot")
        return None

    bits_to_take_values = _bits_to_take_values(grouped)
    if not bits_to_take_values:
        print("[plot] no pseudo_MSE3 choice counts; skipped format choices plot")
        return None

    bit_widths = list(grouped)
    categories = [
        (bit_width, choice)
        for bit_width in bit_widths
        for choice in ("e1", "e2")
    ]
    max_local_series = max(
        1 + len(grouped[bit_width]["pseudo"])
        for bit_width in bit_widths
    )
    bar_width = min(0.8 / max_local_series, 0.22)
    figure_width = max(
        12.0,
        len(categories) * max(0.9, max_local_series * bar_width * 1.15),
    )
    legend_columns = min(max(3, math.ceil((len(bits_to_take_values) + 1) / 2)), 7)
    legend_rows = math.ceil((len(bits_to_take_values) + 1) / legend_columns)

    fig, ax = plt.subplots(figsize=(figure_width, 7.0 + 0.3 * max(legend_rows - 1, 0)))
    x_positions = list(range(len(categories)))

    for group_index, bit_width in enumerate(bit_widths):
        start = group_index * 2
        end = start + 1
        if group_index % 2 == 0:
            ax.axvspan(start - 0.5, end + 0.5, color="0.94", zorder=0)
        if start > 0:
            ax.axvline(
                start - 0.5,
                color="0.45",
                linestyle="--",
                linewidth=0.8,
                alpha=0.7,
                zorder=1,
            )
        ax.text(
            (start + end) / 2.0,
            -0.22,
            f"{bit_width}-bit",
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=10,
            fontweight="bold",
            clip_on=False,
        )

    cmap = plt.get_cmap("tab20")
    pseudo_colors = {
        bits_to_take: cmap(index % 20)
        for index, bits_to_take in enumerate(bits_to_take_values)
    }
    mse_color = "#252525"

    for x, (bit_width, choice) in zip(x_positions, categories):
        bucket = grouped[bit_width]
        local_bits = sorted(bucket["pseudo"])
        local_series = [("mse", None)] + [
            ("pseudo", bits_to_take) for bits_to_take in local_bits
        ]
        for series_index, (kind, bits_to_take) in enumerate(local_series):
            offset = (
                series_index - (len(local_series) - 1) / 2.0
            ) * bar_width
            if kind == "mse":
                value = bucket["mse"][choice]
                color = mse_color
                hatch = "//"
            else:
                value = bucket["pseudo"][bits_to_take][choice]
                color = pseudo_colors[bits_to_take]
                hatch = None
            ax.bar(
                x + offset,
                value,
                width=bar_width,
                color=color,
                edgecolor="white" if hatch is None else "#606060",
                linewidth=0.35,
                hatch=hatch,
                zorder=3,
            )

    mode_label = _mode_label(grouped)
    full_title = f"{title}\n{mode_label}" if mode_label else title
    fig.suptitle(full_title, y=0.98)
    ax.set_xlabel("Selected exponent format, grouped by quantizer bit width", labelpad=48)
    ax.set_ylabel("Format selections (chunks)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([choice for _bit_width, choice in categories])
    ax.set_xlim(-0.5, len(categories) - 0.5)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    legend_handles = [
        Patch(
            facecolor=mse_color,
            edgecolor="#606060",
            hatch="//",
            label="MSE",
        )
    ]
    legend_handles.extend(
        Patch(
            facecolor=pseudo_colors[bits_to_take],
            label=_series_label(bits_to_take),
        )
        for bits_to_take in bits_to_take_values
    )
    ax.legend(
        handles=legend_handles,
        title="Decision Method",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17 + 0.055 * max(legend_rows - 1, 0)),
        ncol=legend_columns,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.subplots_adjust(
        top=max(0.62, 0.78 - 0.045 * max(legend_rows - 1, 0)),
        bottom=0.25,
    )
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def default_output_path(summary_csv):
    return os.path.join(os.path.dirname(summary_csv), DEFAULT_PLOT_FILENAME)


def regenerate_plot(summary_csv, output_path=None, title=None, dpi=160):
    rows = read_mismatch_summary_rows(summary_csv)
    return plot_format_choice_counts_from_rows(
        rows=rows,
        output_path=output_path or default_output_path(summary_csv),
        title=title or "MSE vs Pseudo_MSE3 Format Selections",
        dpi=dpi,
    )


def get_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Plot MSE and pseudo_MSE3 e1/e2 format-choice counts from an "
            "existing mismatch_summary.csv."
        )
    )
    parser.add_argument(
        "--summary-csv",
        "--summary_csv",
        required=True,
        help="Path to mismatch_summary.csv.",
    )
    parser.add_argument("--output", default=None, help="Output PNG path.")
    parser.add_argument("--title", default=None, help="Override the plot title.")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args(argv)


def main(argv=None):
    args = get_args(argv)
    path = regenerate_plot(
        summary_csv=args.summary_csv,
        output_path=args.output,
        title=args.title,
        dpi=args.dpi,
    )
    if path:
        print(f"Plot written to {path}")


if __name__ == "__main__":
    main()
