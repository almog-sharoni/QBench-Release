import argparse
import csv
import json
import os
import re
from typing import Dict, Iterable, List


FORMAT_RE = re.compile(r"^(uefp|ufp|efp|fp)(\d+)(?:_e(\d+)(?:m(\d+))?)?")
DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _safe_json_load(value):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return None


def _format_parts(fmt):
    text = str(fmt or "").strip().lower()
    match = FORMAT_RE.match(text)
    if not match:
        return None
    prefix, bits, exp, mant = match.groups()
    return {
        "prefix": prefix,
        "bits": int(bits),
        "exp": int(exp) if exp is not None else -1,
        "mant": int(mant) if mant is not None else -1,
        "text": text,
    }


def _format_sort_key(fmt):
    parts = _format_parts(fmt)
    if parts is None:
        return (999, 999, 999, 999, str(fmt))

    prefix_order = {"fp": 0, "efp": 1, "ufp": 2, "uefp": 3}.get(
        parts["prefix"],
        9,
    )
    return (
        parts["bits"],
        prefix_order,
        parts["exp"],
        parts["mant"],
        parts["text"],
    )


def _format_bit_width(fmt):
    parts = _format_parts(fmt)
    return parts["bits"] if parts is not None else None


def _format_exp_bits(fmt):
    parts = _format_parts(fmt)
    return parts["exp"] if parts is not None else None


def _format_category_key(fmt):
    parts = _format_parts(fmt)
    if parts is None or parts["exp"] < 0:
        return f"raw:{str(fmt).strip().lower()}"
    return f"b{parts['bits']}_e{parts['exp']}"


def _category_sort_key(category):
    text = str(category or "").strip().lower()
    match = re.match(r"^b(\d+)_e(\d+)$", text)
    if not match:
        return (999, 999, text)
    bits, exp = match.groups()
    return (int(bits), int(exp), text)


def _category_bit_width(category):
    match = re.match(r"^b(\d+)_e\d+$", str(category or "").strip().lower())
    return int(match.group(1)) if match else None


def _category_label(category):
    text = str(category or "").strip()
    match = re.match(r"^b\d+_e(\d+)$", text.lower())
    if match:
        return f"e{match.group(1)}"
    if text.startswith("raw:"):
        return text[4:]
    return text


def _short_format_label(fmt):
    parts = _format_parts(fmt)
    if parts is None:
        return str(fmt)

    suffix = ""
    if parts["exp"] >= 0:
        suffix = f"e{parts['exp']}"
        if parts["mant"] >= 0:
            suffix += f"m{parts['mant']}"
    return f"{parts['prefix']} {suffix}".strip()


def _sort_quant_formats(formats: Iterable[str]):
    return sorted(
        {str(fmt) for fmt in formats if str(fmt or "").strip()},
        key=_format_sort_key,
    )


def _sort_categories(categories: Iterable[str]):
    return sorted(
        {str(category) for category in categories if str(category or "").strip()},
        key=_category_sort_key,
    )


def _exp_cap_sort_key(exp_cap):
    text = str(exp_cap or "").strip().lower()
    if text == "all":
        return (0, 0, text)
    if text.startswith("exp"):
        try:
            return (1, -int(text[3:]), text)
        except Exception:
            pass
    return (2, 0, text)


def _add_format_count(totals: Dict[str, int], fmt, count=1):
    if fmt is None:
        return

    key = str(fmt).strip()
    if not key:
        return

    try:
        count_i = int(count)
    except Exception:
        return
    if count_i <= 0:
        return

    totals[key] = totals.get(key, 0) + count_i


def _format_counts_from_summary_row(row):
    counts = _safe_json_load(row.get("format_counts_json"))
    if not isinstance(counts, dict):
        return {}

    totals = {}
    for fmt, count in counts.items():
        _add_format_count(totals, _format_category_key(fmt), count)
    return totals


def _candidate_categories_from_summary_row(row):
    raw_candidates = str(row.get("candidate_formats") or "")
    categories = set()
    for fmt in raw_candidates.split(","):
        fmt = fmt.strip()
        if fmt:
            categories.add(_format_category_key(fmt))
    if categories:
        return categories
    return set(_format_counts_from_summary_row(row).keys())


def format_choice_counts_by_exp_cap(rows):
    series_counts = {}
    for row in rows:
        counts = _format_counts_from_summary_row(row)
        if not counts:
            continue

        exp_cap = str(row.get("exp_cap") or "unknown")
        target = series_counts.setdefault(exp_cap, {})
        for fmt, count in counts.items():
            _add_format_count(target, fmt, count)

    return {
        exp_cap: dict(sorted(counts.items(), key=lambda item: _category_sort_key(item[0])))
        for exp_cap, counts in sorted(
            series_counts.items(),
            key=lambda item: _exp_cap_sort_key(item[0]),
        )
    }


def available_exp_caps_by_category(rows):
    available = {}
    for row in rows:
        exp_cap = str(row.get("exp_cap") or "unknown")
        if not _format_counts_from_summary_row(row):
            continue
        for category in _candidate_categories_from_summary_row(row):
            available.setdefault(category, set()).add(exp_cap)
    return {
        category: [
            exp_cap
            for exp_cap in sorted(exp_caps, key=_exp_cap_sort_key)
        ]
        for category, exp_caps in available.items()
    }


def _bit_group_spans(categories: List[str]):
    spans = []
    start = 0
    current_bits = _category_bit_width(categories[0]) if categories else None

    for idx, category in enumerate(categories[1:], start=1):
        bits = _category_bit_width(category)
        if bits == current_bits:
            continue
        spans.append((current_bits, start, idx - 1))
        current_bits = bits
        start = idx

    if categories:
        spans.append((current_bits, start, len(categories) - 1))
    return spans


def plot_format_choice_counts_from_rows(
    rows,
    output_path,
    title="Dynamic Input Format Selections by Exp Cap",
    dpi=160,
):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] skipped format choices ({exc})")
        return None

    series_counts = format_choice_counts_by_exp_cap(rows)
    if not series_counts:
        print("[plot] no format_counts_json data found; skipped format choices plot")
        return None

    categories = _sort_categories(
        category for counts in series_counts.values() for category in counts.keys()
    )
    if not categories:
        print("[plot] no format categories found; skipped format choices plot")
        return None

    exp_caps = list(series_counts.keys())
    available_caps = available_exp_caps_by_category(rows)
    x_positions = list(range(len(categories)))
    max_local_caps = max(
        (len(available_caps.get(category, exp_caps)) for category in categories),
        default=len(exp_caps),
    )
    bar_width = min(0.8 / max(max_local_caps, 1), 0.22)
    figure_width = max(
        12,
        len(categories) * max(0.68, max_local_caps * bar_width * 0.9),
    )

    fig, ax = plt.subplots(figsize=(figure_width, 6.8))

    spans = _bit_group_spans(categories)
    for span_idx, (bits, start, end) in enumerate(spans):
        if span_idx % 2 == 0:
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

        center = (start + end) / 2.0
        group_label = f"{bits}-bit" if bits is not None else "other"
        ax.text(
            center,
            -0.30,
            group_label,
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=10,
            fontweight="bold",
            clip_on=False,
        )

    cap_to_color = {
        exp_cap: f"C{idx % 10}"
        for idx, exp_cap in enumerate(exp_caps)
    }
    for x, category in zip(x_positions, categories):
        local_caps = available_caps.get(category, exp_caps)
        for idx, exp_cap in enumerate(local_caps):
            offset = (idx - (len(local_caps) - 1) / 2.0) * bar_width
            y = series_counts.get(exp_cap, {}).get(category, 0)
            ax.bar(
                x + offset,
                y,
                width=bar_width,
                color=cap_to_color.get(exp_cap),
                zorder=3,
            )

    ax.set_xlabel("Format, grouped by bit width", labelpad=54)
    ax.set_ylabel("Format selections (chunks)")
    fig.suptitle(title, y=0.97)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [_category_label(category) for category in categories],
        rotation=60,
        ha="right",
    )
    ax.set_xlim(-0.5, len(categories) - 0.5)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    try:
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor=cap_to_color[exp_cap], label=str(exp_cap))
            for exp_cap in exp_caps
        ]
    except Exception:
        legend_handles = None
    ax.legend(
        handles=legend_handles,
        title="Exp Cap",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.11),
        ncol=min(len(exp_caps), 5),
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.subplots_adjust(top=0.78, bottom=0.34)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def read_summary_rows(summary_csv):
    with open(summary_csv, "r", newline="") as f:
        return list(csv.DictReader(f))


def _model_name_from_summary_path(summary_csv):
    stem = os.path.splitext(os.path.basename(summary_csv))[0]
    if stem == "summary":
        return None
    if stem.endswith("_summary"):
        return stem[: -len("_summary")]
    return stem


def default_output_path(summary_csv):
    model_name = _model_name_from_summary_path(summary_csv)
    output_dir = os.path.dirname(summary_csv)
    filename = (
        f"{model_name}_format_choices_by_exp_cap.png"
        if model_name
        else "format_choices_by_exp_cap.png"
    )
    return os.path.join(output_dir, filename)


def default_title(summary_csv):
    model_name = _model_name_from_summary_path(summary_csv)
    if model_name:
        return f"{model_name}: Dynamic Input Format Selections by Exp Cap"
    return "Dynamic Input Format Selections by Exp Cap"


def regenerate_plot(summary_csv, output_path=None, title=None, dpi=160):
    output_path = output_path or default_output_path(summary_csv)
    title = title or default_title(summary_csv)
    rows = read_summary_rows(summary_csv)
    return plot_format_choice_counts_from_rows(
        rows=rows,
        output_path=output_path,
        title=title,
        dpi=dpi,
    )


def discover_summary_csvs(results_dir, include_combined=False):
    if not os.path.isdir(results_dir):
        return []

    paths = []
    for name in sorted(os.listdir(results_dir)):
        if name == "summary.csv":
            if include_combined:
                paths.append(os.path.join(results_dir, name))
            continue
        if name.endswith("_summary.csv"):
            paths.append(os.path.join(results_dir, name))
    return paths


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate activation candidate sweep format-choice plots from "
            "summary CSV format_counts_json."
        )
    )
    parser.add_argument(
        "--summary_csv",
        action="append",
        default=[],
        help="Summary CSV to plot. Can be passed multiple times.",
    )
    parser.add_argument(
        "--results_dir",
        default=DEFAULT_RESULTS_DIR,
        help="Directory to scan when --summary_csv is omitted or --all is set.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Regenerate every *_summary.csv plot in --results_dir.",
    )
    parser.add_argument(
        "--include_combined",
        action="store_true",
        help="When using --all, also regenerate summary.csv as the combined plot.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Only valid with a single --summary_csv.",
    )
    parser.add_argument("--title", default=None, help="Override plot title.")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def main():
    args = get_args()

    summary_paths = list(args.summary_csv)
    if args.all:
        summary_paths.extend(
            discover_summary_csvs(
                args.results_dir,
                include_combined=args.include_combined,
            )
        )
    if not summary_paths:
        summary_paths = discover_summary_csvs(args.results_dir)

    summary_paths = list(dict.fromkeys(summary_paths))
    if not summary_paths:
        raise SystemExit("No summary CSV files found.")
    if args.output and len(summary_paths) != 1:
        raise SystemExit("--output is only valid with a single summary CSV.")

    for summary_csv in summary_paths:
        output_path = args.output if len(summary_paths) == 1 else None
        path = regenerate_plot(
            summary_csv=summary_csv,
            output_path=output_path,
            title=args.title if len(summary_paths) == 1 else None,
            dpi=args.dpi,
        )
        if path:
            print(f"Plot written to {path}")


if __name__ == "__main__":
    main()
