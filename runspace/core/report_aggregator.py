import csv
import os
from typing import List, Dict, Any
from datetime import datetime

_FIXED_FIELDS = [
    'timestamp', 'model_name', 'output_name', 'quant_format',
    'base_config_path', 'generated_config_path', 'status',
]
_COMPRESSION_FIELDS = [
    'weight_comp_red', 'weight_comp_share',
    'input_comp_red', 'input_comp_share',
]
_TAIL_FIELDS = ['primary_metric_drop', 'exec_error', 'report_path']
_KNOWN_NON_METRIC = set(_FIXED_FIELDS + _COMPRESSION_FIELDS + _TAIL_FIELDS)

# Human-readable labels for known metric keys (used in Markdown headers)
_METRIC_LABELS = {
    'acc1': 'Top-1 Acc',
    'acc5': 'Top-5 Acc',
    'ref_acc1': 'Ref Top-1',
    'ref_acc5': 'Ref Top-5',
    'certainty': 'Certainty',
    'ref_certainty': 'Ref Certainty',
    'ppl': 'PPL',
    'ref_ppl': 'Ref PPL',
}

# Metric keys whose values should be displayed with a trailing '%' in Markdown
_PERCENTAGE_METRICS = {'acc1', 'acc5', 'ref_acc1', 'ref_acc5'}

def _collect_metric_keys(results: List[Dict[str, Any]]) -> List[str]:
    """Return metric keys in insertion order, deduplicated, excluding infrastructure keys."""
    seen = set()
    keys = []
    for res in results:
        for k in res:
            if k not in _KNOWN_NON_METRIC and k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


class ReportAggregator:
    def __init__(self):
        pass

    def aggregate(self, results: List[Dict[str, Any]], output_file: str):
        """
        Aggregates run results into a single file.

        Args:
            results: List of result dictionaries from Runner.
            output_file: Path to the output file (e.g., summary.csv or summary.md).
        """
        if not results:
            print("No results to aggregate.")
            return

        # Add timestamp to results
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for res in results:
            res['timestamp'] = timestamp

        # Determine format based on extension
        ext = os.path.splitext(output_file)[1].lower()

        if ext == '.csv':
            self._write_csv(results, output_file)
        elif ext == '.md':
            self._write_markdown(results, output_file, timestamp)
        else:
            raise ValueError(f"Unsupported report format '{ext}'. Use .csv or .md.")

    def _write_csv(self, results: List[Dict[str, Any]], output_file: str):
        metric_keys = _collect_metric_keys(results)
        fieldnames = _FIXED_FIELDS + metric_keys + _COMPRESSION_FIELDS + _TAIL_FIELDS

        file_exists = os.path.isfile(output_file)

        try:
            with open(output_file, 'a' if file_exists else 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                if not file_exists:
                    writer.writeheader()
                writer.writerows(results)
            print(f"Summary report {'appended' if file_exists else 'saved'} to {output_file}")
        except Exception as e:
            print(f"Failed to write CSV report: {e}")

    def _write_markdown(self, results: List[Dict[str, Any]], output_file: str, timestamp: str):
        try:
            metric_keys = _collect_metric_keys(results)
            metric_headers = [_METRIC_LABELS.get(k, k.replace('_', ' ').title()) for k in metric_keys]

            # Group results by base_config_path
            grouped_results: Dict[str, List[Dict[str, Any]]] = {}
            for res in results:
                base_conf_path = res.get('base_config_path', 'Unknown Base Config')
                if base_conf_path not in grouped_results:
                    grouped_results[base_conf_path] = []
                grouped_results[base_conf_path].append(res)

            file_exists = os.path.isfile(output_file)
            mode = 'a' if file_exists else 'w'

            with open(output_file, mode, encoding='utf-8') as f:
                if not file_exists:
                    f.write("# Runspace Summary Report\n\n")

                f.write(f"## Run at {timestamp}\n\n")

                for base_conf_path, group in grouped_results.items():
                    base_conf_name = (
                        os.path.basename(base_conf_path)
                        if base_conf_path != 'Unknown Base Config'
                        else base_conf_path
                    )
                    f.write(f"### Base Config: {base_conf_name}\n\n")

                    headers = (
                        ['Model', 'Quant Format', 'Generated Config', 'Status']
                        + metric_headers
                        + ['Weight Comp Red %', 'Input Comp Red %', 'Metric Drop', 'Exec Error', 'Report']
                    )
                    f.write('| ' + ' | '.join(headers) + ' |\n')
                    f.write('|' + '---|' * len(headers) + '\n')

                    for res in group:
                        model = res.get('model_name', 'N/A')
                        fmt = res.get('quant_format', 'N/A')
                        gen_conf = os.path.basename(res.get('generated_config_path', 'N/A'))
                        status = res.get('status', 'N/A')

                        metric_cells = []
                        for k in metric_keys:
                            val = res.get(k)
                            if val is None:
                                metric_cells.append('N/A')
                            elif isinstance(val, float):
                                suffix = '%' if k in _PERCENTAGE_METRICS else ''
                                metric_cells.append(f"{val:.4f}{suffix}")
                            else:
                                metric_cells.append(str(val))

                        w_red = res.get('weight_comp_red', 0)
                        w_share = res.get('weight_comp_share', 0)
                        w_str = f"{w_red:.2f}% ({w_share:.1f}%)"

                        i_red = res.get('input_comp_red', 0)
                        i_share = res.get('input_comp_share', 0)
                        i_str = f"{i_red:.2f}% ({i_share:.1f}%)"

                        drop = res.get('primary_metric_drop')
                        drop_str = f"{drop:.4f}" if drop is not None else 'N/A'

                        exec_error = res.get('exec_error') or ''
                        report = res.get('report_path', '')

                        row = (
                            [model, fmt, gen_conf, status]
                            + metric_cells
                            + [w_str, i_str, drop_str, exec_error, report]
                        )
                        f.write('| ' + ' | '.join(row) + ' |\n')

                    f.write('\n')

            print(f"Summary report {'appended' if file_exists else 'saved'} to {output_file}")
        except Exception as e:
            print(f"Failed to write Markdown report: {e}")
