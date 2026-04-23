import csv
import os
from typing import List, Dict, Any
from datetime import datetime


_CLASSIFICATION_FIELDS = [
    'timestamp', 'model_name', 'output_name', 'quant_format',
    'base_config_path', 'generated_config_path', 'status',
    'acc1', 'acc5', 'certainty', 'ref_acc1', 'ref_acc5', 'ref_certainty',
    'acc_drop', 'weight_comp_red', 'weight_comp_share', 'input_comp_red', 'input_comp_share',
    'exec_error', 'report_path',
]

_FM_FIELDS = [
    'timestamp', 'model_name', 'output_name', 'quant_format',
    'base_config_path', 'generated_config_path', 'status',
    'fm_num_keypoints', 'fm_mean_score', 'fm_desc_norm', 'fm_repeatability',
    'matching_precision', 'matching_score', 'mean_num_matches',
    'pose_auc_5', 'pose_auc_10', 'pose_auc_20',
    'ref_matching_precision', 'ref_matching_score', 'ref_mean_num_matches',
    'ref_pose_auc_5', 'ref_pose_auc_10', 'ref_pose_auc_20',
    'acc_drop', 'weight_comp_red', 'weight_comp_share',
    'exec_error', 'report_path',
]


def _is_fm(result: Dict[str, Any]) -> bool:
    return result.get('adapter_type') == 'feature_matching'


class ReportAggregator:
    def __init__(self):
        pass

    def aggregate(self, results: List[Dict[str, Any]], output_file: str):
        if not results:
            print("No results to aggregate.")
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for res in results:
            res['timestamp'] = timestamp

        classification = [r for r in results if not _is_fm(r)]
        fm = [r for r in results if _is_fm(r)]

        stem, ext = os.path.splitext(output_file)
        ext = ext.lower() if ext else '.csv'

        if classification:
            self._write(classification, f"{stem}_classification{ext}", timestamp, task='classification')
        if fm:
            self._write(fm, f"{stem}_fm{ext}", timestamp, task='fm')

    def _write(self, results: List[Dict[str, Any]], output_file: str, timestamp: str, task: str):
        ext = os.path.splitext(output_file)[1].lower()
        if ext == '.csv':
            self._write_csv(results, output_file, task)
        elif ext == '.md':
            self._write_markdown(results, output_file, timestamp, task)
        else:
            print(f"Unsupported format {ext}, defaulting to CSV.")
            self._write_csv(results, output_file + '.csv', task)

    def _write_csv(self, results: List[Dict[str, Any]], output_file: str, task: str):
        fieldnames = _FM_FIELDS if task == 'fm' else _CLASSIFICATION_FIELDS
        file_exists = os.path.isfile(output_file)
        try:
            with open(output_file, 'a' if file_exists else 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                for res in results:
                    writer.writerow({k: res.get(k, '') for k in fieldnames})
            print(f"Summary report {'appended' if file_exists else 'saved'} to {output_file}")
        except Exception as e:
            print(f"Failed to write CSV report: {e}")

    def _write_markdown(self, results: List[Dict[str, Any]], output_file: str, timestamp: str, task: str):
        try:
            grouped = {}
            for res in results:
                key = res.get('base_config_path', 'Unknown Base Config')
                grouped.setdefault(key, []).append(res)

            file_exists = os.path.isfile(output_file)
            with open(output_file, 'a' if file_exists else 'w') as f:
                if not file_exists:
                    f.write("# Runspace Summary Report\n\n")
                f.write(f"## Run at {timestamp}\n\n")

                for base_conf_path, group in grouped.items():
                    base_conf_name = os.path.basename(base_conf_path) if base_conf_path != 'Unknown Base Config' else base_conf_path
                    f.write(f"### Base Config: {base_conf_name}\n\n")

                    if task == 'fm':
                        self._write_fm_md_table(f, group)
                    else:
                        self._write_classification_md_table(f, group)

                    f.write("\n")

            print(f"Summary report {'appended' if file_exists else 'saved'} to {output_file}")
        except Exception as e:
            print(f"Failed to write Markdown report: {e}")

    def _write_classification_md_table(self, f, group):
        f.write("| Model | Quant Format | Generated Config | Status | Top-1 Acc | Top-5 Acc | Certainty | Ref Top-1 | Ref Top-5 | Ref Certainty | Acc Drop | Weight Comp Red % | Input Comp Red % | Exec Error | Report |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for res in group:
            model    = res.get('model_name', 'N/A')
            fmt      = res.get('quant_format', 'N/A')
            gen_conf = os.path.basename(res.get('generated_config_path', 'N/A'))
            status   = res.get('status', 'N/A')
            acc1         = f"{res.get('acc1', 0):.4f}%"
            acc5         = f"{res.get('acc5', 0):.4f}%"
            certainty    = f"{res.get('certainty', 0):.4f}"
            ref_acc1     = f"{res.get('ref_acc1', 0):.4f}%"
            ref_acc5     = f"{res.get('ref_acc5', 0):.4f}%"
            ref_certainty = f"{res.get('ref_certainty', 0):.4f}"
            acc_drop     = f"{res.get('acc_drop', 0):.4f}%"
            w_str = f"{res.get('weight_comp_red', 0):.2f}% ({res.get('weight_comp_share', 0):.1f}%)"
            i_str = f"{res.get('input_comp_red', 0):.2f}% ({res.get('input_comp_share', 0):.1f}%)"
            exec_error = res.get('exec_error') or ""
            report     = res.get('report_path', '')
            f.write(f"| {model} | {fmt} | {gen_conf} | {status} | {acc1} | {acc5} | {certainty} | {ref_acc1} | {ref_acc5} | {ref_certainty} | {acc_drop} | {w_str} | {i_str} | {exec_error} | {report} |\n")

    def _write_fm_md_table(self, f, group):
        f.write("| Model | Quant Format | Generated Config | Status | FM Repeatability | Ref FM Repeatability | Match Precision | Ref Match Precision | Match Score | Num Matches | Pose AUC@5 | Pose AUC@10 | Pose AUC@20 | Acc Drop | Weight Comp % | Exec Error | Report |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for res in group:
            model    = res.get('model_name', 'N/A')
            fmt      = res.get('quant_format', 'N/A')
            gen_conf = os.path.basename(res.get('generated_config_path', 'N/A'))
            status   = res.get('status', 'N/A')

            def _fmt(key, decimals=4):
                v = res.get(key)
                return f"{float(v):.{decimals}f}" if v else ""

            fm_rep      = _fmt('fm_repeatability')
            ref_fm_rep  = _fmt('fm_repeatability') if not res.get('ref_fm_repeatability') else _fmt('ref_fm_repeatability')
            prec        = _fmt('matching_precision')
            ref_prec    = _fmt('ref_matching_precision')
            score       = _fmt('matching_score')
            matches     = _fmt('mean_num_matches', 1)
            auc5        = _fmt('pose_auc_5')
            auc10       = _fmt('pose_auc_10')
            auc20       = _fmt('pose_auc_20')
            acc_drop    = f"{res.get('acc_drop', 0):.4f}"
            w_str       = f"{res.get('weight_comp_red', 0):.2f}% ({res.get('weight_comp_share', 0):.1f}%)"
            exec_error  = res.get('exec_error') or ""
            report      = res.get('report_path', '')
            f.write(f"| {model} | {fmt} | {gen_conf} | {status} | {fm_rep} | {ref_fm_rep} | {prec} | {ref_prec} | {score} | {matches} | {auc5} | {auc10} | {auc20} | {acc_drop} | {w_str} | {exec_error} | {report} |\n")
