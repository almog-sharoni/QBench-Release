import csv
import os
from typing import List, Dict, Any
from datetime import datetime

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
            print(f"Unsupported format {ext}, defaulting to CSV.")
            self._write_csv(results, output_file + '.csv')

    def _write_csv(self, results: List[Dict[str, Any]], output_file: str):
        fieldnames = ['timestamp', 'model_name', 'output_name', 'quant_format', 'base_config_path', 'generated_config_path', 'status', 'acc1', 'acc5', 'certainty', 'ref_acc1', 'ref_acc5', 'ref_certainty', 'acc_drop', 'weight_comp_red', 'weight_comp_share', 'input_comp_red', 'input_comp_share', 'exec_error', 'report_path']
        
        file_exists = os.path.isfile(output_file)
        
        try:
            with open(output_file, 'a' if file_exists else 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                for res in results:
                    # Filter keys to match fieldnames
                    row = {k: res.get(k, '') for k in fieldnames}
                    writer.writerow(row)
            print(f"Summary report {'appended' if file_exists else 'saved'} to {output_file}")
        except Exception as e:
            print(f"Failed to write CSV report: {e}")

    def _write_markdown(self, results: List[Dict[str, Any]], output_file: str, timestamp: str):
        try:
            # Group results by base_config_path
            grouped_results = {}
            for res in results:
                base_conf_path = res.get('base_config_path', 'Unknown Base Config')
                if base_conf_path not in grouped_results:
                    grouped_results[base_conf_path] = []
                grouped_results[base_conf_path].append(res)

            file_exists = os.path.isfile(output_file)
            mode = 'a' if file_exists else 'w'

            with open(output_file, mode) as f:
                if not file_exists:
                    f.write("# Runspace Summary Report\n\n")
                
                f.write(f"## Run at {timestamp}\n\n")
                
                for base_conf_path, group in grouped_results.items():
                    base_conf_name = os.path.basename(base_conf_path) if base_conf_path != 'Unknown Base Config' else base_conf_path
                    f.write(f"### Base Config: {base_conf_name}\n\n")
                    f.write("| Model | Quant Format | Generated Config | Status | Top-1 Acc | Top-5 Acc | Certainty | Ref Top-1 | Ref Top-5 | Ref Certainty | Acc Drop | Weight Comp Red % | Input Comp Red % | Exec Error | Report |\n")
                    f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
                    
                    for res in group:
                        model = res.get('model_name', 'N/A')
                        fmt = res.get('quant_format', 'N/A')
                        gen_conf = os.path.basename(res.get('generated_config_path', 'N/A'))
                        status = res.get('status', 'N/A')
                        acc1 = f"{res.get('acc1', 0):.4f}%"
                        acc5 = f"{res.get('acc5', 0):.4f}%"
                        certainty = f"{res.get('certainty', 0):.4f}"
                        ref_acc1 = f"{res.get('ref_acc1', 0):.4f}%"
                        ref_acc5 = f"{res.get('ref_acc5', 0):.4f}%"
                        ref_certainty = f"{res.get('ref_certainty', 0):.4f}"
                        acc_drop = f"{res.get('acc_drop', 0):.4f}%"
                        
                        w_red = f"{res.get('weight_comp_red', 0):.2f}%"
                        w_share = f"{res.get('weight_comp_share', 0):.1f}%"
                        w_str = f"{w_red} ({w_share})"
                        
                        i_red = f"{res.get('input_comp_red', 0):.2f}%"
                        i_share = f"{res.get('input_comp_share', 0):.1f}%"
                        i_str = f"{i_red} ({i_share})"
                        
                        exec_error = res.get('exec_error') if res.get('exec_error') else ""
                        report = res.get('report_path', '')
                        
                        f.write(f"| {model} | {fmt} | {gen_conf} | {status} | {acc1} | {acc5} | {certainty} | {ref_acc1} | {ref_acc5} | {ref_certainty} | {acc_drop} | {w_str} | {i_str} | {exec_error} | {report} |\n")
                    
                    f.write("\n") # Add spacing between tables

            print(f"Summary report {'appended' if file_exists else 'saved'} to {output_file}")
        except Exception as e:
            print(f"Failed to write Markdown report: {e}")
