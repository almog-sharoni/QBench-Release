import csv
import os
import sys

def main():
    input_path = 'runspace/outputs/summary_report.csv'
    
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    data = {}
    all_types = set()

    with open(input_path, 'r') as f:
        # Handle potential whitespace in headers
        reader = csv.DictReader(f, skipinitialspace=True)
        # Clean up fieldnames just in case
        reader.fieldnames = [name.strip() for name in reader.fieldnames]

        for row in reader:
            # Strip whitespace from values
            model = row['model_name'].strip()
            quant = row['quant_format'].strip()
            config_path = row['base_config_path'].strip()
            
            config_basename = os.path.splitext(os.path.basename(config_path))[0]
            type_str = f"{quant}_{config_basename}"
            all_types.add(type_str)

            if model not in data:
                data[model] = {
                    'ref_acc1': row['ref_acc1'].strip(),
                    'ref_acc5': row['ref_acc5'].strip(),
                    'configs': {}
                }
            
            data[model]['configs'][type_str] = {
                'acc1': row['acc1'].strip(),
                'acc5': row['acc5'].strip()
            }

    sorted_types = sorted(list(all_types))
    
    # Header
    header = ['model', 'ref_acc1', 'ref_acc5']
    for t in sorted_types:
        header.append(f"acc1_{t}")
        header.append(f"acc5_{t}")
    
    output_path = 'runspace/outputs/transposed_summary_report.csv'
    md_output_path = 'runspace/outputs/transposed_summary_report.md'
    
    # Write to CSV
    with open(output_path, 'w') as f:
        f.write(','.join(header) + '\n')
        for model in sorted(data.keys()):
            row_items = [model, data[model]['ref_acc1'], data[model]['ref_acc5']]
            for t in sorted_types:
                if t in data[model]['configs']:
                    row_items.append(data[model]['configs'][t]['acc1'])
                    row_items.append(data[model]['configs'][t]['acc5'])
                else:
                    row_items.append('')
                    row_items.append('')
            f.write(','.join(row_items) + '\n')

    print(f"Transposed report saved to: {output_path}")

    # Write to Markdown
    with open(md_output_path, 'w') as f:
        # Header
        f.write('| ' + ' | '.join(header) + ' |\n')
        # Separator
        f.write('| ' + ' | '.join(['---'] * len(header)) + ' |\n')
        
        for model in sorted(data.keys()):
            row_items = [model, data[model]['ref_acc1'], data[model]['ref_acc5']]
            for t in sorted_types:
                if t in data[model]['configs']:
                    row_items.append(data[model]['configs'][t]['acc1'])
                    row_items.append(data[model]['configs'][t]['acc5'])
                else:
                    row_items.append('')
                    row_items.append('')
            f.write('| ' + ' | '.join(row_items) + ' |\n')

    print(f"Transposed report (MD) saved to: {md_output_path}")
    
    # Also print to stdout as before (optional, but good for verification)
    print(','.join(header))
    for model in sorted(data.keys()):
        row_items = [model, data[model]['ref_acc1'], data[model]['ref_acc5']]
        for t in sorted_types:
            if t in data[model]['configs']:
                row_items.append(data[model]['configs'][t]['acc1'])
                row_items.append(data[model]['configs'][t]['acc5'])
            else:
                row_items.append('')
                row_items.append('')
        print(','.join(row_items))

if __name__ == "__main__":
    main()
