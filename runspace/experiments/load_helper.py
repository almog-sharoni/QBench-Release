def load_cached_results(csv_path, metric, format_col_suffix="_error"):
    """
    Load layer results from an existing CSV file.
    Returns:
       - results_list: List of dicts matching internal record structure (partial)
       - successful: Boolean indicating success
    """
    print(f"Loading cached results from {csv_path}...")
    try:
        results = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                layer = row['layer']
                
                # Reconstruct metrics dict
                metrics_data = {}
                best_error = float(row.get('best_error', float('inf')))
                
                # We need to reconstruct the errors for all formats
                # The CSV has columns like "fp8_e4m3_error"
                for key, val in row.items():
                    if key.endswith(format_col_suffix):
                         fmt = key.replace(format_col_suffix, "")
                         try:
                             if val and val != "":
                                 metrics_data[fmt] = float(val)
                             else:
                                 metrics_data[fmt] = float('inf')
                         except:
                             metrics_data[fmt] = float('inf')
                             
                record = {
                    'layer': layer,
                    'shape': row.get('shape', ''),
                    'max_val': float(row.get('max_val', 0.0)),
                    'metrics': {metric: metrics_data},
                    # We might not have chunk data in CSV easily if strictly parsing cols,
                    # but if 'chunk_format_distribution' is there, we can parse it?
                    # For now, minimal reconstruction for 'Pre-calculate theoretical errors' 
                    # which needs 'metrics' and 'layer'.
                    'best_error': best_error
                }
                
                # Attempt to load chunk data if present
                if 'chunk_format_distribution' in row:
                    try:
                        dist = json.loads(row['chunk_format_distribution'])
                        # We don't have full win maps, but we have distribution
                        # record['chunk_wins'] = ... 
                        pass
                    except:
                        pass
                        
                results.append(record)
        return results, True
    except Exception as e:
        print(f"Failed to load cached results: {e}")
        return [], False
