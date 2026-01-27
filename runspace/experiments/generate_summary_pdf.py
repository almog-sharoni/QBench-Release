import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

def get_args():
    parser = argparse.ArgumentParser(description="Generate PDF summary for optimal_layer_quant results")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to optimal_layer_quant results")
    parser.add_argument("--pdf_name", type=str, default="optimal_layer_quant_summary.pdf", help="Name of output PDF file")
    return parser.parse_args()

def find_models(root_dir):
    """
    Finds model directories that contain the required plots.
    Returns a list of dicts: {'name': model_name, 'acc_plot': path, 'mse_plot': path}
    """
    models = []
    # Walk directory to find potential model folders
    # Assuming shallow structure: root_dir/model_name or root_dir/category/model_name
    # But based on list_dir, it seems like root_dir/model_name directly.
    
    if not os.path.isdir(root_dir):
        print(f"Error: {root_dir} is not a directory")
        return []

    print(f"Scanning {root_dir} for models...")
    
    subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for model_dir in sorted(subdirs):
        model_name = os.path.basename(model_dir)
        
        # Check for accuracy plot
        acc_plot = os.path.join(model_dir, "accuracy_comparison.png")
        
        # Check for MSE chunk win rate plot
        mse_plot = os.path.join(model_dir, "mse", "chunk_win_rate_mse.png")
        
        if os.path.exists(acc_plot) and os.path.exists(mse_plot):
            print(f"Found valid results for: {model_name}")
            models.append({
                'name': model_name,
                'acc_plot': acc_plot,
                'mse_plot': mse_plot
            })
        else:
            # Verbose skip?
            # print(f"Skipping {model_name}: Missing plots")
            pass
            
    return models

def create_pdf(models, output_path):
    if not models:
        print("No models found to report.")
        return

    print(f"Generating PDF report: {output_path}")
    
    with PdfPages(output_path) as pdf:
        for i, model in enumerate(models):
            print(f"Processing page {i+1}/{len(models)}: {model['name']}")
            
            # Create a figure for the page (A4ish aspect ratio)
            # A4 is approx 8.27 x 11.69 inches
            fig = plt.figure(figsize=(8.27, 11.69))
            
            # Title
            plt.suptitle(f"Model: {model['name']}", fontsize=16, fontweight='bold', y=0.98)
            
            # Top: Accuracy Comparison
            # Axes: [left, bottom, width, height]
            
            # Accuracy Image
            try:
                img_acc = mpimg.imread(model['acc_plot'])
                # Layout: Top half, full width
                ax1 = fig.add_axes([0.0, 0.52, 1.0, 0.44]) 
                ax1.imshow(img_acc, aspect='auto') # aspect='auto' might distort if not careful, keep default aspect?
                # Actually, 'aspect=equal' is default. We should let it fit.
                # To maximize resolution/size, let's reset to allow centering but max size.
                ax1.clear()
                ax1.set_axis_off()
                ax1.imshow(img_acc) # defaults to aspect equal
                ax1.set_title("Accuracy Comparison", fontsize=12)
            except Exception as e:
                print(f"Failed to load acc plot for {model['name']}: {e}")
            
            # Bottom: MSE Chunk Win Rate
            try:
                img_mse = mpimg.imread(model['mse_plot'])
                # Layout: Bottom half, full width
                ax2 = fig.add_axes([0.0, 0.05, 1.0, 0.44])
                ax2.set_axis_off()
                ax2.imshow(img_mse)
                ax2.set_title("MSE Chunk Win Rate", fontsize=12)
            except Exception as e:
                print(f"Failed to load mse plot for {model['name']}: {e}")

            # Add Metadata or Page Number
            fig.text(0.5, 0.02, f"Page {i+1}", ha='center', fontsize=10)
            
            # Save with high DPI
            pdf.savefig(fig, dpi=1000)
            plt.close(fig)
            
    print("PDF generation complete.")

def main():
    args = get_args()
    
    models = find_models(args.output_dir)
    
    if not models:
        print("No models found with required plots.")
        return

    output_path = os.path.join(args.output_dir, args.pdf_name)
    create_pdf(models, output_path)

if __name__ == "__main__":
    main()
