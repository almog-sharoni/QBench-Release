import os
import sys
import torch
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Define paths
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
WEIGHT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "runspace/experiments/find_optimal_weight_quant/results")
INPUT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "runspace/experiments/find_optimal_input_quant/results")
OUTPUT_PDF = os.path.join(PROJECT_ROOT, "accuracy_comparison_report.pdf")

def get_unique_layer_types(model_name):
    try:
        if not hasattr(models, model_name):
            return "Model not found in torchvision.models"
        
        model_fn = getattr(models, model_name)
        # Load without weights for speed
        model = model_fn(weights=None)
        
        unique_types = set()
        for module in model.modules():
            layer_type = type(module).__name__
            # Skip basic containers and the model itself if it has a custom name
            if layer_type in ['Sequential', 'ModuleList', 'Bottleneck', 'BasicBlock', 'InvertedResidual', model_name.capitalize()]:
                continue
            unique_types.add(layer_type)
        
        return sorted(list(unique_types))
    except Exception as e:
        return [f"Error loading model: {str(e)}"]

def main():
    # Find common models
    weight_models = set(d for d in os.listdir(WEIGHT_RESULTS_DIR) if os.path.isdir(os.path.join(WEIGHT_RESULTS_DIR, d)))
    input_models = set(d for d in os.listdir(INPUT_RESULTS_DIR) if os.path.isdir(os.path.join(INPUT_RESULTS_DIR, d)))
    common_models = sorted(list(weight_models.intersection(input_models)))

    print(f"Found {len(common_models)} common models: {common_models}")

    # Set DPI for "original resolution" calculation
    DPI = 100 
    # Example Image Size is 1400x800
    IMG_W, IMG_H = 1400, 800
    
    # Define Figure Size (inches)
    # Total width: Images (14) + Side panel (4) = 18 inches
    # Total height: Top margin (1) + Image 1 (8) + Image 2 (8) = 17 inches
    FIG_W, FIG_H = 18, 17

    with PdfPages(OUTPUT_PDF) as pdf:
        for model_name in common_models:
            print(f"Processing {model_name}...")
            
            # Create a large figure to accommodate "original resolution" images and text
            fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
            plt.suptitle(f"Model Architecture and Accuracy: {model_name}", fontsize=28, fontweight='bold', y=0.98)

            # Get unique layer types
            layer_types = get_unique_layer_types(model_name)
            layer_text = "Unique Layer Types:\n\n"
            for l_type in layer_types:
                layer_text += f" • {l_type}\n"

            # Add layer info in a side panel (right side)
            plt.figtext(0.82, 0.92, layer_text, fontsize=14, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='azure', alpha=0.3))

            # Add Images (left side, stacked)
            weight_img_path = os.path.join(WEIGHT_RESULTS_DIR, model_name, "accuracy_comparison.png")
            input_img_path = os.path.join(INPUT_RESULTS_DIR, model_name, "accuracy_comparison.png")

            # Positioning: [left, bottom, width, height] in normalized coordinates (0 to 1)
            # Image width = 14/18 = 0.77, height = 8/17 = 0.47
            
            # Weight image (Top)
            if os.path.exists(weight_img_path):
                ax1 = fig.add_axes([0.05, 0.52, 0.75, 0.42]) # Adjusting to fit properly
                img1 = Image.open(weight_img_path)
                ax1.imshow(img1)
                ax1.set_title("Weight Quantization Accuracy (Original Resolution)", fontsize=18, pad=15)
                ax1.axis('off')
            else:
                print(f"  Missing weight image for {model_name}")

            # Input image (Bottom)
            if os.path.exists(input_img_path):
                ax2 = fig.add_axes([0.05, 0.05, 0.75, 0.42])
                img2 = Image.open(input_img_path)
                ax2.imshow(img2)
                ax2.set_title("Input Quantization Accuracy (Original Resolution)", fontsize=18, pad=15)
                ax2.axis('off')
            else:
                print(f"  Missing input image for {model_name}")

            pdf.savefig(fig, dpi=DPI)
            plt.close(fig)

    print(f"PDF report generated at: {OUTPUT_PDF}")

if __name__ == "__main__":
    main()
