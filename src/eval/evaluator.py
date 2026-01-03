import torch
from tqdm import tqdm

class Evaluator:
    """
    Coordinates the evaluation flow.
    """
    def __init__(self, adapter, metrics_engine, device=None):
        self.adapter = adapter
        self.metrics_engine = metrics_engine
        self.device = device if device else torch.device("cpu")

    def evaluate(self, model, data_loader):
        """
        Runs evaluation on the given model and data loader.
        """
        model.eval()
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating", unit="batch"):
                # Use adapter to prepare batch (e.g. move to device, unpack)
                inputs, targets = self.adapter.prepare_batch(batch)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.adapter.forward(model, (inputs, targets))
                
                # Update metrics
                self.metrics_engine.update(outputs, targets)
        
        return self.metrics_engine.compute()
