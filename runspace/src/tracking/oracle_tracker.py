
import threading
import torch
import numpy as np

class OracleTracker:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(OracleTracker, cls).__new__(cls)
                    cls._instance.reset()
        return cls._instance
    
    def reset(self):
        self.enabled = False
        self.current_context = None # (run_id, layer_name)
        # Storage: {run_id: {layer_name: {batch_idx: counts_dict}}}
        # counts_dict: {format_name: count}
        self.stats = {}
        self.candidates = [] # Store the order of candidates
        
    def enable(self):
        self.enabled = True
        
    def disable(self):
        self.enabled = False
        
    def set_candidates(self, candidates):
        self.candidates = candidates
        
    def set_context(self, run_id, layer_name):
        self.current_context = (run_id, layer_name)
        
    def clear_context(self):
        self.current_context = None
        
    def log_selection(self, best_indices, batch_idx=0):
        if not self.enabled or self.current_context is None:
            return
            
        run_id, layer_name = self.current_context
            
        # best_indices is a tensor of indices [B, N_chunks]
        # We aggregate counts here to save memory (don't store every chunk choice)
        
        # Move to CPU for counting
        indices = best_indices.detach().cpu().numpy().flatten()
        counts = np.bincount(indices, minlength=len(self.candidates))
        
        # Ensure run_id exists
        if run_id not in self.stats:
            self.stats[run_id] = {}
            
        layer_stats = self.stats[run_id].setdefault(layer_name, {})
        # Accumulate over batches or store per batch? 
        # Plan asked for "heatmap", aggregating over all batches is probably fine for "Format Distribution"
        # provided we handle normalization later.
        
        # Let's simple accumulate total counts per layer for now
        if 'total_counts' not in layer_stats:
            layer_stats['total_counts'] = np.zeros(len(self.candidates), dtype=np.int64)
            
        layer_stats['total_counts'] += counts

    def get_stats(self):
        return self.stats, self.candidates
