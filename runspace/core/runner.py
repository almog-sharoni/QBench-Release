import os
import sys
import copy
import json
from tqdm import tqdm
import torch
import yaml
import time
import gc
from typing import List, Dict, Any, Optional

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.adapters.adapter_factory import create_adapter
from src.eval.comparator import LayerComparator
from torch.utils.data import DataLoader


class Runner:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._db = None
        self._fm_db = None
        print(f"Runner initialized on device: {self.device}")

        # Inference throughput knobs: enable cuDNN autotune for fixed-shape convs,
        # and let matmul use TF32 where it's available (SuperGlue attention is
        # matmul-heavy). These only affect FP32 reference passes; quantized paths
        # already control their own precision.
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')

        # Aggressive cleanup on init
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _to_device(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        if isinstance(x, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in x.items()}
        return x

    @staticmethod
    def _running_accuracy(metrics_engine):
        """Returns running top-1/top-5 accuracy from a MetricsEngine instance."""
        total = getattr(metrics_engine, 'total', 0)
        if total == 0:
            return 0.0, 0.0
        acc1 = 100.0 * getattr(metrics_engine, 'correct_1', 0) / total
        acc5 = 100.0 * getattr(metrics_engine, 'correct_5', 0) / total
        return acc1, acc5

    @staticmethod
    def _is_file_backed_weights(weights_spec) -> bool:
        return (
            isinstance(weights_spec, str)
            and weights_spec not in ("", "DEFAULT", "default", "None", "none")
            and os.path.isfile(weights_spec)
        )

    @staticmethod
    def _expects_quantized_layers(config: Dict[str, Any]) -> bool:
        """
        Return True when adapter config indicates wrapper-based quantized layers
        are expected in the built model graph.
        """
        adapter_cfg = config.get('adapter', {}) or {}
        quant_cfg = config.get('quantization', {}) or {}

        quantized_ops = adapter_cfg.get('quantized_ops', ['all'])
        if quantized_ops is None:
            quantized_ops = []
        if not isinstance(quantized_ops, list):
            quantized_ops = [quantized_ops]
        has_qops = any(bool(str(op).strip()) for op in quantized_ops)

        layer_cfg = quant_cfg.get('layers', adapter_cfg.get('layers'))
        has_layer_overrides = isinstance(layer_cfg, dict) and len(layer_cfg) > 0
        quantize_first_layer = bool(adapter_cfg.get('quantize_first_layer', False))

        explicit_build_quantized = adapter_cfg.get('build_quantized')
        if explicit_build_quantized is None:
            build_quantized = bool(quantize_first_layer or has_qops or has_layer_overrides)
        else:
            build_quantized = bool(explicit_build_quantized)

        return build_quantized

    @staticmethod
    def _looks_like_weight_file_path(weights_spec) -> bool:
        if not isinstance(weights_spec, str):
            return False
        token = weights_spec.strip()
        if token.lower() in ("", "default", "none", "false", "0"):
            return False
        # Path-like hints or common checkpoint suffixes.
        return (
            os.path.sep in token
            or token.endswith(".pt")
            or token.endswith(".pth")
            or token.endswith(".bin")
            or token.endswith(".ckpt")
        )

    @staticmethod
    def _shutdown_dataloader_workers(data_loader):
        """
        Best-effort explicit DataLoader worker shutdown.
        Helps avoid lingering worker processes between sequential runs.
        """
        if data_loader is None:
            return
        try:
            iterator = getattr(data_loader, "_iterator", None)
            if iterator is not None:
                if hasattr(iterator, "_shutdown_workers"):
                    try:
                        iterator._shutdown_workers()
                    except Exception:
                        pass
                # Fallback: forcibly terminate any surviving worker processes
                # (handles the case where __init__ failed before _workers_status was set).
                for w in getattr(iterator, "_workers", []):
                    try:
                        if w.is_alive():
                            w.terminate()
                            w.join(timeout=2)
                    except Exception:
                        pass
                data_loader._iterator = None
        except Exception:
            pass

    @staticmethod
    def _requires_runtime_weight_calibration(config: Dict[str, Any]) -> bool:
        """
        Return True when this config relies on runtime weight quantization
        (quantized wrappers + weight_quantization=True), which requires
        calibration before snapshotting a materialized weight file.
        """
        adapter_cfg = config.get('adapter', {}) or {}
        quant_cfg = config.get('quantization', {}) or {}

        quantized_ops = adapter_cfg.get('quantized_ops', ['all'])
        if quantized_ops is None:
            quantized_ops = []
        if not isinstance(quantized_ops, list):
            quantized_ops = [quantized_ops]
        has_qops = any(bool(str(op).strip()) for op in quantized_ops)

        layer_cfg = quant_cfg.get('layers', adapter_cfg.get('layers'))
        has_layer_overrides = isinstance(layer_cfg, dict) and len(layer_cfg) > 0
        quantize_first_layer = bool(adapter_cfg.get('quantize_first_layer', False))
        default_build_quantized = bool(quantize_first_layer or has_qops or has_layer_overrides)

        explicit_build_quantized = adapter_cfg.get('build_quantized')
        if explicit_build_quantized is None:
            build_quantized = default_build_quantized
        else:
            build_quantized = bool(explicit_build_quantized)

        weight_quantization = bool(adapter_cfg.get('weight_quantization', True))
        return bool(build_quantized and weight_quantization)

    def _resolve_skip_calibration_for_build(
        self,
        config: Dict[str, Any],
        source_is_file: bool
    ) -> bool:
        """
        Decide whether adapter construction should skip calibration.
        """
        if source_is_file:
            return True

        adapter_cfg = config.get('adapter', {}) or {}
        explicit_skip = adapter_cfg.get('skip_calibration')
        runtime_weight_calibration = self._requires_runtime_weight_calibration(config)

        if runtime_weight_calibration:
            if explicit_skip is True:
                print(
                    "Warning: adapter.skip_calibration=true ignored because "
                    "runtime weight quantization is enabled."
                )
            return False

        if explicit_skip is None:
            return True
        return bool(explicit_skip)

    def _get_db(self, db_path: Optional[str] = None):
        from src.database.handler import RunDatabase
        if self._db is None:
            self._db = RunDatabase(db_path=db_path)
        elif db_path and os.path.abspath(self._db.db_path) != os.path.abspath(db_path):
            self._db = RunDatabase(db_path=db_path)
        return self._db

    @staticmethod
    def _fm_db_path(db_path: Optional[str]) -> str:
        """Derive the FM database path alongside the classification database."""
        from src.database.handler import RunDatabase
        base = db_path if db_path else RunDatabase._default_db_path()
        return os.path.join(os.path.dirname(base), 'fm_runs.db')

    def _get_fm_db(self, db_path: Optional[str] = None):
        from src.database.handler import RunDatabase
        fm_path = self._fm_db_path(db_path)
        if self._fm_db is None:
            self._fm_db = RunDatabase(db_path=fm_path)
        elif os.path.abspath(self._fm_db.db_path) != os.path.abspath(fm_path):
            self._fm_db = RunDatabase(db_path=fm_path)
        return self._fm_db

    @staticmethod
    def _to_json_string(value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return json.dumps(value, default=str)

    @classmethod
    def _build_input_map_json(cls, layer_stats: Optional[Dict[str, Any]]):
        if not layer_stats:
            return None
        result = {}
        for layer_name, stats in layer_stats.items():
            if not isinstance(stats, dict):
                continue

            counts = {}
            raw_counts = stats.get('format_counts', {})
            if isinstance(raw_counts, dict):
                for fmt, cnt in raw_counts.items():
                    try:
                        cnt_i = int(cnt)
                    except Exception:
                        continue
                    if cnt_i > 0:
                        counts[str(fmt)] = cnt_i

            if not counts:
                fmt_spec = stats.get('format')
                counts = cls._count_formats_from_spec(fmt_spec)
            if not counts:
                continue

            total_chunks = stats.get('total_chunks')
            try:
                total_chunks = int(total_chunks) if total_chunks is not None else int(sum(counts.values()))
            except Exception:
                total_chunks = int(sum(counts.values()))
            if total_chunks <= 0:
                total_chunks = int(sum(counts.values()))

            dominant_format = stats.get('dominant_format')
            if dominant_format is None:
                dominant_format = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]

            fmt_spec = stats.get('format', dominant_format)
            result[str(layer_name)] = {
                'format': fmt_spec,
                'type': str(stats.get('type', 'unknown')),
                'format_counts': counts,
                'total_chunks': total_chunks,
                'dominant_format': str(dominant_format),
            }

        enriched = cls._enrich_quant_map(result)
        if enriched is None:
            return None
        return json.dumps(enriched, default=str)

    @staticmethod
    def _safe_json_load(raw_value):
        if raw_value is None:
            return None
        if isinstance(raw_value, (dict, list)):
            return raw_value
        if isinstance(raw_value, str):
            try:
                return json.loads(raw_value)
            except Exception:
                return None
        return None

    @staticmethod
    def _count_formats_from_spec(fmt_spec):
        counts = {}
        if isinstance(fmt_spec, list):
            for fmt in fmt_spec:
                key = str(fmt)
                counts[key] = counts.get(key, 0) + 1
        elif fmt_spec is not None:
            key = str(fmt_spec)
            counts[key] = counts.get(key, 0) + 1
        return counts

    @classmethod
    def _enrich_quant_map(cls, quant_map):
        """
        Normalize quant-map payloads to a schema that supports dashboard win-rate views.
        """
        if not isinstance(quant_map, dict) or not quant_map:
            return None

        enriched = {}
        for layer, value in quant_map.items():
            entry = value if isinstance(value, dict) else {}
            fmt_spec = entry.get('format') if isinstance(value, dict) else value
            if fmt_spec is None and isinstance(entry, dict):
                fmt_spec = entry.get('chunk_formats')
            if fmt_spec is None:
                continue

            counts = {}
            raw_counts = entry.get('format_counts') if isinstance(entry, dict) else None
            if isinstance(raw_counts, dict):
                for fmt, cnt in raw_counts.items():
                    try:
                        counts[str(fmt)] = int(cnt)
                    except Exception:
                        continue
            if not counts:
                counts = cls._count_formats_from_spec(fmt_spec)
            if not counts:
                continue

            try:
                total_chunks = int(entry.get('total_chunks')) if isinstance(entry, dict) and entry.get('total_chunks') is not None else int(sum(counts.values()))
            except Exception:
                total_chunks = int(sum(counts.values()))
            if total_chunks <= 0:
                total_chunks = int(sum(counts.values()))

            dominant_format = entry.get('dominant_format') if isinstance(entry, dict) else None
            if dominant_format is None:
                dominant_format = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
            else:
                dominant_format = str(dominant_format)

            normalized = dict(entry) if isinstance(entry, dict) else {}
            normalized['format'] = fmt_spec
            normalized['format_counts'] = counts
            normalized['total_chunks'] = total_chunks
            normalized['dominant_format'] = dominant_format
            enriched[str(layer)] = normalized

        return enriched if enriched else None

    @classmethod
    def _extract_quant_map_from_checkpoint_obj(cls, state_obj):
        if not isinstance(state_obj, dict):
            return None

        candidates = []
        for key in ('quant_map', 'quant_map_json', '_qbench_quant_map', 'qbench_quant_map'):
            if key in state_obj:
                candidates.append(state_obj.get(key))

        for meta_key in ('_qbench_meta', 'qbench_meta', 'metadata'):
            meta_val = state_obj.get(meta_key)
            if isinstance(meta_val, dict):
                for nested_key in ('quant_map', 'quant_map_json'):
                    if nested_key in meta_val:
                        candidates.append(meta_val.get(nested_key))

        for candidate in candidates:
            parsed = cls._safe_json_load(candidate)
            if parsed is None and isinstance(candidate, dict):
                parsed = candidate
            enriched = cls._enrich_quant_map(parsed)
            if enriched is not None:
                return enriched
        return None

    @classmethod
    def _load_quant_map_from_weight_file(cls, weight_file_path: str):
        if not isinstance(weight_file_path, str) or not os.path.isfile(weight_file_path):
            return None
        try:
            state_obj = torch.load(weight_file_path, map_location='cpu')
        except Exception:
            return None
        return cls._extract_quant_map_from_checkpoint_obj(state_obj)

    @classmethod
    def _load_quant_map_from_sidecar(cls, weight_file_path: str):
        if not isinstance(weight_file_path, str):
            return None

        base_dir = os.path.dirname(weight_file_path)
        base_name = os.path.basename(weight_file_path)
        stem, _ = os.path.splitext(base_name)

        candidates = []
        if 'chunk' in stem:
            candidates.append(os.path.join(base_dir, "quantization_map_chunk.json"))
        if 'layer' in stem:
            candidates.append(os.path.join(base_dir, "quantization_map_layer.json"))
        candidates.extend([
            os.path.join(base_dir, "quantization_map.json"),
            os.path.join(base_dir, f"{stem}.quant_map.json"),
            os.path.join(base_dir, f"{stem}_quant_map.json"),
        ])

        for path in candidates:
            if not os.path.isfile(path):
                continue
            try:
                with open(path, 'r') as f:
                    raw_map = json.load(f)
            except Exception:
                continue
            enriched = cls._enrich_quant_map(raw_map)
            if enriched is not None:
                return enriched
        return None

    @classmethod
    def _build_quant_map_from_layer_config(cls, config: Dict[str, Any]):
        q_cfg = config.get('quantization', {})
        layer_cfgs = q_cfg.get('layers')
        if not isinstance(layer_cfgs, dict) or not layer_cfgs:
            return None

        quant_map = {}
        for layer, layer_cfg in layer_cfgs.items():
            fmt_spec = None
            if isinstance(layer_cfg, dict):
                if isinstance(layer_cfg.get('chunk_formats'), list):
                    fmt_spec = layer_cfg.get('chunk_formats')
                elif layer_cfg.get('format') is not None:
                    fmt_spec = layer_cfg.get('format')
            elif isinstance(layer_cfg, (str, list)):
                fmt_spec = layer_cfg

            if fmt_spec is not None:
                quant_map[str(layer)] = fmt_spec

        return cls._enrich_quant_map(quant_map)

    def _resolve_quant_map_for_run(self, config: Dict[str, Any], materialized_weight_path: Optional[str] = None):
        exp_cfg = config.get('experiment', {})

        explicit_quant_map = self._safe_json_load(exp_cfg.get('quant_map_json'))
        enriched_explicit = self._enrich_quant_map(explicit_quant_map)
        if enriched_explicit is not None:
            return enriched_explicit

        cfg_quant_map = self._build_quant_map_from_layer_config(config)
        if cfg_quant_map is not None:
            return cfg_quant_map

        model_weights = config.get('model', {}).get('weights')
        paths_to_probe = []
        if isinstance(model_weights, str) and os.path.isfile(model_weights):
            paths_to_probe.append(model_weights)
        if isinstance(materialized_weight_path, str) and os.path.isfile(materialized_weight_path):
            if materialized_weight_path not in paths_to_probe:
                paths_to_probe.append(materialized_weight_path)

        for path in paths_to_probe:
            from_ckpt = self._load_quant_map_from_weight_file(path)
            if from_ckpt is not None:
                return from_ckpt
            from_sidecar = self._load_quant_map_from_sidecar(path)
            if from_sidecar is not None:
                return from_sidecar

        return None

    def run_exists_for_config(self, config: Dict[str, Any], db_path: Optional[str] = None) -> bool:
        exp_cfg = config.get('experiment', {})
        model_name = config.get('model', {}).get('name', 'unknown')
        experiment_type = exp_cfg.get('type') or exp_cfg.get('name')
        weight_dt = exp_cfg.get('weight_dt')
        activation_dt = exp_cfg.get('activation_dt')
        if not experiment_type:
            return False
        is_fm = config.get('adapter', {}).get('type') == 'feature_matching'
        db = self._get_fm_db(db_path=db_path) if is_fm else self._get_db(db_path=db_path)
        if is_fm:
            return db.fm_run_exists(
                model_name=model_name,
                experiment_type=experiment_type,
                weight_dt=weight_dt,
                activation_dt=activation_dt,
                status="SUCCESS",
            )
        return db.run_exists(
            model_name=model_name,
            experiment_type=experiment_type,
            weight_dt=weight_dt,
            activation_dt=activation_dt,
            status="SUCCESS",
        )

    def _load_state_dict_resilient(self, model, state_dict):
        """
        Robust loading for quantized models where some buffers can be registered as None.
        """
        for fqn, tensor in state_dict.items():
            parts = fqn.split('.')
            mod = model
            try:
                for part in parts[:-1]:
                    mod = getattr(mod, part)
                buf_name = parts[-1]
                if buf_name in getattr(mod, '_buffers', {}) and mod._buffers[buf_name] is None:
                    mod.register_buffer(buf_name, torch.empty_like(tensor))
            except AttributeError:
                continue

        try:
            model.load_state_dict(state_dict, strict=True, assign=True)
        except TypeError:
            # Older PyTorch fallback.
            for module in model.modules():
                for name, buf in list(getattr(module, '_buffers', {}).items()):
                    if buf is not None and not buf.is_contiguous():
                        module._buffers[name] = buf.contiguous()
            model.load_state_dict(state_dict, strict=False)

    @staticmethod
    def _extract_raw_state_dict(state_obj):
        """Extract a raw tensor state_dict from common checkpoint container layouts."""
        if not isinstance(state_obj, dict):
            return state_obj
        if 'state_dict' in state_obj and isinstance(state_obj['state_dict'], dict):
            return state_obj['state_dict']
        if 'model' in state_obj and isinstance(state_obj['model'], dict):
            return state_obj['model']
        return state_obj

    @staticmethod
    def _state_dict_match_stats(source_state, target_state, atol: float = 0.0, key_filter=None):
        """
        Compare source_state tensors against target_state tensors (same key+shape only).
        Returns aggregate stats for coverage and mismatches.
        """
        stats = {
            'source_tensor_keys': 0,
            'comparable_tensors': 0,
            'mismatched_tensors': 0,
            'max_abs_diff': 0.0,
            'sample_mismatches': [],
        }

        for key, src in source_state.items():
            if key_filter is not None and not key_filter(key):
                continue
            if not torch.is_tensor(src):
                continue
            stats['source_tensor_keys'] += 1
            tgt = target_state.get(key)
            if not torch.is_tensor(tgt):
                continue
            if src.shape != tgt.shape:
                continue

            stats['comparable_tensors'] += 1
            src_cpu = src.detach().cpu()
            tgt_cpu = tgt.detach().cpu()
            # Compare in float for numeric tensors; exact for others.
            if torch.is_floating_point(src_cpu) or torch.is_floating_point(tgt_cpu):
                diff = (src_cpu.float() - tgt_cpu.float()).abs()
                max_diff = float(diff.max().item()) if diff.numel() > 0 else 0.0
                if max_diff > stats['max_abs_diff']:
                    stats['max_abs_diff'] = max_diff
                same = torch.allclose(
                    src_cpu.float(),
                    tgt_cpu.float(),
                    atol=atol,
                    rtol=0.0,
                    equal_nan=True,
                )
            else:
                same = torch.equal(src_cpu, tgt_cpu)
                max_diff = 0.0 if same else float('inf')

            if not same:
                stats['mismatched_tensors'] += 1
                if len(stats['sample_mismatches']) < 8:
                    stats['sample_mismatches'].append((key, max_diff))

        return stats

    @staticmethod
    def _is_parameter_like_state_key(key: str) -> bool:
        # Focus on learned tensors to avoid false negatives from runtime buffers.
        return (
            key.endswith('.weight')
            or key.endswith('.bias')
            or key.endswith('in_proj_weight')
            or key.endswith('in_proj_bias')
        )

    def _assert_state_dict_loaded(
        self,
        source_state,
        target_state,
        context: str,
        min_coverage: float = 0.95,
        atol: float = 0.0,
        key_filter=None,
    ):
        """
        Ensure source_state is actually loaded into target_state with high coverage and low mismatch.
        Raises RuntimeError on suspicious load.
        """
        stats = self._state_dict_match_stats(
            source_state,
            target_state,
            atol=atol,
            key_filter=key_filter,
        )
        if stats['source_tensor_keys'] == 0:
            raise RuntimeError(
                f"{context}: state_dict validation found no comparable tensor keys "
                "(after filtering)."
            )
        src_keys = max(1, stats['source_tensor_keys'])
        coverage = stats['comparable_tensors'] / src_keys
        mismatches = stats['mismatched_tensors']

        if coverage < min_coverage or mismatches > 0:
            mismatch_preview = ", ".join(
                [f"{k}:{d:.3e}" if d != float('inf') else f"{k}:inf" for k, d in stats['sample_mismatches']]
            )
            raise RuntimeError(
                f"{context}: state_dict load validation failed "
                f"(coverage={coverage:.3f}, comparable={stats['comparable_tensors']}/{stats['source_tensor_keys']}, "
                f"mismatched={mismatches}, max_abs_diff={stats['max_abs_diff']:.3e}, "
                f"samples=[{mismatch_preview}])"
            )

        print(
            f"{context}: state_dict validated "
            f"(coverage={coverage:.3f}, comparable={stats['comparable_tensors']}/{stats['source_tensor_keys']}, "
            f"mismatched={mismatches}, max_abs_diff={stats['max_abs_diff']:.3e})"
        )

    def materialize_weight_file(
        self,
        config: Dict[str, Any],
        weight_file_path: str,
        force_rebuild: bool = False
    ) -> str:
        """
        Builds a model from config, snapshots its weights to disk, and returns the path.
        """
        os.makedirs(os.path.dirname(weight_file_path), exist_ok=True)
        if not force_rebuild and os.path.exists(weight_file_path):
            return os.path.abspath(weight_file_path)

        adapter = None
        try:
            source_weights = config.get('model', {}).get('weights')
            source_quant_map = None
            source_is_file = self._is_file_backed_weights(source_weights)

            if source_is_file:
                src_obj = torch.load(source_weights, map_location='cpu')
                source_quant_map = self._extract_quant_map_from_checkpoint_obj(src_obj)
                if source_quant_map is None:
                    source_quant_map = self._load_quant_map_from_sidecar(source_weights)
                src_state = self._extract_raw_state_dict(src_obj)
                if not isinstance(src_state, dict):
                    raise RuntimeError(
                        f"materialize_weight_file(source={source_weights}): "
                        "checkpoint did not contain a valid state_dict mapping."
                    )
                print(
                    "materialize_weight_file: source is a checkpoint file; "
                    "canonicalizing state_dict directly (no model rebuild in this step)."
                )
                state_dict = {}
                for key, value in src_state.items():
                    if torch.is_tensor(value):
                        state_dict[key] = value.detach().clone().contiguous().cpu()
            else:
                adapter_cfg = copy.deepcopy(config)
                adapter_cfg.setdefault('adapter', {})
                adapter_cfg['adapter']['skip_calibration'] = self._resolve_skip_calibration_for_build(
                    config=adapter_cfg,
                    source_is_file=False,
                )
                adapter = create_adapter(adapter_cfg)
                state_dict = {
                    key: value.detach().clone().contiguous().cpu()
                    for key, value in adapter.model.state_dict().items()
                }

            if source_quant_map is not None:
                torch.save(
                    {
                        'state_dict': state_dict,
                        '_qbench_meta': {'quant_map': source_quant_map},
                    },
                    weight_file_path
                )
            else:
                torch.save(state_dict, weight_file_path)
            return os.path.abspath(weight_file_path)
        finally:
            if adapter is not None:
                del adapter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load_model_from_weight_file(
        self,
        config: Dict[str, Any],
        weight_file_path: str,
        skip_calibration: bool = True
    ):
        """
        Rebuild model from config and load weights from a saved state_dict file.
        """
        cfg = copy.deepcopy(config)
        cfg.setdefault('adapter', {})
        if skip_calibration:
            cfg['adapter']['skip_calibration'] = True

        # Canonical source of weights for this method is `weight_file_path`.
        # Always build architecture skeleton (weights=None), then load state_dict.
        source_weights = cfg.get('model', {}).get('weights')
        cfg.setdefault('model', {})
        cfg['model']['weights'] = None
        if source_weights is not None:
            print(
                "load_model_from_weight_file: building skeleton model with "
                "weights=None and loading from materialized checkpoint."
            )

        adapter = create_adapter(cfg)
        # Preserve original weights spec so build_reference_model can reload
        # pretrained weights via the original source (e.g. torchvision enum).
        adapter._reference_weights_spec = source_weights
        state_obj = torch.load(weight_file_path, map_location='cpu')
        state_dict = self._extract_raw_state_dict(state_obj)
        self._load_state_dict_resilient(adapter.model, state_dict)
        self._assert_state_dict_loaded(
            source_state=state_dict,
            target_state=adapter.model.state_dict(),
            context=f"load_model_from_weight_file(file={weight_file_path})",
            min_coverage=0.98,
            atol=1e-6,
            key_filter=self._is_parameter_like_state_key,
        )
        adapter.model.to(self.device)
        adapter.model.eval()
        return adapter.model, adapter

    @staticmethod
    def _adapter_manages_own_weights(config: dict) -> bool:
        """True for adapters that load weights internally (e.g. feature_matching via PipelineRegistry)."""
        return config.get('adapter', {}).get('type') == 'feature_matching'

    def _build_reference_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates a version of the config for the FP32 reference model that matches 
        the structure of the quantized model (same ops, folding, etc) but with 
        quantization disabled.
        """
        ref_cfg = copy.deepcopy(config)
        
        # 1. Disable all quantization flags in adapter
        adapter_cfg = ref_cfg.setdefault('adapter', {})
        adapter_cfg['input_quantization'] = False
        adapter_cfg['weight_quantization'] = False
        
        # 2. Set global format to fp32
        quant_cfg = ref_cfg.setdefault('quantization', {})
        quant_cfg['format'] = 'fp32'
        quant_cfg['input_format'] = 'fp32'
        
        # 3. Handle layer-specific overrides
        if 'layers' in quant_cfg and isinstance(quant_cfg['layers'], dict):
            for name, layer_cfg in quant_cfg['layers'].items():
                if isinstance(layer_cfg, dict):
                    layer_cfg['format'] = 'fp32'
                    layer_cfg['input_format'] = 'fp32'
        
        # 4. Disable dynamic quantization for reference
        eval_cfg = ref_cfg.setdefault('evaluation', {})
        eval_cfg['dynamic_input_quant'] = None
        eval_cfg['input_quant'] = None
        
        return ref_cfg


    def prepare_model_with_materialized_weights(
        self,
        config: Dict[str, Any],
        output_dir: str
    ):
        """
        Enforces "weights-from-file" execution:
        1) materialize state_dict to disk
        2) rebuild model and load from that file
        """
        if self._adapter_manages_own_weights(config):
            adapter = create_adapter(config)
            adapter.model.to(self.device)
            adapter.model.eval()
            return adapter.model, adapter, None

        exp_cfg = config.get('experiment', {})
        materialize_cfg = exp_cfg.get('materialize_weights', {})
        force_rebuild = bool(materialize_cfg.get('force_rebuild', False))
        source_weights = config.get('model', {}).get('weights')
        if self._looks_like_weight_file_path(source_weights) and not os.path.isfile(str(source_weights)):
            raise FileNotFoundError(
                f"model.weights points to a checkpoint-like path that does not exist: {source_weights}"
            )
        if isinstance(source_weights, str) and os.path.isfile(source_weights):
            # When source weights come from a file (e.g. generated quantized weights),
            # never trust stale materialization caches under output_dir.
            if not force_rebuild:
                print(
                    "Info: model.weights points to a file; forcing materialized "
                    "weights rebuild to avoid stale cache reuse."
                )
            force_rebuild = True
        if materialize_cfg.get('enabled', True) is False:
            print(
                "Warning: experiment.materialize_weights.enabled=false is ignored; "
                "Runner always enforces file-backed weights."
            )

        weights_dir = os.path.join(output_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        weight_filename = materialize_cfg.get('file_name', "materialized_weights.pt")
        weight_file_path = os.path.join(weights_dir, weight_filename)

        source_is_file = self._is_file_backed_weights(source_weights)
        runtime_weight_calibration = (
            (not source_is_file)
            and self._requires_runtime_weight_calibration(config)
        )
        if runtime_weight_calibration and not force_rebuild:
            print(
                "Info: runtime weight quantization requested; forcing materialized "
                "weights rebuild to avoid stale FP32 cache reuse."
            )
            force_rebuild = True
        if source_is_file:
            try:
                sz_mb = os.path.getsize(source_weights) / (1024 * 1024)
                print(f"prepare_model_with_materialized_weights: source checkpoint={source_weights} ({sz_mb:.1f} MB)")
            except Exception:
                print(f"prepare_model_with_materialized_weights: source checkpoint={source_weights}")

        # One-build path for non-file sources (e.g., torchvision DEFAULT):
        # build once -> save weight file -> reload from saved file into same model.
        # This avoids back-to-back architecture construction in a single run.
        if not source_is_file:
            if not force_rebuild and os.path.exists(weight_file_path):
                load_cfg = copy.deepcopy(config)
                load_cfg.pop('experiment', None)
                load_cfg.pop('meta', None)
                load_cfg.pop('debug', None)
                model, adapter = self.load_model_from_weight_file(
                    config=load_cfg,
                    weight_file_path=weight_file_path,
                    skip_calibration=True,
                )
                return model, adapter, os.path.abspath(weight_file_path)

            adapter = None
            try:
                build_cfg = copy.deepcopy(config)
                build_cfg.pop('experiment', None)
                build_cfg.pop('meta', None)
                build_cfg.pop('debug', None)
                build_cfg.setdefault('adapter', {})
                build_cfg['adapter']['skip_calibration'] = self._resolve_skip_calibration_for_build(
                    config=build_cfg,
                    source_is_file=False,
                )

                adapter = create_adapter(build_cfg)
                state_dict = {
                    key: value.detach().clone().contiguous().cpu()
                    for key, value in adapter.model.state_dict().items()
                }
                torch.save(state_dict, weight_file_path)

                # Enforce file-backed loading semantics on the same model instance.
                state_obj = torch.load(weight_file_path, map_location='cpu')
                loaded_state = self._extract_raw_state_dict(state_obj)
                self._load_state_dict_resilient(adapter.model, loaded_state)
                self._assert_state_dict_loaded(
                    source_state=loaded_state,
                    target_state=adapter.model.state_dict(),
                    context=f"prepare_model_with_materialized_weights(file={weight_file_path})",
                    min_coverage=0.98,
                    atol=1e-6,
                    key_filter=self._is_parameter_like_state_key,
                )

                adapter.model.to(self.device)
                adapter.model.eval()
                return adapter.model, adapter, os.path.abspath(weight_file_path)
            except Exception:
                if adapter is not None:
                    del adapter
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise

        snapshot_cfg = copy.deepcopy(config)
        snapshot_cfg.pop('experiment', None)
        snapshot_cfg.pop('meta', None)
        snapshot_cfg.pop('debug', None)

        weight_file_path = self.materialize_weight_file(
            config=snapshot_cfg,
            weight_file_path=weight_file_path,
            force_rebuild=force_rebuild,
        )

        load_cfg = copy.deepcopy(config)
        load_cfg.pop('experiment', None)
        load_cfg.pop('meta', None)
        load_cfg.pop('debug', None)
        model, adapter = self.load_model_from_weight_file(
            config=load_cfg,
            weight_file_path=weight_file_path,
            skip_calibration=True,
        )

        return model, adapter, weight_file_path

    def _resolve_ref_metrics(self, db, model_name: str, exp_cfg: Dict[str, Any], result: Dict[str, Any]):
        ref_acc1 = exp_cfg.get('ref_acc1', result.get('ref_acc1'))
        ref_acc5 = exp_cfg.get('ref_acc5', result.get('ref_acc5'))
        ref_certainty = exp_cfg.get('ref_certainty', result.get('ref_certainty'))

        has_ref = all(v is not None and float(v) != 0.0 for v in [ref_acc1, ref_acc5])
        if not has_ref and exp_cfg.get('resolve_ref_from_db', True):
            ref = db.get_reference_metrics(model_name)
            if ref is not None:
                ref_acc1, ref_acc5, ref_certainty = ref

        return (
            float(ref_acc1 or 0.0),
            float(ref_acc5 or 0.0),
            float(ref_certainty or 0.0),
        )

    def log_experiment_result(
        self,
        config: Dict[str, Any],
        result: Dict[str, Any],
        db_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Central logging entrypoint: all experiments should log via this method.
        """
        exp_cfg = config.get('experiment', {})
        if exp_cfg.get('log_to_db', True) is False:
            return None

        is_fm = config.get('adapter', {}).get('type') == 'feature_matching'
        db = self._get_fm_db(db_path=db_path) if is_fm else self._get_db(db_path=db_path)
        model_name = result.get('model_name', config.get('model', {}).get('name', 'unknown'))
        experiment_type = exp_cfg.get('type') or exp_cfg.get('name') or "runner_eval"
        weight_dt = exp_cfg.get('weight_dt') or result.get('quant_format') or config.get('quantization', {}).get('format') or "fp32"

        activation_dt = exp_cfg.get('activation_dt')
        if activation_dt is None:
            input_quant = result.get('input_quant') or result.get('dynamic_input_quant')
            if isinstance(input_quant, dict) and input_quant.get('mode') == 'dynamic':
                activation_dt = f"dyn_input_{input_quant.get('metric', 'mse')}"
            elif isinstance(input_quant, dict) and input_quant.get('mode') in ('uniform', 'input_only'):
                activation_dt = input_quant.get('format', 'fp32')
            else:
                activation_dt = config.get('quantization', {}).get('input_format', 'fp32')

        if is_fm:
            def _fm_val(key):
                v = result.get(key)
                return float(v) if v else None

            db.log_fm_run(
                model_name=model_name,
                weight_dt=str(weight_dt),
                activation_dt=str(activation_dt),
                experiment_type=experiment_type,
                status=result.get('status', 'SUCCESS'),
                fm_num_keypoints=_fm_val('fm_num_keypoints'),
                fm_mean_score=_fm_val('fm_mean_score'),
                fm_desc_norm=_fm_val('fm_desc_norm'),
                fm_repeatability=_fm_val('fm_repeatability'),
                matching_precision=_fm_val('matching_precision'),
                matching_score=_fm_val('matching_score'),
                mean_num_matches=_fm_val('mean_num_matches'),
                pose_auc_5=_fm_val('pose_auc_5'),
                pose_auc_10=_fm_val('pose_auc_10'),
                pose_auc_20=_fm_val('pose_auc_20'),
                ref_matching_precision=_fm_val('ref_matching_precision'),
                ref_matching_score=_fm_val('ref_matching_score'),
                ref_mean_num_matches=_fm_val('ref_mean_num_matches'),
                ref_pose_auc_5=_fm_val('ref_pose_auc_5'),
                ref_pose_auc_10=_fm_val('ref_pose_auc_10'),
                ref_pose_auc_20=_fm_val('ref_pose_auc_20'),
                config_json=self._to_json_string(exp_cfg.get('config_json') or config),
            )
            return None

        is_input_only_exp = (
            str(experiment_type).strip().lower().startswith('input_quant')
            and str(weight_dt).strip().lower() == 'fp32'
        )

        ref_acc1, ref_acc5, ref_certainty = self._resolve_ref_metrics(
            db=db,
            model_name=model_name,
            exp_cfg=exp_cfg,
            result=result,
        )

        metrics_cfg = exp_cfg.get('metrics', {})
        input_quant = result.get('input_quant') or result.get('dynamic_input_quant') or {}

        mse = metrics_cfg.get('mse')
        l1 = metrics_cfg.get('l1')
        if mse is None:
            mse = input_quant.get('norm_mse', result.get('dyn_norm_mse'))
        if l1 is None:
            l1 = input_quant.get('norm_l1', result.get('dyn_norm_l1'))

        quant_map_json = None
        if not is_input_only_exp:
            quant_map_json = self._to_json_string(exp_cfg.get('quant_map_json'))
            if quant_map_json is None and result.get('weight_quant_map') is not None:
                quant_map_json = self._to_json_string(result.get('weight_quant_map'))
            if quant_map_json is None:
                inferred_map = self._resolve_quant_map_for_run(
                    config=config,
                    materialized_weight_path=result.get('materialized_weight_path')
                )
                if inferred_map is not None:
                    quant_map_json = self._to_json_string(inferred_map)
        input_map_json = self._to_json_string(exp_cfg.get('input_map_json'))
        if input_map_json is None and isinstance(input_quant.get('layer_stats'), dict):
            input_map_json = self._build_input_map_json(input_quant.get('layer_stats'))

        payload = {
            'model_name': model_name,
            'weight_dt': str(weight_dt),
            'activation_dt': str(activation_dt),
            'acc1': float(result.get('acc1', 0.0) or 0.0),
            'acc5': float(result.get('acc5', 0.0) or 0.0),
            'ref_acc1': ref_acc1,
            'ref_acc5': ref_acc5,
            'ref_certainty': ref_certainty,
            'experiment_type': experiment_type,
            'status': result.get('status', 'SUCCESS'),
            'mse': float(mse) if mse is not None else None,
            'l1': float(l1) if l1 is not None else None,
            'certainty': float(metrics_cfg.get('certainty', result.get('certainty', 0.0)) or 0.0),
            'quant_map_json': quant_map_json,
            'input_map_json': input_map_json,
            'config_json': self._to_json_string(exp_cfg.get('config_json') or config),
        }
        db.log_run(**payload)
        return payload

    def run_single_logged(
        self,
        config: Dict[str, Any],
        output_root: str = "runspace/outputs",
        db_path: Optional[str] = None
    ) -> Dict[str, Any]:
        exp_cfg = config.get('experiment', {})
        if exp_cfg.get('skip_if_exists', False) and not exp_cfg.get('force_rerun', False):
            if self.run_exists_for_config(config, db_path=db_path):
                model_name = config.get('model', {}).get('name', 'unknown')
                print(f"Skipping {model_name} run: already exists in DB.")
                return {
                    'model_name': model_name,
                    'output_name': config.get('output_name', ''),
                    'status': 'SKIPPED_DB',
                    'acc1': 0.0,
                    'acc5': 0.0,
                    'certainty': 0.0,
                }

        result = self.run_single(config, output_root=output_root)
        self.log_experiment_result(config, result, db_path=db_path)
        return result

    def run_batch_logged(
        self,
        configs: List[Dict[str, Any]],
        output_root: str = "runspace/outputs",
        db_path: Optional[str] = None,
        parallel: bool = False
    ) -> List[Dict[str, Any]]:
        raise RuntimeError(
            "Runner batch APIs are deprecated. "
            "Iterate configs in the caller and use run_single_logged(config, ...)."
        )

    def _normalize_input_quant_cfg(
        self,
        input_quant_cfg: Optional[Dict[str, Any]] = None,
        dynamic_input_quant_cfg: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Normalize input quantization config into a single schema:
          {
            enabled: bool,
            mode: dynamic|uniform|input_only,
            metric: mse|l1,           # dynamic
            format: fp*_e*m*,         # uniform/input_only
            chunk_size: int,
            candidate_formats: [...], # dynamic optional
          }
        """
        cfg = copy.deepcopy(input_quant_cfg or {})
        if not cfg and dynamic_input_quant_cfg:
            dyn = copy.deepcopy(dynamic_input_quant_cfg)
            if dyn.get('enabled', False):
                dyn['mode'] = 'dynamic'
            cfg = dyn
        cfg.setdefault('enabled', False)
        if cfg.get('enabled') and 'mode' not in cfg:
            cfg['mode'] = 'dynamic'
        return cfg

    def _build_layer_input_quantizer(self, model, input_quant_cfg: Dict[str, Any]):
        if not input_quant_cfg.get('enabled', False):
            return None

        mode = input_quant_cfg.get('mode')
        chunk_size = int(input_quant_cfg.get('chunk_size', 128))

        if mode == 'dynamic':
            from src.quantization.dynamic_input_quantizer import DynamicInputQuantizer
            candidate_formats = input_quant_cfg.get('candidate_formats')
            if isinstance(candidate_formats, str):
                candidate_formats = [f.strip() for f in candidate_formats.split(',') if f.strip()]
            metric = input_quant_cfg.get('metric', 'mse')
            quantizer = DynamicInputQuantizer(
                model=model,
                metric=metric,
                chunk_size=chunk_size,
                candidate_formats=candidate_formats,
            )
            quantizer.register_hooks()
            print(f"Input quantization enabled: mode=dynamic metric={metric} chunk_size={chunk_size}")
            return quantizer

        if mode == 'uniform':
            from src.quantization.uniform_input_quantizer import UniformInputQuantizer
            fmt = input_quant_cfg.get('format')
            if not fmt:
                raise ValueError("input_quant.mode=uniform requires input_quant.format")
            quantizer = UniformInputQuantizer(
                model=model,
                fmt=fmt,
                chunk_size=chunk_size,
            )
            quantizer.register_hooks()
            print(f"Input quantization enabled: mode=uniform format={fmt} chunk_size={chunk_size}")
            return quantizer

        # input_only is handled in the evaluation loop
        if mode == 'input_only':
            print(
                "Input quantization enabled: mode=input_only "
                f"format={input_quant_cfg.get('format')} chunk_size={chunk_size}"
            )
            return None

        raise ValueError(f"Unsupported input quantization mode: {mode}")

    @staticmethod
    def _collect_layer_input_quant_stats(quantizer, input_quant_cfg: Dict[str, Any]):
        mode = input_quant_cfg.get('mode')
        final_stats = quantizer.get_final_stats()
        raw_layer_stats = getattr(quantizer, 'layer_stats', {}) or {}
        layer_type_map = {}
        model = getattr(quantizer, 'model', None)
        if model is not None:
            try:
                layer_type_map = {
                    str(name): module.__class__.__name__
                    for name, module in model.named_modules()
                }
            except Exception:
                layer_type_map = {}

        layer_stats = {}
        if isinstance(raw_layer_stats, dict):
            for layer_name, layer_entry in raw_layer_stats.items():
                if not isinstance(layer_entry, dict):
                    continue
                normalized = dict(layer_entry)
                normalized.setdefault('type', layer_type_map.get(str(layer_name), 'unknown'))
                layer_stats[str(layer_name)] = normalized

        stats = {
            'mode': mode,
            'chunk_size': int(input_quant_cfg.get('chunk_size', 128)),
            'norm_l1': final_stats.get('norm_l1', 0.0),
            'norm_mse': final_stats.get('norm_mse', 0.0),
            'total_l1': final_stats.get('total_l1', 0.0),
            'total_mse': final_stats.get('total_mse', 0.0),
            'layer_stats': layer_stats,
        }
        if mode == 'dynamic':
            stats['metric'] = input_quant_cfg.get('metric', 'mse')
        if mode == 'uniform':
            stats['format'] = input_quant_cfg.get('format')
        return stats

    def evaluate_model(
        self,
        model,
        data_loader,
        adapter,
        max_batches=-1,
        desc="Evaluating",
        batch_callback=None,
        dynamic_input_quant_cfg: Optional[Dict[str, Any]] = None,
        input_quant_cfg: Optional[Dict[str, Any]] = None
    ):
        """Evaluate a model via Runner while updating running accuracy on the progress bar."""
        metrics_engine = adapter.create_metrics()
        model.eval()
        input_quantizer = None
        input_quant_stats = None
        normalized_input_quant_cfg = self._normalize_input_quant_cfg(
            input_quant_cfg=input_quant_cfg,
            dynamic_input_quant_cfg=dynamic_input_quant_cfg,
        )

        loader_len = None
        try:
            loader_len = len(data_loader)
        except Exception:
            loader_len = None

        if max_batches > 0:
            total_batches = min(max_batches, loader_len) if loader_len is not None else max_batches
        else:
            total_batches = loader_len

        input_only_stats = {
            'sum_l1_err': 0.0,
            'sum_mse_err': 0.0,
            'sum_l1_norm': 0.0,
            'sum_l2_norm': 0.0,
        }

        try:
            input_quantizer = self._build_layer_input_quantizer(
                model,
                input_quant_cfg=normalized_input_quant_cfg
            )

            with torch.inference_mode():
                pbar = tqdm(
                    total=total_batches,
                    desc=desc,
                    unit="batch",
                    dynamic_ncols=True,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                )
                for batch_idx, batch in enumerate(data_loader):
                    if max_batches > 0 and batch_idx >= max_batches:
                        break

                    inputs, targets = adapter.prepare_batch(batch)
                    inputs = self._to_device(inputs)
                    targets = self._to_device(targets)

                    if normalized_input_quant_cfg.get('enabled', False) and normalized_input_quant_cfg.get('mode') == 'input_only':
                        from src.ops.quant_base import quantize_tensor
                        fmt = normalized_input_quant_cfg.get('format')
                        if not fmt:
                            raise ValueError("input_quant.mode=input_only requires input_quant.format")
                        chunk_size = int(normalized_input_quant_cfg.get('chunk_size', 128))
                        q_inputs, _ = quantize_tensor(
                            inputs,
                            q_type=fmt,
                            mode='chunk',
                            chunk_size=chunk_size,
                            rounding='nearest',
                        )
                        diff = inputs - q_inputs
                        input_only_stats['sum_l1_err'] += diff.abs().sum().item()
                        input_only_stats['sum_mse_err'] += diff.pow(2).sum().item()
                        input_only_stats['sum_l1_norm'] += inputs.abs().sum().item()
                        input_only_stats['sum_l2_norm'] += inputs.pow(2).sum().item()
                        inputs = q_inputs

                    outputs = adapter.forward(model, (inputs, targets))
                    metrics_engine.update(outputs, targets)
                    if batch_callback is not None:
                        batch_callback(inputs, targets, outputs, batch_idx)

                    pbar.update(1)
                    acc1, acc5 = self._running_accuracy(metrics_engine)
                    remaining_batches = (
                        max((total_batches or 0) - (batch_idx + 1), 0)
                        if total_batches is not None else "?"
                    )
                    pbar.set_postfix({
                        'acc1': f"{acc1:.2f}%",
                        'acc5': f"{acc5:.2f}%",
                        'remaining_batches': remaining_batches
                    })
                pbar.close()
        finally:
            if input_quantizer is not None:
                input_quant_stats = self._collect_layer_input_quant_stats(
                    input_quantizer,
                    input_quant_cfg=normalized_input_quant_cfg
                )
                input_quantizer.cleanup()

        if (
            input_quant_stats is None
            and normalized_input_quant_cfg.get('enabled', False)
            and normalized_input_quant_cfg.get('mode') == 'input_only'
        ):
            norm_l1 = (
                input_only_stats['sum_l1_err'] / input_only_stats['sum_l1_norm']
                if input_only_stats['sum_l1_norm'] > 0 else 0.0
            )
            norm_mse = (
                input_only_stats['sum_mse_err'] / input_only_stats['sum_l2_norm']
                if input_only_stats['sum_l2_norm'] > 0 else 0.0
            )
            input_quant_stats = {
                'mode': 'input_only',
                'format': normalized_input_quant_cfg.get('format'),
                'chunk_size': int(normalized_input_quant_cfg.get('chunk_size', 128)),
                'norm_l1': norm_l1,
                'norm_mse': norm_mse,
                'total_l1': input_only_stats['sum_l1_err'],
                'total_mse': input_only_stats['sum_mse_err'],
                'layer_stats': {},
            }

        eval_results = metrics_engine.compute()
        if input_quant_stats is not None:
            eval_results['input_quant'] = input_quant_stats
            # Backward compatibility for scripts expecting dynamic_input_quant.
            if input_quant_stats.get('mode') == 'dynamic':
                eval_results['dynamic_input_quant'] = input_quant_stats
        return eval_results

    @staticmethod
    def _resolve_paths(cfg: dict, keys: tuple) -> dict:
        """Resolve relative path values in cfg against PROJECT_ROOT (runspace/).

        Skips torchvision/timm weight-enum sentinels (e.g. 'DEFAULT',
        'IMAGENET1K_V1') and bare names without separators or known checkpoint
        suffixes — these are not filesystem paths and must not be rewritten.
        """
        _path_suffixes = ('.pt', '.pth', '.bin', '.ckpt', '.safetensors')
        for key in keys:
            if key not in cfg or not isinstance(cfg[key], str):
                continue
            value = cfg[key]
            if os.path.isabs(value):
                continue
            looks_pathy = (
                ('/' in value) or ('\\' in value)
                or value.endswith(_path_suffixes)
            )
            if not looks_pathy:
                continue
            cfg[key] = os.path.normpath(os.path.join(PROJECT_ROOT, value))
        return cfg

    @staticmethod
    def _check_dataset_pipeline_compatibility(config: dict) -> None:
        """
        Fail fast when a feature-matching pipeline is paired with an
        incompatible dataset or a `quantize_components` list referencing
        components the pipeline does not declare. Silently skips for non-FM
        adapters and for pipelines/datasets that didn't declare their keys.
        """
        adapter_cfg = config.get('adapter', {})
        if adapter_cfg.get('type') != 'feature_matching':
            return

        pipeline_name = config.get('model', {}).get('name')
        dataset_name = config.get('dataset', {}).get('name')
        if not pipeline_name:
            return

        import src.adapters.pipelines  # noqa: F401 — trigger registrations
        import src.datasets  # noqa: F401 — trigger registrations
        from src.adapters.pipeline_registry import (
            get_pipeline_components,
            get_pipeline_required_keys,
        )
        from src.datasets.dataset_registry import get_dataset_provided_keys

        # 1) quantize_components ⊆ pipeline-declared components
        requested_components = adapter_cfg.get('quantize_components') or []
        if requested_components:
            try:
                declared = get_pipeline_components(pipeline_name)
            except KeyError:
                declared = None
            if declared is not None:
                unknown = [c for c in requested_components if c not in declared]
                if unknown:
                    raise ValueError(
                        f"Pipeline '{pipeline_name}' declares components "
                        f"{sorted(declared)}, but adapter.quantize_components "
                        f"requests {sorted(requested_components)}. "
                        f"Unknown: {sorted(unknown)}."
                    )

        # 2) dataset provided_keys ⊇ pipeline required_input_keys
        if not dataset_name:
            return
        try:
            required = set(get_pipeline_required_keys(pipeline_name))
            provided = set(get_dataset_provided_keys(dataset_name))
        except KeyError:
            return
        if not required or not provided:
            return
        missing = required - provided
        if missing:
            raise ValueError(
                f"Dataset '{dataset_name}' provides keys {sorted(provided)}, but "
                f"pipeline '{pipeline_name}' requires {sorted(required)}. "
                f"Missing: {sorted(missing)}."
            )

    def setup_data_loader(self, config: dict):
        """Setup the data loader from config via DatasetRegistry."""
        import resource
        import src.datasets  # triggers all @register_dataset decorators
        from src.datasets.dataset_registry import build_data_loader
        dataset_cfg = dict(config['dataset'])
        # Pass model info so imagenet loader can resolve timm transforms
        dataset_cfg['model_source'] = config.get('model', {}).get('source', 'torchvision')
        dataset_cfg['model_name'] = config.get('model', {}).get('name', '')
        self._resolve_paths(dataset_cfg, ('path', 'pairs_file', 'root'))

        # Cap num_workers to avoid exhausting fds or RAM during long sequential runs.
        # On unified-memory hosts (e.g. GB10: CPU+GPU share one LPDDR5X pool),
        # worker RSS starves the NVIDIA driver and can crash the host, so we also
        # apply a memory-based cap. x86_64 hosts with discrete VRAM don't need it.
        requested = dataset_cfg.get('num_workers', 0)
        if requested > 0:
            try:
                import platform
                soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
                fd_cap = max(1, (soft_limit - 512) // 20)
                cpu_cap = os.cpu_count() or 4
                caps = {'fd_cap': fd_cap, 'cpu_cap': cpu_cap}

                # Only apply a memory-based cap on unified-memory (ARM64) hosts.
                if platform.machine() in ('aarch64', 'arm64'):
                    try:
                        with open('/proc/meminfo') as f:
                            for line in f:
                                if line.startswith('MemTotal:'):
                                    total_kb = int(line.split()[1])
                                    total_gb = total_kb / (1024 * 1024)
                                    # ~4GB RSS per worker; reserve 24GB for model,
                                    # activations, and torch allocator cache growth.
                                    caps['mem_cap'] = max(1, int((total_gb - 24) // 4))
                                    break
                    except Exception:
                        pass

                safe_max = max(1, min(caps.values()))
                if requested > safe_max:
                    cap_str = ", ".join(f"{k}={v}" for k, v in caps.items())
                    print(f"Info: Capping num_workers from {requested} to {safe_max} "
                          f"({cap_str}).")
                    dataset_cfg['num_workers'] = safe_max
            except Exception:
                pass

        return build_data_loader(dataset_cfg['name'], dataset_cfg)

    def run_single(self, config: Dict[str, Any], output_root: str = "runspace/outputs") -> Dict[str, Any]:
        """Runs a single configuration and returns the results."""
        model_config = config.get('model', {})
        self._resolve_paths(model_config, ('weights', 'repo_path', 'root', 'path'))
        model_name = model_config.get('name', 'unknown')

        self._check_dataset_pipeline_compatibility(config)
        weights_spec = model_config.get('weights')
        has_file_backed_weights = (
            isinstance(weights_spec, str)
            and weights_spec not in ("", "DEFAULT", "default", "None", "none")
            and os.path.isfile(weights_spec)
        )
        quant_format = config.get('quantization', {}).get('format', 'fp8')
        
        # Check for mixed precision
        layer_configs = config.get('quantization', {}).get('layers', {})
        if layer_configs:
            unique_formats = set()
            for layer_name, layer_cfg in layer_configs.items():
                if isinstance(layer_cfg, dict) and 'format' in layer_cfg:
                    unique_formats.add(layer_cfg['format'])
            
            if len(unique_formats) > 1:
                quant_format = "mixed"
            elif len(unique_formats) == 1:
                # If all layers override to the same thing, report that thing
                quant_format = list(unique_formats)[0]
        
        meta = config.get('meta', {})
        
        print(f"--- Running {model_name} with {quant_format} ---")
        
        results = {
            'model_name': model_name,
            'adapter_type': config.get('adapter', {}).get('type', 'generic'),
            'quant_format': quant_format,
            'base_config_path': meta.get('base_config_path', 'N/A'),
            'generated_config_path': meta.get('generated_config_path', 'N/A'),
            'output_name': config.get('output_name', ''), # Custom identifier
            'source_weights': weights_spec,
            'materialized_weight_path': None,
            'status': 'FAILED',
            'acc1': 0.0,
            'acc5': 0.0,
            'certainty': 0.0,
            'ref_acc1': 0.0,
            'ref_acc5': 0.0,
            'ref_certainty': 0.0,
            'acc_drop': 0.0,
            'weight_comp_red': 0.0,
            'weight_comp_share': 0.0,
            'input_comp_red': 0.0,
            'input_comp_share': 0.0,
            'dyn_metric': None,
            'dyn_norm_l1': 0.0,
            'dyn_norm_mse': 0.0,
            'fm_num_keypoints': 0.0,
            'fm_mean_score': 0.0,
            'fm_desc_norm': 0.0,
            'fm_repeatability': 0.0,
            'matching_precision': 0.0,
            'matching_score': 0.0,
            'mean_num_matches': 0.0,
            'pose_auc_5': 0.0,
            'pose_auc_10': 0.0,
            'pose_auc_20': 0.0,
            'ref_matching_precision': 0.0,
            'ref_matching_score': 0.0,
            'ref_mean_num_matches': 0.0,
            'ref_pose_auc_5': 0.0,
            'ref_pose_auc_10': 0.0,
            'ref_pose_auc_20': 0.0,
            'exec_error': None
        }

        try:
            # Generate Output Directory
            base_config_path = meta.get('base_config_path', '')
            output_name = config.get('output_name', '')
            
            if output_name:
                output_dir = os.path.join(output_root, output_name)
            elif base_config_path:
                base_config_name = os.path.splitext(os.path.basename(base_config_path))[0]
                output_dir = os.path.join(output_root, base_config_name, model_name, quant_format.replace('_', ''))
            else:
                output_dir = os.path.join(output_root, model_name, quant_format.replace('_', ''))
            
            os.makedirs(output_dir, exist_ok=True)

            # Save Config
            config_path = os.path.join(output_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"Config saved to {config_path}")

            # Check if GRAPH ONLY mode
            if config.get('evaluation', {}).get('graph_only', False):
                print("Running in GRAPH ONLY mode. Skipping data loading.")
            else:
                # Setup data
                data_loader = self.setup_data_loader(config)
                if data_loader is None:
                    results['error'] = "Data loader failed"
                    return results

            # Build model from materialized weight file (single source of truth).
            model, adapter, materialized_weight_path = self.prepare_model_with_materialized_weights(
                config=config,
                output_dir=output_dir
            )
            results['materialized_weight_path'] = materialized_weight_path
            resolved_quant_map = self._resolve_quant_map_for_run(
                config=config,
                materialized_weight_path=materialized_weight_path
            )
            if resolved_quant_map is not None:
                results['weight_quant_map'] = resolved_quant_map
            
            # Check for quantized layers
            quant_layer_count = 0
            for module in model.modules():
                # Check if module name contains 'Quant' (simple heuristic based on class name)
                if 'Quant' in module.__class__.__name__:
                    quant_layer_count += 1
            
            if quant_layer_count == 0:
                if has_file_backed_weights:
                    print(
                        f"Info: No quantized wrapper layers found for {model_name}, "
                        "but file-backed weights are in use."
                    )
                else:
                    if self._expects_quantized_layers(config):
                        print(f"Warning: No quantized layers found for {model_name}")
                        results['status'] = 'NO_QUANT'
                    else:
                        print(
                            f"Info: No quantized layers found for {model_name}, "
                            "and none were requested (reference/FP32 mode)."
                        )
            
            # Generate Quantization Graph
            generate_graph = config.get('evaluation', {}).get('generate_graph_svg', True)
            if generate_graph:
                try:
                    from src.utils.graph_viz import generate_quantization_graph
                    graph_path = os.path.join(output_dir, "quant_graph.svg")
                    print(f"Generating quantization graph at {graph_path}...")
                    generate_quantization_graph(model, graph_path, model_name=model_name)
                    
                    if config.get('evaluation', {}).get('graph_only', False):
                        results['status'] = 'GRAPH_GENERATED'
                        print("Graph generation complete. Exiting due to graph-only mode.")
                        return results
                        
                except FileNotFoundError as e:
                    if '"dot"' in str(e) or "'dot'" in str(e) or 'dot' in str(e).lower():
                        print("Skipping graph SVG: Graphviz 'dot' not found. Install with: winget install graphviz")
                    else:
                        print(f"Failed to generate graph: {e}")
                except Exception as e:
                    print(f"Failed to generate graph: {e}")
            else:
                print("Skipping graph generation (disabled in config)")
                if config.get('evaluation', {}).get('graph_only', False):
                     print("Graph-only mode set but graph generation disabled. Nothing to do.")
                     results['status'] = 'SKIPPED'
                     return results

            # Check evaluation mode
            eval_mode = config.get('evaluation', {}).get('mode', 'compare')
            dynamic_input_quant_cfg = config.get('evaluation', {}).get('dynamic_input_quant', {})
            input_quant_cfg = config.get('evaluation', {}).get('input_quant', {})

            if eval_mode == 'evaluate':
                print(f"Running in EVALUATE mode (Quantized Model Only)...")
                max_batches = config.get('evaluation', {}).get('max_batches', -1)
                eval_results = self.evaluate_model(
                    model=model,
                    data_loader=data_loader,
                    adapter=adapter,
                    max_batches=max_batches,
                    dynamic_input_quant_cfg=dynamic_input_quant_cfg,
                    input_quant_cfg=input_quant_cfg
                )

                results.update({k: v for k, v in eval_results.items() if k in results})
                input_quant_stats = eval_results.get('input_quant') or eval_results.get('dynamic_input_quant')
                if input_quant_stats:
                    results['input_quant'] = input_quant_stats
                    if input_quant_stats.get('mode') == 'dynamic':
                        results['dyn_metric'] = input_quant_stats.get('metric')
                        results['dyn_norm_l1'] = input_quant_stats.get('norm_l1', 0.0)
                        results['dyn_norm_mse'] = input_quant_stats.get('norm_mse', 0.0)

                if results['status'] != 'NO_QUANT':
                    results['status'] = 'SUCCESS'
                
            else:
                # COMPARE mode
                print(f"Running in COMPARE mode (Reference vs Quantized)...")
                
                # Build structured reference model using a proper config to ensure 
                # identical structure (folding, ops) to the quantized model.
                print(f"Building structured reference model (FP32 baseline)...")
                ref_config = self._build_reference_config(config)
                
                # We use a sub-directory for the reference to avoid overwriting the main config/weights
                ref_output_dir = os.path.join(output_dir, "reference_fp32")
                os.makedirs(ref_output_dir, exist_ok=True)
                
                ref_model, _, _ = self.prepare_model_with_materialized_weights(
                    config=ref_config,
                    output_dir=ref_output_dir
                )
                # Note: prepare_model_with_materialized_weights already moved ref_model to self.device

                eval_cfg = config.get('evaluation', {})
                comparator = LayerComparator(
                    ref_model,
                    model,
                    model_name=model_name,
                    quant_type=quant_format.replace('_', ''),
                    adapter=adapter,
                    device=self.device,
                    output_dir=output_dir,
                    save_histograms=eval_cfg.get('save_histograms', False),
                    save_visualizations=eval_cfg.get('save_visualizations', False),
                    num_viz_samples=eval_cfg.get('num_viz_samples', 5),
                )
                
                # Determine number of batches
                compare_batches = config.get('evaluation', {}).get('compare_batches', -1)
                if compare_batches == -1:
                    compare_batches = len(data_loader)

                dynamic_quantizer = None
                dynamic_stats = None
                try:
                    # compare-mode currently supports hook-based modes only.
                    compare_input_quant_cfg = self._normalize_input_quant_cfg(
                        input_quant_cfg=input_quant_cfg,
                        dynamic_input_quant_cfg=dynamic_input_quant_cfg
                    )
                    if compare_input_quant_cfg.get('enabled', False) and compare_input_quant_cfg.get('mode') == 'input_only':
                        print("Warning: input_quant.mode=input_only is not supported in compare mode. Ignoring.")
                        compare_input_quant_cfg['enabled'] = False
                    dynamic_quantizer = self._build_layer_input_quantizer(
                        model,
                        input_quant_cfg=compare_input_quant_cfg
                    )

                    # Run Comparison (Single Pass)
                    comparator.compare(data_loader, num_batches=compare_batches, global_metrics=None)
                finally:
                    if dynamic_quantizer is not None:
                        dynamic_stats = self._collect_layer_input_quant_stats(dynamic_quantizer, compare_input_quant_cfg)
                        dynamic_quantizer.cleanup()
                
                # Retrieve metrics from comparator
                print("Retrieving metrics from comparator...")
                quant_metrics = comparator.quant_metrics.compute()
                ref_metrics = comparator.ref_metrics.compute()

                results.update({k: v for k, v in quant_metrics.items() if k in results})
                results.update({f'ref_{k}': v for k, v in ref_metrics.items() if f'ref_{k}' in results})

                primary = quant_metrics['acc1'] if 'acc1' in quant_metrics else quant_metrics.get('fm_repeatability', 0.0)
                ref_primary = ref_metrics['acc1'] if 'acc1' in ref_metrics else ref_metrics.get('fm_repeatability', 0.0)
                results['acc_drop'] = ref_primary - primary
                if dynamic_stats:
                    results['input_quant'] = dynamic_stats
                    results['dyn_metric'] = dynamic_stats.get('metric')
                    results['dyn_norm_l1'] = dynamic_stats.get('norm_l1', 0.0)
                    results['dyn_norm_mse'] = dynamic_stats.get('norm_mse', 0.0)
                
                # Retrieve Compression Stats
                comp_stats = comparator.compression_tracker.get_stats()
                results['weight_comp_red'] = comp_stats['weight_compression_reduction']
                results['weight_comp_share'] = comp_stats['weight_share_of_total']
                results['input_comp_red'] = comp_stats['input_compression_reduction']
                results['input_comp_share'] = comp_stats['input_share_of_total']

                comparator.close()
                
                if results['status'] != 'NO_QUANT':
                    results['status'] = 'SUCCESS'
                    # Check for unquantized supported ops
                    if hasattr(comparator, 'unquantized_supported_count') and comparator.unquantized_supported_count > 0:
                        results['status'] = f"SUCCESS ({comparator.unquantized_supported_count} Unquantized)"
                        
                results['report_path'] = f"{output_dir}/comparison_report.txt"

        except Exception as e:
            print(f"Error running config for {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results['status'] = 'FAILED'
            results['exec_error'] = str(e)
        
        finally:
            print(f"Cleaning up memory for {model_name}...")
            if 'model' in locals(): del model
            if 'ref_model' in locals(): del ref_model
            if 'adapter' in locals(): del adapter
            if 'evaluator' in locals(): del evaluator
            if 'ref_evaluator' in locals(): del ref_evaluator
            if 'comparator' in locals(): del comparator
            if 'dynamic_quantizer' in locals(): del dynamic_quantizer
            if 'input_quantizer' in locals(): del input_quantizer
            if 'eval_results' in locals(): del eval_results
            if 'quant_metrics' in locals(): del quant_metrics
            if 'ref_metrics' in locals(): del ref_metrics
            if 'data_loader' in locals():
                self._shutdown_dataloader_workers(data_loader)
                del data_loader
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
            
        return results

    def run_batch(self, configs: List[Dict[str, Any]], output_root: str = "runspace/outputs") -> List[Dict[str, Any]]:
        raise RuntimeError(
            "Runner batch APIs are deprecated. "
            "Iterate configs in the caller and use run_single(config, ...)."
        )

    def run_batch_parallel(self, configs: List[Dict[str, Any]], output_root: str = "runspace/outputs") -> List[Dict[str, Any]]:
        raise RuntimeError(
            "Runner parallel batch API is deprecated. "
            "Iterate configs in the caller and use run_single(config, ...)."
        )

    def _run_parallel_group_safe(self, configs: List[Dict[str, Any]], output_root: str) -> List[Dict[str, Any]]:
        """Recursively try to run configs. If OOM, split."""
        if not configs: return []
        
        # Ensure clean slate
        gc.collect()
        torch.cuda.empty_cache()
        
        try:
            return self._run_parallel_execution(configs, output_root)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM detected with {len(configs)} models. Splitting batch...")
                torch.cuda.empty_cache()
                gc.collect()
                
                if len(configs) <= 1:
                     print(f"Error: Single model caused OOM. Marking as failed.")
                     return [self._create_failed_result(configs[0], f"OOM Error: {str(e)}")]
                     
                mid = len(configs) // 2
                left = configs[:mid]
                right = configs[mid:]
                
                print(f"Retrying with split batches: {len(left)} and {len(right)}")
                res_left = self._run_parallel_group_safe(left, output_root)
                
                torch.cuda.empty_cache()
                gc.collect()
                
                res_right = self._run_parallel_group_safe(right, output_root)
                return res_left + res_right
            else:
                print(f"Unexpected error in parallel batch: {e}")
                import traceback
                traceback.print_exc()
                # Fail all
                return [self._create_failed_result(c, str(e)) for c in configs]

    def _create_failed_result(self, config, error_msg):
        model_name = config.get('model', {}).get('name', 'unknown')
        quant_format = config.get('quantization', {}).get('format', 'fp8')
        return {
            'model_name': model_name,
            'quant_format': quant_format,
            'output_name': config.get('output_name', ''),
            'status': 'FAILED',
            'exec_error': error_msg,
            'acc1': 0.0, 'acc5': 0.0, 'certainty': 0.0
        }

    def _setup_single_context(self, config, output_root):
        """Prepares a single model context for parallel execution."""
        model_config = config.get('model', {})
        model_name = model_config.get('name', 'unknown')
        weights_spec = model_config.get('weights')
        has_file_backed_weights = (
            isinstance(weights_spec, str)
            and weights_spec not in ("", "DEFAULT", "default", "None", "none")
            and os.path.isfile(weights_spec)
        )
        quant_format = config.get('quantization', {}).get('format', 'fp8')
        
        # Determine paths (same logic as run_single)
        meta = config.get('meta', {})
        base_config_path = meta.get('base_config_path', '')
        output_name = config.get('output_name', '')
        
        if output_name:
            output_dir = os.path.join(output_root, output_name)
        elif base_config_path:
            base_config_name = os.path.splitext(os.path.basename(base_config_path))[0]
            output_dir = os.path.join(output_root, base_config_name, model_name, quant_format.replace('_', ''))
        else:
            output_dir = os.path.join(output_root, model_name, quant_format.replace('_', ''))
            
        result_template = {
            'model_name': model_name,
            'quant_format': quant_format,
            'output_name': output_name,
            'materialized_weight_path': None,
            'status': 'FAILED',
            'acc1': 0.0, 'acc5': 0.0, 'certainty': 0.0,
            'ref_acc1': 0.0, 'ref_acc5': 0.0
        }
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save Config
            config_path = os.path.join(output_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
                
            # Build model from materialized weight file.
            model, adapter, materialized_weight_path = self.prepare_model_with_materialized_weights(
                config=config,
                output_dir=output_dir
            )
            result_template['materialized_weight_path'] = materialized_weight_path
            
            # Check quant layers
            quant_layer_count = 0
            for module in model.modules():
                if 'Quant' in module.__class__.__name__:
                    quant_layer_count += 1
            
            if quant_layer_count == 0:
                if has_file_backed_weights:
                    print(
                        f"Info: No quantized wrapper layers found for {model_name} ({output_name}), "
                        "but file-backed weights are in use."
                    )
                else:
                    print(f"Warning: No quantized layers found for {model_name} ({output_name})")
                
            # Metrics
            metrics_engine = MetricsEngine()
            
            return {
                'status': 'READY',
                'model': model,
                'adapter': adapter,
                'metrics': metrics_engine,
                'result': result_template,
                'output_dir': output_dir,
                'quant_layer_count': quant_layer_count,
                'has_file_backed_weights': has_file_backed_weights,
            }
            
        except Exception as e:
            print(f"Error setting up {output_name}: {e}")
            result_template['exec_error'] = str(e)
            return {'status': 'FAILED', 'result': result_template}

    def _run_parallel_execution(self, configs, output_root):
        print(f"--- Loading {len(configs)} models for parallel execution ---")
        
        # Suppress torchvision warnings during bulk loading
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
        
        # 1. Setup Models
        contexts = []
        for cfg in tqdm(configs, desc="Loading Models"):
            ctx = self._setup_single_context(cfg, output_root)
            contexts.append(ctx)
            
        active_contexts = [c for c in contexts if c['status'] == 'READY']
        
        if not active_contexts:
            print("No valid models to run.")
            return [c['result'] for c in contexts]
            
        print(f"Successfully loaded {len(active_contexts)} models.")
        
        # 2. Setup Data Loader (using first config)
        loader = self.setup_data_loader(configs[0])
        if loader is None:
             raise RuntimeError("Failed to setup data loader")
             
        # 3. Execution Loop
        max_batches = configs[0].get('evaluation', {}).get('max_batches', -1)
        try:
            with torch.inference_mode():
                pbar = tqdm(loader, desc=f"Parallel Eval ({len(active_contexts)} models)", unit="batch")
                for batch_idx, batch in enumerate(pbar):
                    if max_batches > 0 and batch_idx >= max_batches:
                        break
                    # Use first adapter for preparation (assuming compatible input pipeline)
                    adapter = active_contexts[0]['adapter']
                    inputs, targets = adapter.prepare_batch(batch)
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    for ctx in active_contexts:
                        mdl = ctx['model']
                        adp = ctx['adapter']
                        
                        # Forward
                        outputs = adp.forward(mdl, (inputs, targets))
                        ctx['metrics'].update(outputs, targets)
                        
                        # Optional: clear intermediate tensors if tight on memory?
                        # outputs usually needed for metrics.
                        del outputs

                    running_acc1 = []
                    running_acc5 = []
                    for ctx in active_contexts:
                        acc1, acc5 = self._running_accuracy(ctx['metrics'])
                        running_acc1.append(acc1)
                        running_acc5.append(acc5)

                    if running_acc1 and running_acc5:
                        avg_acc1 = sum(running_acc1) / len(running_acc1)
                        avg_acc5 = sum(running_acc5) / len(running_acc5)
                        pbar.set_postfix({
                            'avg_acc1': f"{avg_acc1:.2f}%",
                            'avg_acc5': f"{avg_acc5:.2f}%"
                        })
                    
        except RuntimeError as e:
            # If OOM happens here, it bubbles up
            raise e
        finally:
             if 'inputs' in locals(): del inputs
             if 'targets' in locals(): del targets
             
        # 4. Finalize
        results = []
        for ctx in contexts:
            if ctx['status'] == 'READY':
                # Compute metrics
                m = ctx['metrics'].compute()
                res = ctx['result']
                res['acc1'] = m.get('acc1', 0.0)
                res['acc5'] = m.get('acc5', 0.0)
                res['certainty'] = m.get('certainty', 0.0)
                
                if ctx['quant_layer_count'] == 0 and not ctx.get('has_file_backed_weights', False):
                    res['status'] = 'NO_QUANT'
                else:
                    res['status'] = 'SUCCESS'
                    
                results.append(res)
                
                # Cleanup
                del ctx['model']
                del ctx['adapter']
                del ctx['metrics']
            else:
                results.append(ctx['result'])
        
        # Global Cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        return results
