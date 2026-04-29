import copy
import gc
import json
import os

import torch


def build_runtime_config(args, model_name=None, weights=None):
    """Build the shared generic fp32 runtime config used by experiment scripts."""
    model_cfg = {
        'name': model_name or args.model_name,
        'weights': weights or args.weights,
    }
    model_source = getattr(args, 'model_source', None)
    if model_source is not None:
        model_cfg['source'] = model_source

    return {
        'model': model_cfg,
        'adapter': {'type': 'generic', 'quantized_ops': []},
        'dataset': {
            'name': args.dataset_name,
            'path': args.dataset_path,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
        },
        'experiment': {
            'materialize_weights': {
                'force_rebuild': bool(getattr(args, 'force_rerun', False)),
            },
        },
    }


def build_fp32_runtime_config(args, model_name=None, weights=None):
    """Build a plain fp32 runtime config with no quantized adapter wrappers."""
    cfg = build_runtime_config(args, model_name=model_name, weights=weights)
    cfg['adapter'].update({
        'build_quantized': False,
        'weight_quantization': False,
        'input_quantization': False,
    })
    cfg['quantization'] = {'weight_source': 'fp32'}
    return cfg


def build_prequantized_weight_runtime_config(args, model_name=None, weights=None):
    """
    Build a config for evaluating weights that were already quantized into the state_dict.

    The adapter flags stay false on purpose: this path must not quantize weights a
    second time. The explicit weight_source records that policy in logged configs.
    """
    cfg = build_fp32_runtime_config(args, model_name=model_name, weights=weights)
    cfg['quantization']['weight_source'] = 'prequantized_state_dict'
    return cfg


def build_runtime_weight_quant_config(args, fmt, chunk_size, model_name=None, weights=None):
    """Build a config that asks the adapter/runner to quantize weights at runtime."""
    cfg = build_runtime_config(args, model_name=model_name, weights=weights)
    cfg['adapter'].update({
        'quantized_ops': ['all'],
        'build_quantized': True,
        'weight_quantization': True,
        'input_quantization': False,
    })
    cfg['quantization'] = {
        'format': fmt,
        'weight_mode': 'chunk',
        'weight_chunk_size': chunk_size,
        'weight_source': 'runtime_quantized',
    }
    return cfg


def build_loader(args, device, runner, config_builder=build_runtime_config):
    """Build a dataloader from an experiment runtime config."""
    return runner.setup_data_loader(config_builder(args))


def run_inference(runner, model, adapter, loader, args, input_quant_cfg=None, desc=""):
    """Run one inference pass and return accuracy/certainty/input-quant stats."""
    eval_results = runner.evaluate_model(
        model=model,
        data_loader=loader,
        adapter=adapter,
        max_batches=args.limit_batches,
        desc=desc,
        input_quant_cfg=input_quant_cfg,
    )
    return (
        eval_results.get('acc1', 0.0),
        eval_results.get('acc5', 0.0),
        eval_results.get('certainty', 0.0),
        eval_results.get('input_quant'),
    )


def load_fp32_model(runner, args, device, config_builder=build_runtime_config):
    """Load a model with original fp32 weights through the shared runtime config."""
    config = config_builder(args)
    fp32_dir = os.path.join(args.output_dir, args.model_name, "fp32_ref")
    model, adapter, _ = runner.prepare_model_with_materialized_weights(
        config=config,
        output_dir=fp32_dir,
    )
    return model, adapter


def get_or_run_fp32_ref(
    runner,
    args,
    device,
    db,
    model_name,
    experiment_name,
    config_builder=build_runtime_config,
    config_json_builder=None,
    respect_force_rerun=False,
):
    """Fetch an fp32/fp32 reference from DB or run and log one."""
    runs = db.get_runs()
    should_check_db = not (respect_force_rerun and getattr(args, 'force_rerun', False))
    if should_check_db and not runs.empty:
        fp32_rows = runs[
            (runs['model_name'] == model_name) &
            (runs['weight_dt'] == 'fp32') &
            (runs['activation_dt'] == 'fp32') &
            (runs['status'] == 'SUCCESS') &
            (runs['acc1'].notna())
        ]
        if not fp32_rows.empty:
            row = fp32_rows.iloc[0]
            ref_acc1 = float(row['acc1'])
            ref_acc5 = float(row['acc5'])
            ref_certainty = float(row['certainty']) if row['certainty'] is not None else 0.0
            print(
                f"[FP32 ref] Found in DB: Top1={ref_acc1:.2f}%, "
                f"Top5={ref_acc5:.2f}%"
            )
            return ref_acc1, ref_acc5, ref_certainty

    print("[FP32 ref] Not found in DB. Running fp32 inference ...")
    model, adapter = load_fp32_model(runner, args, device, config_builder=config_builder)
    loader = build_loader(args, device, runner, config_builder=config_builder)
    acc1, acc5, certainty, _ = run_inference(
        runner, model, adapter, loader, args, desc="FP32 ref"
    )
    del model, adapter
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[FP32 ref] Top1={acc1:.2f}%, Top5={acc5:.2f}%, Certainty={certainty:.4f}")
    log_cfg = config_builder(args, model_name=model_name, weights=args.weights)
    log_cfg['experiment'] = {
        'name': experiment_name,
        'type': 'fp32_ref',
        'weight_dt': 'fp32',
        'activation_dt': 'fp32',
        'ref_acc1': acc1,
        'ref_acc5': acc5,
        'ref_certainty': certainty,
        'metrics': {'certainty': certainty},
    }
    if config_json_builder is not None:
        log_cfg['experiment']['config_json'] = config_json_builder(config_builder(args))
    runner.log_experiment_result(
        config=log_cfg,
        result={
            'model_name': model_name,
            'status': 'SUCCESS',
            'acc1': acc1,
            'acc5': acc5,
            'certainty': certainty,
        },
    )
    return acc1, acc5, certainty


def run_exists(db, model_name, experiment_type, weight_dt, activation_dt):
    """Return True if a successful run exists in DB for the exact experiment combo."""
    runs = db.get_runs()
    if runs.empty:
        return False
    match = runs[
        (runs['model_name'] == model_name) &
        (runs['experiment_type'] == experiment_type) &
        (runs['weight_dt'] == weight_dt) &
        (runs['activation_dt'] == activation_dt) &
        (runs['status'] == 'SUCCESS')
    ]
    return not match.empty


def layer_types_from_model(model):
    """Return {layer_name: class_name} for each named module."""
    return {
        name: type(module).__name__
        for name, module in model.named_modules()
        if name
    }


def build_weight_map_json(quant_map, layer_types):
    """Build enriched weight quant map JSON with format counts and layer types."""
    enriched = {}
    for layer, fmt_spec in quant_map.items():
        counts = {}
        if isinstance(fmt_spec, list):
            for fmt in fmt_spec:
                key = str(fmt)
                counts[key] = counts.get(key, 0) + 1
        elif fmt_spec is not None:
            key = str(fmt_spec)
            counts[key] = counts.get(key, 0) + 1

        dominant = None
        if counts:
            dominant = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]

        enriched[layer] = {
            "format": fmt_spec,
            "type": layer_types.get(layer, "unknown"),
            "format_counts": counts,
            "total_chunks": int(sum(counts.values())),
            "dominant_format": dominant,
        }

    return json.dumps(enriched)


def build_uniform_input_quant_cfg(fmt, chunk_size, enabled=True):
    """Build the uniform layer-input quantizer config used by input quant baselines."""
    if not enabled or str(fmt).strip().lower() == 'fp32':
        return None
    return {
        'enabled': True,
        'mode': 'uniform',
        'format': fmt,
        'chunk_size': int(chunk_size),
    }


def build_dynamic_input_quant_cfg(metric, chunk_size, candidate_formats, enabled=True):
    """Build the dynamic layer-input quantizer config used by input quant experiments."""
    if not enabled:
        return None
    return {
        'enabled': True,
        'mode': 'dynamic',
        'metric': metric,
        'chunk_size': int(chunk_size),
        'candidate_formats': list(candidate_formats),
    }


def serialize_config(cfg, activation_fmt=None, chunk_size=128, weight_source=None):
    """
    Serialize a full adapter config dict to JSON for DB storage.
    Optionally merges activation quantization fields.
    """
    cfg = copy.deepcopy(cfg)
    cfg.get('adapter', {}).pop('skip_calibration', None)
    if weight_source is not None:
        cfg.setdefault('quantization', {})
        cfg['quantization']['weight_source'] = weight_source
    if activation_fmt is not None:
        cfg.setdefault('quantization', {})
        cfg['quantization']['input_quantization'] = True
        cfg['quantization']['activation_format'] = activation_fmt
        cfg['quantization']['activation_mode'] = 'chunk'
        cfg['quantization']['activation_chunk_size'] = chunk_size
    return json.dumps(cfg)
