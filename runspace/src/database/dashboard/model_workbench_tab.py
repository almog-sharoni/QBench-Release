"""Interactive model-analysis and quantization-conversion workbench.

This file is executed in the shared dashboard namespace by ``dashboard.py``.
The expensive work is button-driven and all model/conversion objects live in
``st.session_state`` so switching dashboard tabs does not rebuild a model.
"""

# This file is executed after setup.py/experiments_tab.py in a deliberately
# shared namespace; ``st`` and ``tab_workbench`` are runtime globals.
# ruff: noqa: F821

with tab_workbench:
    import copy as _mw_copy
    import hashlib as _mw_hashlib
    import io as _mw_io
    import json as _mw_json
    import math as _mw_math
    import re as _mw_re
    from collections import Counter as _MwCounter

    import pandas as _mw_pd
    import streamlit.components.v1 as _mw_components

    _MW_REQUIRED_ANALYSIS_SCHEMA = 2
    _MW_REQUIRED_BENCHMARK_API = 1
    try:
        import torch as _mw_torch
        import runspace.src.quantization.model_workbench as _mw_backend_module
        from runspace.src.quantization.model_workbench import (
            analyze_model as _mw_analyze_model,
            build_conversion_plan as _mw_build_conversion_plan,
            convert_model as _mw_convert_model,
            list_model_names as _mw_list_model_names,
            load_model as _mw_load_model,
            preview_conversion_plan as _mw_preview_conversion_plan,
            run_sample_inference as _mw_run_sample_inference,
        )
        from runspace.src.utils.model_input_utils import (
            resolve_model_input_size as _mw_resolve_model_input_size,
        )
        _MW_LOADED_ANALYSIS_SCHEMA = int(
            getattr(_mw_backend_module, "WORKBENCH_ANALYSIS_SCHEMA_VERSION", 0)
        )
        _MW_LOADED_BENCHMARK_API = int(
            getattr(
                _mw_backend_module,
                "WORKBENCH_DATASET_BENCHMARK_API_VERSION",
                0,
            )
        )
        _mw_build_classification_validation_loader = getattr(
            _mw_backend_module,
            "build_classification_validation_loader",
            None,
        )
        _mw_benchmark_classification_models = getattr(
            _mw_backend_module,
            "benchmark_classification_models",
            None,
        )
        _mw_list_replacement_targets = getattr(
            _mw_backend_module,
            "list_replacement_targets",
            None,
        )
        _mw_inspect_replacement_target = getattr(
            _mw_backend_module,
            "inspect_replacement_target",
            None,
        )
        _mw_validate_replacement_spec = getattr(
            _mw_backend_module,
            "validate_replacement_spec",
            None,
        )
        _MW_IMPORT_ERROR = None
    except Exception as _mw_exc:
        _MW_LOADED_ANALYSIS_SCHEMA = 0
        _MW_LOADED_BENCHMARK_API = 0
        _MW_IMPORT_ERROR = _mw_exc


    def _mw_as_dict(value):
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return to_dict()
        if hasattr(value, "__dict__"):
            return dict(value.__dict__)
        return {}


    def _mw_field(value, name, default=None):
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)


    def _mw_safe_filename(value):
        cleaned = _mw_re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "model"))
        return cleaned.strip("._") or "model"


    def _mw_parse_input_shape(value):
        parts = [part for part in _mw_re.split(r"[xX,\s]+", str(value).strip()) if part]
        if not 1 <= len(parts) <= 8:
            raise ValueError("Input shape must contain between 1 and 8 dimensions.")
        try:
            shape = tuple(int(part) for part in parts)
        except ValueError as exc:
            raise ValueError("Input shape must contain positive integers separated by commas or `x`.") from exc
        if any(dim <= 0 for dim in shape):
            raise ValueError("Every input dimension must be positive.")
        element_count = 1
        for dim in shape:
            element_count *= dim
        max_elements = 16_777_216  # 64 MiB for the float32 sample alone.
        if element_count > max_elements:
            raise ValueError(
                f"The sample would contain {element_count:,} elements. "
                f"The dashboard limit is {max_elements:,} elements (64 MiB float32)."
            )
        return shape


    _MW_REPLACEMENT_INITIALIZERS = {
        "initializer:target_default": "Keep target initialized (may be random)",
        "initializer:zeros": "Initialize with zeros",
        "initializer:ones": "Initialize with ones",
        "initializer:constant": "Initialize with a constant",
    }


    def _mw_stable_digest(value):
        payload = _mw_json.dumps(value, sort_keys=True, default=str).encode("utf-8")
        return _mw_hashlib.sha256(payload).hexdigest()[:12]


    def _mw_parse_replacement_json(text, expected_type, label):
        try:
            value = _mw_json.loads(str(text).strip())
        except _mw_json.JSONDecodeError as exc:
            raise ValueError(f"{label} must be valid JSON: {exc.msg}.") from exc
        if not isinstance(value, expected_type):
            expected = "array" if expected_type is list else "object"
            raise ValueError(f"{label} must be a JSON {expected}.")
        return value


    def _mw_replacement_state_fields(section):
        fields = []
        for raw_field in dict(section or {}).get("state_fields", []) or []:
            field = dict(_mw_as_dict(raw_field))
            local_key = str(
                field.get("local_key")
                or field.get("key")
                or field.get("qualified_key")
                or ""
            )
            if not local_key:
                continue
            field["local_key"] = local_key
            fields.append(field)
        return fields


    def _mw_inspect_replacement_draft(model, source_path, draft):
        target_id = str(draft.get("target_id", "") or "")
        if not target_id:
            raise ValueError("Choose a target from the safe backend catalog.")
        constructor_args = _mw_parse_replacement_json(
            draft.get("constructor_args_text", "[]"),
            list,
            "Constructor positional arguments",
        )
        constructor_kwargs = _mw_parse_replacement_json(
            draft.get("constructor_kwargs_text", "{}"),
            dict,
            "Constructor keyword arguments",
        )
        if not callable(_mw_inspect_replacement_target):
            raise RuntimeError(
                "The loaded Model Workbench backend does not expose replacement inspection. "
                "Restart the dashboard after updating the backend."
            )
        inspection = _mw_inspect_replacement_target(
            model,
            source_path,
            target_id,
            constructor_args=constructor_args,
            constructor_kwargs=constructor_kwargs,
        )
        return dict(_mw_as_dict(inspection)), constructor_args, constructor_kwargs


    def _mw_default_state_choices(inspection):
        source_fields = {
            field["local_key"]: field
            for field in _mw_replacement_state_fields(inspection.get("source"))
        }
        suggestions = dict(inspection.get("suggested_state_mapping", {}) or {})
        choices = {}
        for field in _mw_replacement_state_fields(inspection.get("target")):
            target_key = field["local_key"]
            suggested_source = str(suggestions.get(target_key, "") or "")
            if suggested_source in source_fields:
                choices[target_key] = f"source:{suggested_source}"
            else:
                choices[target_key] = "initializer:target_default"
        return choices


    def _mw_compile_replacement_spec(source_path, draft, inspection):
        constructor_args = _mw_parse_replacement_json(
            draft.get("constructor_args_text", "[]"),
            list,
            "Constructor positional arguments",
        )
        constructor_kwargs = _mw_parse_replacement_json(
            draft.get("constructor_kwargs_text", "{}"),
            dict,
            "Constructor keyword arguments",
        )
        source_fields = {
            field["local_key"]: field
            for field in _mw_replacement_state_fields(inspection.get("source"))
        }
        target_fields = _mw_replacement_state_fields(inspection.get("target"))
        choices = dict(_mw_default_state_choices(inspection))
        choices.update(dict(draft.get("state_choices", {}) or {}))
        constant_values = dict(draft.get("constant_values", {}) or {})
        state_mapping = {}
        state_initializers = {}
        for target_field in target_fields:
            target_key = target_field["local_key"]
            choice = str(choices.get(target_key, "") or "")
            if choice.startswith("source:"):
                source_key = choice.split(":", 1)[1]
                if source_key not in source_fields:
                    raise ValueError(
                        f"State field {target_key!r} selects unavailable source field "
                        f"{source_key!r}."
                    )
                state_mapping[target_key] = source_key
            elif choice == "initializer:constant":
                state_initializers[target_key] = {
                    "kind": "constant",
                    "value": float(constant_values.get(target_key, 0.0)),
                }
            elif choice.startswith("initializer:"):
                initializer = choice.split(":", 1)[1]
                if initializer not in {"target_default", "zeros", "ones"}:
                    raise ValueError(
                        f"State field {target_key!r} has unknown initializer {initializer!r}."
                    )
                state_initializers[target_key] = initializer
            else:
                raise ValueError(
                    f"Choose a source field or explicit initializer for {target_key!r}."
                )

        unconfirmed_spec = {
            "target_id": str(draft.get("target_id", "") or ""),
            "constructor_args": constructor_args,
            "constructor_kwargs": constructor_kwargs,
            "state_mapping": state_mapping,
            "state_initializers": state_initializers,
        }
        fingerprint = _mw_stable_digest({
            "source_path": source_path,
            "spec": unconfirmed_spec,
        })
        spec = dict(unconfirmed_spec)
        spec["confirmed"] = draft.get("confirmed_fingerprint") == fingerprint
        return spec, fingerprint


    def _mw_replacement_compatibility_rows(inspection, draft):
        source_fields = {
            field["local_key"]: field
            for field in _mw_replacement_state_fields(inspection.get("source"))
        }
        choices = dict(_mw_default_state_choices(inspection))
        choices.update(dict(draft.get("state_choices", {}) or {}))
        rows = []
        for target_field in _mw_replacement_state_fields(inspection.get("target")):
            target_key = target_field["local_key"]
            target_shape = list(target_field.get("shape", []) or [])
            target_dtype = str(target_field.get("dtype", "") or "")
            choice = str(choices.get(target_key, "") or "")
            source_field = None
            if choice.startswith("source:"):
                source_field = source_fields.get(choice.split(":", 1)[1])
            if source_field is None:
                selected_shape = target_shape
                selected_dtype = target_dtype
                compatibility = "Target initializer"
                selected = _MW_REPLACEMENT_INITIALIZERS.get(choice, choice)
            else:
                selected_shape = list(source_field.get("shape", []) or [])
                selected_dtype = str(source_field.get("dtype", "") or "")
                shape_ok = selected_shape == target_shape
                dtype_ok = selected_dtype == target_dtype
                compatibility = (
                    "Compatible"
                    if shape_ok and dtype_ok
                    else "Shape mismatch" if not shape_ok else "Dtype conversion"
                )
                selected = f"Source · {source_field['local_key']}"
            rows.append({
                "Target state": target_key,
                "Target shape": str(target_shape),
                "Target dtype": target_dtype,
                "Selected source / initializer": selected,
                "Selected shape": str(selected_shape),
                "Selected dtype": selected_dtype,
                "Compatibility preview": compatibility,
            })
        return rows


    def _mw_recipe_with_replacements(conversion_result, plan, replacement_specs):
        recipe = _mw_copy.deepcopy(
            _mw_field(conversion_result, "recipe", _mw_as_dict(plan))
        )
        if not isinstance(recipe, dict):
            recipe = dict(_mw_as_dict(plan))
        if replacement_specs:
            recipe["replacement_specs"] = _mw_copy.deepcopy(replacement_specs)
        return recipe


    def _mw_graph_payload(graph, side):
        graph = _mw_as_dict(graph)
        raw_nodes = list(graph.get("nodes", []) or [])
        raw_edges = list(graph.get("edges", []) or [])
        node_ids = {
            str(_mw_field(node, "id", ""))
            for node in raw_nodes
            if _mw_field(node, "id", None) is not None
        }
        nodes = []
        for raw_node in raw_nodes:
            node = _mw_as_dict(raw_node)
            node_id = str(node.get("id", ""))
            if not node_id:
                continue
            status = str(node.get("status", "structural") or "structural")
            parent = node.get("parent")
            data = dict(node)
            data.update({
                "id": node_id,
                "label": str(
                    node.get("label")
                    or node.get("module_type")
                    or node.get("target")
                    or node_id
                ),
                "status": status,
                "side": side,
                "reason": str(node.get("reason", "") or ""),
                "module_path": str(node.get("module_path", "") or ""),
                "module_type": str(node.get("module_type", "") or ""),
                "op": str(node.get("op", "") or ""),
                "target": str(node.get("target", "") or ""),
            })
            if parent is not None and str(parent) in node_ids:
                data["parent"] = str(parent)
            else:
                data.pop("parent", None)
            is_group = bool(
                node.get("is_group")
                or node.get("type") in {"group", "compound", "module_group"}
                or node.get("op") == "module_group"
                or node.get("kind") == "module_group"
            )
            classes = [side, status]
            if is_group:
                classes.append("module-group")
            nodes.append({"data": data, "classes": " ".join(classes)})

        edges = []
        for index, raw_edge in enumerate(raw_edges):
            edge = _mw_as_dict(raw_edge)
            source = str(edge.get("source", ""))
            target = str(edge.get("target", ""))
            if source not in node_ids or target not in node_ids:
                continue
            data = dict(edge)
            data.update({
                "id": str(edge.get("id") or f"{side}-edge-{index}"),
                "source": source,
                "target": target,
            })
            edges.append({"data": data, "classes": f"{side} data-flow"})
        return nodes + edges


    def _mw_filter_graph(graph, detail):
        graph = _mw_as_dict(graph)
        raw_nodes = [dict(_mw_as_dict(node)) for node in graph.get("nodes", []) or []]
        if detail == "Full graph":
            return {
                "nodes": raw_nodes,
                "edges": [dict(_mw_as_dict(edge)) for edge in graph.get("edges", []) or []],
            }

        def is_group(node):
            return bool(
                node.get("is_group")
                or node.get("kind") == "module_group"
                or node.get("op") == "module_group"
            )

        if detail == "Modules only":
            nodes = [node for node in raw_nodes if is_group(node)]
        else:
            attention_statuses = {
                "custom_expanded",
                "mixed_quantized_fp32",
                "fp32_fallback",
                "unsupported",
                "native_alias",
                "user_replacement",
            }
            nodes = [
                node
                for node in raw_nodes
                if (
                    is_group(node) and str(node.get("status", "")) in attention_statuses
                )
                or (
                    not is_group(node)
                    and str(node.get("status", ""))
                    not in {"structural_passthrough", "input", "output"}
                )
            ]

        node_ids = {str(node.get("id")) for node in nodes}
        edges = [
            dict(_mw_as_dict(edge))
            for edge in graph.get("edges", []) or []
            if str(_mw_field(edge, "source", "")) in node_ids
            and str(_mw_field(edge, "target", "")) in node_ids
        ]
        if detail == "Modules only":
            existing = {
                (str(edge.get("source")), str(edge.get("target"))) for edge in edges
            }
            for node in nodes:
                parent = node.get("parent")
                relation = (str(parent), str(node.get("id")))
                if parent in node_ids and relation not in existing:
                    edges.append({
                        "source": relation[0],
                        "target": relation[1],
                        "kind": "contains",
                    })
        return {"nodes": nodes, "edges": edges}


    def _mw_filter_mappings(mappings, source_graph, target_graph):
        source_ids = {str(node.get("id")) for node in source_graph.get("nodes", [])}
        target_ids = {str(node.get("id")) for node in target_graph.get("nodes", [])}
        filtered = []
        for raw_mapping in mappings or []:
            mapping = dict(_mw_as_dict(raw_mapping))
            sources = [
                str(node_id)
                for node_id in mapping.get("source_node_ids", [])
                if str(node_id) in source_ids
            ]
            targets = [
                str(node_id)
                for node_id in mapping.get("target_node_ids", [])
                if str(node_id) in target_ids
            ]
            if sources and targets:
                mapping["source_node_ids"] = sources
                mapping["target_node_ids"] = targets
                filtered.append(mapping)
        return filtered


    def _mw_type_overview(source_graph, target_graph, mappings):
        """Build a strict original-layer-type → converted-layer-type map.

        Only module groups appear on the reference side. Internal FX calls,
        tensor bookkeeping, and data-flow edges are intentionally excluded;
        they remain available in the detailed graph modes. A custom module may
        still point to several target types through its explicit decomposition
        mappings.
        """

        source_nodes = {
            str(node.get("id")): dict(_mw_as_dict(node))
            for node in _mw_as_dict(source_graph).get("nodes", []) or []
        }
        target_nodes = {
            str(node.get("id")): dict(_mw_as_dict(node))
            for node in _mw_as_dict(target_graph).get("nodes", []) or []
        }

        def mapping_ids(mapping, field):
            values = mapping.get(field, [])
            if isinstance(values, str):
                values = [values]
            return {str(value) for value in values or []}

        def is_group(node):
            return bool(
                node.get("is_group")
                or node.get("kind") == "module_group"
                or node.get("op") == "module_group"
            )

        def clean_type_name(value):
            name = str(value or "").strip()
            for prefix in ("Expanded · ", "Alias · ", "FP32 · "):
                name = name.removeprefix(prefix)
            return name or "Unknown"

        def combined_status(statuses, *, target=False):
            normalized = {str(status) for status in statuses}
            if normalized & {"unsupported"}:
                return "unsupported"
            if normalized & {"fp32_fallback", "mixed_quantized_fp32"}:
                return "fp32_fallback"
            if normalized & {"custom_expanded", "decomposed"}:
                return "custom_expanded"
            if normalized & {"user_replacement"}:
                return "user_replacement"
            return "proposed_quantized" if target else "exact_native_support"

        source_groups = {
            node_id: node
            for node_id, node in source_nodes.items()
            if is_group(node)
            and str(node.get("module_path", ""))
            and str(node.get("status", "")) != "structural_passthrough"
        }
        source_types = {}
        source_type_by_id = {}
        for node_id, node in source_groups.items():
            source_type = clean_type_name(
                node.get("module_type") or node.get("label") or node.get("target")
            )
            source_type_by_id[node_id] = source_type
            entry = source_types.setdefault(
                source_type,
                {
                    "ids": set(),
                    "statuses": set(),
                    "reasons": set(),
                },
            )
            entry["ids"].add(node_id)
            entry["statuses"].add(str(node.get("status", "")))
            if node.get("reason"):
                entry["reasons"].add(str(node["reason"]))

        normalized_mappings = [
            dict(_mw_as_dict(raw_mapping)) for raw_mapping in mappings or []
        ]
        mappings_by_source = {}
        for mapping in normalized_mappings:
            for source_id in mapping_ids(mapping, "source_node_ids"):
                mappings_by_source.setdefault(source_id, []).append(mapping)

        def intrinsic_target_type(source_node, target_node):
            """Return type support before an ancestor FP32 choice is applied."""

            candidate = source_node.get("quantized_target")
            if not candidate:
                recommended = str(source_node.get("recommended", "") or "")
                if recommended.startswith(("Quant", "Decomposed", "Observed")):
                    candidate = recommended
            if not candidate:
                for field in ("quantized_target", "module_type", "target", "label"):
                    value = clean_type_name(target_node.get(field))
                    value = value.rsplit(".", 1)[-1]
                    if value.startswith(("Quant", "Decomposed", "Observed")):
                        candidate = value
                        break
            if not candidate:
                return None
            return clean_type_name(candidate).rsplit(".", 1)[-1]

        nonstructural_direct_ops = {}
        for source_id, node in source_nodes.items():
            if is_group(node):
                continue
            status = str(node.get("status", ""))
            if status in {"structural_passthrough", "input", "output"}:
                continue
            owner_path = str(node.get("module_path", ""))
            if owner_path:
                nonstructural_direct_ops.setdefault(owner_path, {})[source_id] = node

        # A custom composite can be conservatively selected as one FP32 island
        # because only one of its direct operations is unresolved. Descendant
        # Conv/Dropout/etc. mappings are still intrinsically supported and must
        # not be presented as if those layer types themselves require FP32.
        composite_fallback_roots = {}

        def source_has_intrinsic_target(source_id, source_node):
            for mapping in mappings_by_source.get(source_id, []):
                for target_id in mapping_ids(mapping, "target_node_ids"):
                    target_node = target_nodes.get(target_id)
                    if (
                        target_node is not None
                        and intrinsic_target_type(source_node, target_node) is not None
                    ):
                        return True
            return False

        for source_id, node in source_groups.items():
            path = str(node.get("module_path", ""))
            status = str(node.get("status", ""))
            has_fallback_mapping = any(
                str(mapping.get("kind", "")) == "fp32_fallback"
                for mapping in mappings_by_source.get(source_id, [])
            )
            direct_ops = nonstructural_direct_ops.get(path, {})
            has_unresolved_direct_op = any(
                not source_has_intrinsic_target(operation_id, operation_node)
                for operation_id, operation_node in direct_ops.items()
            )
            has_supported_component = any(
                source_has_intrinsic_target(operation_id, operation_node)
                for operation_id, operation_node in direct_ops.items()
            ) or any(
                source_has_intrinsic_target(child_id, child_node)
                for child_id, child_node in source_groups.items()
                if str(child_node.get("module_path", "")).startswith(path + ".")
            )
            if (
                status in {"fp32_fallback", "unsupported", "mixed_quantized_fp32"}
                and has_fallback_mapping
                and has_unresolved_direct_op
                and has_supported_component
            ):
                composite_fallback_roots[path] = source_id

        def containing_composite_root(path):
            candidates = [
                root_path
                for root_path in composite_fallback_roots
                if path != root_path and path.startswith(root_path + ".")
            ]
            return max(candidates, key=len) if candidates else None

        composite_fallback_root_ids = set(composite_fallback_roots.values())

        def target_type_for(node, source_type):
            status = str(node.get("status", ""))
            label = str(node.get("label", ""))
            if status == "structural_passthrough":
                return None
            if status == "custom_expanded" and label.startswith("Expanded · "):
                # This is a preview-only container. Its concrete converted
                # children arrive through a separate decomposition mapping.
                return None
            if status in {"fp32_fallback", "unsupported"}:
                return f"FP32 / {source_type}"
            target_type = clean_type_name(
                node.get("module_type")
                or node.get("quantized_target")
                or node.get("target")
                or label
            )
            if target_type == "Identity" and status == "proposed_quantized":
                # Composite preview builders use Identity nodes as no-op
                # placeholders. They are not quantized layers.
                return None
            return target_type

        def target_signature(node):
            status = str(node.get("status", ""))
            if status == "structural_passthrough":
                return None
            label = str(node.get("label", ""))
            if status == "custom_expanded" and label.startswith("Expanded · "):
                return None
            target_type = clean_type_name(
                node.get("module_type")
                or node.get("quantized_target")
                or node.get("target")
                or label
            )
            if target_type == "Identity" and status == "proposed_quantized":
                return None
            return (
                str(node.get("module_path", "")),
                target_type,
                status,
            )

        # Preview graphs can contain both a target module group and its FX call
        # at the same path. Canonicalize only operation→group twins. Never
        # merge operation→operation matches: two residual QuantAdd calls owned
        # by one transformer block are two real target operations.
        target_group_by_signature = {}
        for target_id, target_node in target_nodes.items():
            signature = target_signature(target_node)
            if signature is not None and is_group(target_node):
                target_group_by_signature.setdefault(signature, target_id)

        def canonical_target_id(target_id):
            target_node = target_nodes.get(target_id)
            if target_node is None or is_group(target_node):
                return target_id
            signature = target_signature(target_node)
            return target_group_by_signature.get(signature, target_id)

        target_types = {}
        edge_groups = {}
        mapped_source_ids = set()

        def record_type_mapping(
            *,
            source_id,
            source_type,
            target_id,
            target_type,
            target_status,
            kind,
            reason,
        ):
            mapped_source_ids.add(source_id)
            target_entry = target_types.setdefault(
                target_type,
                {"ids": set(), "statuses": set(), "reasons": set()},
            )
            target_entry["ids"].add(target_id)
            target_entry["statuses"].add(str(target_status))
            if reason:
                target_entry["reasons"].add(str(reason))
            edge_entry = edge_groups.setdefault(
                (source_type, target_type),
                {"pairs": set(), "kinds": set(), "reasons": set()},
            )
            edge_entry["pairs"].add((source_id, target_id))
            edge_entry["kinds"].add(str(kind))
            if reason:
                edge_entry["reasons"].add(str(reason))

        for mapping in normalized_mappings:
            source_ids = mapping_ids(mapping, "source_node_ids") & set(source_groups)
            if not source_ids:
                continue
            mapping_kind = str(mapping.get("kind", "mapped"))
            for source_id in source_ids:
                if source_id in composite_fallback_root_ids:
                    # The parent receives operation-specific decomposition
                    # arrows below instead of a misleading generic FP32 arrow.
                    continue
                source_node = source_groups[source_id]
                source_type = source_type_by_id[source_id]
                inherited_root = containing_composite_root(
                    str(source_node.get("module_path", ""))
                )
                for target_id in mapping_ids(mapping, "target_node_ids"):
                    target_id = canonical_target_id(target_id)
                    target_node = target_nodes.get(target_id)
                    if target_node is None:
                        continue
                    intrinsic_type = None
                    if inherited_root is not None:
                        intrinsic_type = intrinsic_target_type(source_node, target_node)
                    if intrinsic_type is not None:
                        target_type = intrinsic_type
                        resolved_status = "proposed_quantized"
                        resolved_kind = "one_to_one"
                        reason = str(
                            source_node.get("reason")
                            or f"{source_type} has intrinsic QBench support."
                        )
                    else:
                        target_type = target_type_for(target_node, source_type)
                        resolved_status = str(target_node.get("status", ""))
                        resolved_kind = mapping_kind
                        reason = str(
                            mapping.get("reason") or target_node.get("reason", "")
                        )
                    if target_type is None:
                        continue
                    record_type_mapping(
                        source_id=source_id,
                        source_type=source_type,
                        target_id=target_id,
                        target_type=target_type,
                        target_status=resolved_status,
                        kind=resolved_kind,
                        reason=reason,
                    )

        # Attribute a composite's supported internal modules/operations and its
        # genuinely unresolved operations to the composite type itself. This
        # expresses `LinearSelfAttention -> QuantConv2d + ... + FP32/sum`
        # without falsely claiming `Conv2d -> FP32`.
        for root_path, root_id in composite_fallback_roots.items():
            root_type = source_type_by_id[root_id]

            descendant_groups = {
                source_id: node
                for source_id, node in source_groups.items()
                if str(node.get("module_path", "")).startswith(root_path + ".")
            }
            for child_id, child_node in descendant_groups.items():
                for mapping in mappings_by_source.get(child_id, []):
                    for target_id in mapping_ids(mapping, "target_node_ids"):
                        target_id = canonical_target_id(target_id)
                        target_node = target_nodes.get(target_id)
                        if target_node is None:
                            continue
                        target_type = intrinsic_target_type(child_node, target_node)
                        if target_type is None:
                            continue
                        record_type_mapping(
                            source_id=root_id,
                            source_type=root_type,
                            target_id=target_id,
                            target_type=target_type,
                            target_status="proposed_quantized",
                            kind="decomposed",
                            reason=(
                                f"{root_type} contains supported "
                                f"{clean_type_name(child_node.get('module_type'))} layers. "
                                "Choose `expand` on the composite to materialize them "
                                "separately from its FP32 operation."
                            ),
                        )

            for operation_id, operation_node in nonstructural_direct_ops.get(
                root_path, {}
            ).items():
                operation_label = clean_type_name(
                    operation_node.get("label")
                    or operation_node.get("target")
                    or operation_node.get("op")
                )
                for mapping in mappings_by_source.get(operation_id, []):
                    for target_id in mapping_ids(mapping, "target_node_ids"):
                        target_id = canonical_target_id(target_id)
                        target_node = target_nodes.get(target_id)
                        if target_node is None:
                            continue
                        target_type = intrinsic_target_type(
                            operation_node, target_node
                        )
                        if target_type is not None:
                            target_status = "proposed_quantized"
                            mapping_kind = "decomposed"
                            operation_reason = str(
                                operation_node.get("reason")
                                or f"{operation_label} has QBench support."
                            )
                            reason = (
                                f"{operation_reason} Choose `expand` on {root_type} to "
                                "materialize this supported operation separately."
                            )
                        else:
                            target_type = f"FP32 / {operation_label}"
                            target_status = "fp32_fallback"
                            mapping_kind = "fp32_fallback"
                            reason = str(
                                operation_node.get("reason")
                                or f"{operation_label} remains in FP32."
                            )
                        record_type_mapping(
                            source_id=root_id,
                            source_type=root_type,
                            target_id=target_id,
                            target_type=target_type,
                            target_status=target_status,
                            kind=mapping_kind,
                            reason=reason,
                        )

        # Every displayed reference layer type must have an outgoing arrow. A
        # missing preview mapping is explicit instead of silently disappearing.
        for source_id, source_node in source_groups.items():
            if source_id in mapped_source_ids:
                continue
            source_type = source_type_by_id[source_id]
            target_type = f"Unresolved / {source_type}"
            target_entry = target_types.setdefault(
                target_type,
                {"ids": set(), "statuses": set(), "reasons": set()},
            )
            synthetic_target_id = f"unresolved:{source_id}"
            target_entry["ids"].add(synthetic_target_id)
            target_entry["statuses"].add("unsupported")
            edge_entry = edge_groups.setdefault(
                (source_type, target_type),
                {"pairs": set(), "kinds": set(), "reasons": set()},
            )
            edge_entry["pairs"].add((source_id, synthetic_target_id))
            edge_entry["kinds"].add("unsupported")
            edge_entry["reasons"].add(
                str(source_node.get("reason", "No converted target was proposed."))
            )

        source_summary_nodes = []
        source_summary_id = {}
        for index, type_name in enumerate(sorted(source_types, key=str.casefold)):
            entry = source_types[type_name]
            node_id = f"type:source:{index}"
            source_summary_id[type_name] = node_id
            count = len(entry["ids"])
            source_summary_nodes.append({
                "id": node_id,
                "label": f"{type_name} × {count}",
                "module_type": type_name,
                "kind": "layer_type",
                "op": "layer_type",
                "status": combined_status(entry["statuses"]),
                "reason": f"{count} reference layer instance(s).",
                "count": count,
                "side": "source",
            })

        target_summary_nodes = []
        target_summary_id = {}
        for index, type_name in enumerate(sorted(target_types, key=str.casefold)):
            entry = target_types[type_name]
            node_id = f"type:target:{index}"
            target_summary_id[type_name] = node_id
            count = len(entry["ids"])
            target_summary_nodes.append({
                "id": node_id,
                "label": f"{type_name} × {count}",
                "module_type": type_name,
                "kind": "layer_type",
                "op": "layer_type",
                "status": combined_status(entry["statuses"], target=True),
                "reason": f"{count} proposed converted layer instance(s).",
                "count": count,
                "side": "target",
            })

        summary_mappings = []
        for (source_type, target_type), entry in sorted(
            edge_groups.items(), key=lambda item: (item[0][0].casefold(), item[0][1].casefold())
        ):
            kinds = set(entry["kinds"])
            if "unsupported" in kinds:
                kind = "unsupported"
            elif "fp32_fallback" in kinds:
                kind = "fp32_fallback"
            elif "decomposed" in kinds:
                kind = "decomposed"
            elif "user_replacement" in kinds:
                kind = "user_replacement"
            elif "many_to_one_alias" in kinds or "native_alias" in kinds:
                kind = "native_alias"
            else:
                kind = "one_to_one"
            count = len(entry["pairs"])
            summary_mappings.append({
                "source_node_ids": [source_summary_id[source_type]],
                "target_node_ids": [target_summary_id[target_type]],
                "source_type": source_type,
                "target_type": target_type,
                "kind": kind,
                "count": count,
                "reason": " · ".join(sorted(entry["reasons"]))[:600],
            })

        return (
            {"nodes": source_summary_nodes, "edges": []},
            {"nodes": target_summary_nodes, "edges": []},
            summary_mappings,
        )


    def _mw_render_type_mapping_overview(source_graph, target_graph, mappings, key):
        """Render the strict layer-type map as one persistent bipartite graph."""

        elements = []
        for element in _mw_graph_payload(source_graph, "source"):
            if "source" not in element.get("classes", "").split():
                continue
            element = dict(element)
            element["classes"] = f"{element.get('classes', '')} source-type"
            elements.append(element)
        for element in _mw_graph_payload(target_graph, "target"):
            if "target" not in element.get("classes", "").split():
                continue
            element = dict(element)
            element["classes"] = f"{element.get('classes', '')} target-type"
            elements.append(element)

        edge_index = 0
        for raw_mapping in mappings or []:
            mapping = dict(_mw_as_dict(raw_mapping))
            source_ids = mapping.get("source_node_ids", [])
            target_ids = mapping.get("target_node_ids", [])
            if isinstance(source_ids, str):
                source_ids = [source_ids]
            if isinstance(target_ids, str):
                target_ids = [target_ids]
            count = int(mapping.get("count", 0) or 0)
            kind = str(mapping.get("kind", "mapped"))
            for source_id in source_ids or []:
                for target_id in target_ids or []:
                    elements.append({
                        "data": {
                            "id": f"type-map-edge-{edge_index}",
                            "source": str(source_id),
                            "target": str(target_id),
                            "label": f"×{count}",
                            "count": count,
                            "kind": kind,
                            "reason": str(mapping.get("reason", "") or ""),
                            "source_type": str(mapping.get("source_type", "") or ""),
                            "target_type": str(mapping.get("target_type", "") or ""),
                        },
                        "classes": f"conversion-edge {kind}",
                    })
                    edge_index += 1

        def _safe_json(value):
            return _mw_json.dumps(
                value, separators=(",", ":"), default=str
            ).replace("</", "<\\/")

        html_template = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.26.0/cytoscape.min.js"></script>
  <script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.min.js"></script>
  <style>
    :root { color-scheme: light; }
    html, body { margin: 0; padding: 0; background: #f8fafc; color: #0f172a; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
    #shell { position: relative; height: 660px; overflow: hidden; border: 1px solid #cbd5e1; border-radius: 14px; background: radial-gradient(circle at 12% 10%, #eff6ff 0, transparent 32%), radial-gradient(circle at 88% 10%, #ecfdf5 0, transparent 32%), #f8fafc; }
    #cy { position: absolute; inset: 46px 0 106px; }
    .column-title { position: absolute; z-index: 4; top: 13px; padding: 6px 12px; border-radius: 999px; font-size: 11px; font-weight: 800; letter-spacing: .04em; pointer-events: none; }
    #source-title { left: 18px; color: #1e40af; border: 1px solid #bfdbfe; background: #dbeafe; }
    #target-title { right: 18px; color: #166534; border: 1px solid #bbf7d0; background: #dcfce7; }
    #toolbar { position: absolute; z-index: 5; top: 11px; left: 50%; display: flex; gap: 7px; transform: translateX(-50%); }
    button { cursor: pointer; border: 1px solid #cbd5e1; border-radius: 8px; padding: 6px 10px; color: #334155; background: rgba(255,255,255,.96); font-weight: 700; font-size: 11px; box-shadow: 0 2px 8px rgba(15,23,42,.07); }
    button:hover { background: #eff6ff; border-color: #93c5fd; }
    #details { position: absolute; z-index: 5; left: 14px; right: 14px; bottom: 12px; min-height: 72px; display: grid; grid-template-columns: minmax(180px,.75fr) minmax(240px,1.25fr); gap: 14px; box-sizing: border-box; padding: 10px 13px; border: 1px solid #cbd5e1; border-radius: 12px; background: rgba(255,255,255,.97); box-shadow: 0 8px 24px rgba(15,23,42,.08); }
    .detail-block { min-width: 0; overflow: hidden; }
    .detail-kicker { color: #64748b; text-transform: uppercase; letter-spacing: .08em; font-size: 9px; font-weight: 800; }
    .detail-value { margin-top: 4px; max-height: 52px; overflow: auto; font-size: 12px; line-height: 1.3; overflow-wrap: anywhere; }
    .hint { color: #64748b; }
  </style>
</head>
<body>
<div id="shell" data-analysis-key="__KEY__" data-conversion-edge-count="__EDGE_COUNT__">
  <div id="source-title" class="column-title">ORIGINAL LAYER TYPES</div>
  <div id="target-title" class="column-title">CONVERTED LAYER TYPES</div>
  <div id="toolbar"><button id="fit">Fit map</button><button id="show-all">Show all</button></div>
  <div id="cy"></div>
  <div id="details">
    <div class="detail-block"><div class="detail-kicker">Selection</div><div id="selection" class="detail-value hint">All conversion arrows are shown. Select a layer type or arrow to focus it.</div></div>
    <div class="detail-block"><div class="detail-kicker">Conversion</div><div id="conversion" class="detail-value hint">Arrow counts are concrete layer/operation instances, not model data-flow connections.</div></div>
  </div>
</div>
<script>
(() => {
  const elements = __ELEMENTS__;
  const shell = document.getElementById('shell');
  if (typeof window.cytoscape !== 'function') {
    shell.innerHTML = '<div style="margin:24px;padding:18px;border:1px solid #f59e0b;border-radius:12px;background:#fffbeb;color:#92400e;font:14px system-ui">The layer-type map assets could not be loaded. The exact conversion table remains available below.</div>';
    return;
  }
  const selectionEl = document.getElementById('selection');
  const conversionEl = document.getElementById('conversion');
  const style = [
    { selector: 'node', style: {
      'label': 'data(label)', 'shape': 'round-rectangle', 'width': 'label', 'height': 'label',
      'padding': '11px', 'font-size': '11px', 'font-weight': 700, 'text-wrap': 'wrap',
      'text-max-width': '180px', 'text-valign': 'center', 'text-halign': 'center',
      'color': '#0f172a', 'border-width': 1.5,
      'transition-property': 'opacity, border-width, border-color', 'transition-duration': '120ms'
    }},
    { selector: 'node.source-type', style: { 'background-color': '#dbeafe', 'border-color': '#3b82f6' }},
    { selector: 'node.target-type', style: { 'background-color': '#dcfce7', 'border-color': '#22c55e' }},
    { selector: 'node.target-type[status *= "custom"]', style: { 'background-color': '#e0e7ff', 'border-color': '#6366f1' }},
    { selector: 'node.target-type[status *= "decompos"]', style: { 'background-color': '#e0e7ff', 'border-color': '#6366f1' }},
    { selector: 'node.target-type[status = "user_replacement"]', style: { 'background-color': '#ccfbf1', 'border-color': '#0f766e' }},
    { selector: 'node.target-type[status *= "fallback"]', style: { 'background-color': '#fef3c7', 'border-color': '#f59e0b' }},
    { selector: 'node.target-type[status *= "unsupported"]', style: { 'background-color': '#ffe4e6', 'border-color': '#f43f5e' }},
    { selector: 'edge', style: {
      'label': 'data(label)', 'font-size': '9px', 'font-weight': 700, 'color': '#475569',
      'text-background-color': '#f8fafc', 'text-background-opacity': .92, 'text-background-padding': '2px',
      'width': 2.2, 'line-color': '#64748b', 'target-arrow-color': '#64748b',
      'target-arrow-shape': 'triangle', 'arrow-scale': 1.05, 'curve-style': 'bezier', 'opacity': .8
    }},
    { selector: 'edge.decomposed', style: { 'line-color': '#6366f1', 'target-arrow-color': '#6366f1' }},
    { selector: 'edge.user_replacement', style: { 'line-color': '#0f766e', 'target-arrow-color': '#0f766e', 'width': 3.2 }},
    { selector: 'edge.fp32_fallback', style: { 'line-color': '#d97706', 'target-arrow-color': '#d97706', 'line-style': 'dashed' }},
    { selector: 'edge.unsupported', style: { 'line-color': '#e11d48', 'target-arrow-color': '#e11d48', 'line-style': 'dashed' }},
    { selector: '.dimmed', style: { 'opacity': .11 }},
    { selector: '.focused', style: { 'opacity': 1, 'border-width': 4, 'border-color': '#7c3aed', 'line-color': '#7c3aed', 'target-arrow-color': '#7c3aed', 'width': 4 }}
  ];
  const cy = cytoscape({
    container: document.getElementById('cy'), elements, style,
    minZoom: .15, maxZoom: 3, wheelSensitivity: .18
  });
  function layout() {
    try {
      cy.layout({ name: 'dagre', rankDir: 'LR', rankSep: 240, nodeSep: 28, edgeSep: 16, padding: 38, animate: false }).run();
    } catch (_layoutError) {
      cy.layout({ name: 'breadthfirst', directed: true, circle: false, spacingFactor: 1.35, padding: 38, animate: false }).run();
    }
    cy.fit(undefined, 36);
  }
  function showAll() {
    cy.elements().removeClass('dimmed focused');
    selectionEl.textContent = 'All conversion arrows are shown. Select a layer type or arrow to focus it.';
    conversionEl.textContent = 'Arrow counts are concrete layer/operation instances, not model data-flow connections.';
  }
  function focus(elementsToFocus) {
    cy.elements().addClass('dimmed').removeClass('focused');
    elementsToFocus.removeClass('dimmed').addClass('focused');
  }
  cy.on('tap', 'node', event => {
    const node = event.target;
    const edges = node.connectedEdges();
    focus(node.union(edges).union(edges.connectedNodes()));
    const peers = edges.connectedNodes().not(node).map(peer => peer.data('module_type') || peer.data('label'));
    selectionEl.textContent = node.data('label');
    conversionEl.textContent = peers.length ? `Connected to: ${peers.join(' · ')}` : 'No converted layer type is connected.';
  });
  cy.on('tap', 'edge', event => {
    const edge = event.target;
    focus(edge.union(edge.connectedNodes()));
    selectionEl.textContent = `${edge.data('source_type')} → ${edge.data('target_type')} · ${edge.data('label')}`;
    conversionEl.textContent = edge.data('reason') || `Mapping kind: ${edge.data('kind')}`;
  });
  cy.on('tap', event => { if (event.target === cy) showAll(); });
  document.getElementById('fit').addEventListener('click', () => cy.fit(undefined, 36));
  document.getElementById('show-all').addEventListener('click', showAll);
  new ResizeObserver(() => { cy.resize(); cy.fit(undefined, 36); }).observe(shell);
  layout();
})();
</script>
</body>
</html>
"""
        html = (
            html_template
            .replace("__ELEMENTS__", _safe_json(elements))
            .replace("__KEY__", str(key))
            .replace("__EDGE_COUNT__", str(edge_index))
        )
        _mw_components.html(html, height=676, scrolling=False)


    def _mw_render_mapping_graph(source_graph, target_graph, mappings, key):
        source_elements = _mw_graph_payload(source_graph, "source")
        target_elements = _mw_graph_payload(target_graph, "target")
        mapping_payload = []
        for raw_mapping in mappings or []:
            mapping = _mw_as_dict(raw_mapping)
            source_ids = mapping.get("source_node_ids", mapping.get("sources", []))
            target_ids = mapping.get("target_node_ids", mapping.get("targets", []))
            if isinstance(source_ids, str):
                source_ids = [source_ids]
            if isinstance(target_ids, str):
                target_ids = [target_ids]
            mapping_payload.append({
                "source_node_ids": [str(node_id) for node_id in source_ids or []],
                "target_node_ids": [str(node_id) for node_id in target_ids or []],
                "kind": str(mapping.get("kind", "mapped")),
                "reason": str(mapping.get("reason", "") or ""),
            })

        def _safe_json(value):
            return _mw_json.dumps(value, separators=(",", ":"), default=str).replace("</", "<\\/")

        html_template = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.26.0/cytoscape.min.js"></script>
  <script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.min.js"></script>
  <style>
    :root { color-scheme: light; }
    html, body { margin: 0; padding: 0; background: #f8fafc; color: #0f172a; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
    #shell { position: relative; height: 800px; border: 1px solid #cbd5e1; border-radius: 14px; overflow: hidden; background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%); }
    #panels { position: absolute; inset: 0 0 112px; display: grid; grid-template-columns: 1fr 1fr; gap: 54px; padding: 50px 16px 12px; }
    .panel { position: relative; min-width: 0; border: 1px solid #dbeafe; border-radius: 12px; overflow: hidden; background: rgba(255,255,255,.88); box-shadow: 0 8px 24px rgba(15,23,42,.07); }
    .panel-title { position: absolute; z-index: 5; left: 12px; top: 10px; border-radius: 999px; padding: 5px 10px; font-size: 12px; font-weight: 800; letter-spacing: .02em; background: #e0f2fe; color: #075985; border: 1px solid #bae6fd; pointer-events: none; }
    .panel-title.target { background: #dcfce7; color: #166534; border-color: #bbf7d0; }
    .cy { position: absolute; inset: 0; }
    #mapping-lines { position: absolute; inset: 0; z-index: 8; width: 100%; height: 100%; pointer-events: none; overflow: visible; }
    #toolbar { position: absolute; right: 14px; top: 10px; z-index: 12; display: flex; gap: 7px; }
    button { cursor: pointer; border: 1px solid #93c5fd; border-radius: 8px; padding: 6px 10px; color: #1e40af; background: white; font-weight: 700; font-size: 11px; box-shadow: 0 2px 8px rgba(15,23,42,.08); }
    button:hover { background: #eff6ff; }
    #details { position: absolute; z-index: 10; left: 16px; right: 16px; bottom: 12px; height: 82px; display: grid; grid-template-columns: minmax(180px, .8fr) minmax(260px, 1.4fr) minmax(220px, 1fr); gap: 12px; border: 1px solid #cbd5e1; border-radius: 12px; background: rgba(255,255,255,.96); padding: 8px 12px; box-sizing: border-box; box-shadow: 0 8px 24px rgba(15,23,42,.08); }
    .detail-block { min-width: 0; overflow: hidden; }
    .detail-kicker { color: #64748b; text-transform: uppercase; letter-spacing: .08em; font-size: 9px; font-weight: 800; }
    .detail-value { margin-top: 3px; font-size: 12px; line-height: 1.25; overflow: auto; max-height: 54px; overflow-wrap: anywhere; }
    .hint { color: #64748b; }
    #legend { position: absolute; left: 16px; top: 10px; z-index: 11; display: flex; gap: 6px; align-items: center; font-size: 10px; color: #475569; }
    .swatch { width: 9px; height: 9px; border-radius: 3px; display: inline-block; margin-left: 5px; }
    .supported { background: #86efac; } .custom { background: #93c5fd; } .fallback { background: #fcd34d; } .unsupported { background: #fda4af; } .structural { background: #cbd5e1; }
  </style>
</head>
<body>
<div id="shell">
  <div id="legend">
    <span><i class="swatch supported"></i> supported/quantized</span>
    <span><i class="swatch custom"></i> custom expansion</span>
    <span><i class="swatch fallback"></i> FP32 fallback</span>
    <span><i class="swatch unsupported"></i> unresolved</span>
    <span><i class="swatch structural"></i> structural</span>
  </div>
  <div id="toolbar"><button id="fit">Fit both</button><button id="clear">Clear mapping</button></div>
  <div id="panels">
    <div class="panel"><div class="panel-title">Reference operations</div><div id="source-cy" class="cy"></div></div>
    <div class="panel"><div class="panel-title target">Proposed quantized operations</div><div id="target-cy" class="cy"></div></div>
  </div>
  <svg id="mapping-lines"></svg>
  <div id="details">
    <div class="detail-block"><div class="detail-kicker">Selected node</div><div id="selected" class="detail-value hint">Click a node in either graph.</div></div>
    <div class="detail-block"><div class="detail-kicker">Support decision</div><div id="reason" class="detail-value hint">Mappings appear only for the current selection to keep large models readable.</div></div>
    <div class="detail-block"><div class="detail-kicker">Maps to</div><div id="maps-to" class="detail-value hint">—</div></div>
  </div>
</div>
<script>
(() => {
  const sourceElements = __SOURCE_ELEMENTS__;
  const targetElements = __TARGET_ELEMENTS__;
  const mappings = __MAPPINGS__;
  const shell = document.getElementById('shell');
  if (typeof window.cytoscape !== 'function') {
    shell.innerHTML = '<div style="margin:24px;padding:18px;border:1px solid #f59e0b;border-radius:12px;background:#fffbeb;color:#92400e;font:14px system-ui">The interactive graph assets could not be loaded. Check browser access to the configured Cytoscape CDNs; the support table and conversion controls remain available below.</div>';
    return;
  }
  const svg = document.getElementById('mapping-lines');
  const selectedEl = document.getElementById('selected');
  const reasonEl = document.getElementById('reason');
  const mapsToEl = document.getElementById('maps-to');
  let active = null;

  const style = [
    { selector: 'node', style: {
      'label': 'data(label)', 'shape': 'round-rectangle', 'width': 'label', 'height': 'label',
      'padding': '9px', 'font-size': '10px', 'text-wrap': 'wrap', 'text-max-width': '150px',
      'text-valign': 'center', 'text-halign': 'center', 'background-color': '#cbd5e1',
      'border-width': 1.2, 'border-color': '#94a3b8', 'color': '#0f172a',
      'transition-property': 'opacity, border-width, border-color', 'transition-duration': '120ms'
    }},
    { selector: 'node[status *= "support"]', style: { 'background-color': '#86efac', 'border-color': '#16a34a' }},
    { selector: 'node[status *= "quantized"]', style: { 'background-color': '#86efac', 'border-color': '#15803d' }},
    { selector: 'node[status = "transparent_subclass"]', style: { 'background-color': '#86efac', 'border-color': '#15803d' }},
    { selector: 'node[status = "native_alias"]', style: { 'background-color': '#bbf7d0', 'border-color': '#15803d' }},
    { selector: 'node[status = "user_replacement"]', style: { 'background-color': '#99f6e4', 'border-color': '#0f766e' }},
    { selector: 'node[status *= "custom"]', style: { 'background-color': '#93c5fd', 'border-color': '#2563eb' }},
    { selector: 'node[status *= "decompos"]', style: { 'background-color': '#93c5fd', 'border-color': '#2563eb' }},
    { selector: 'node[status *= "fallback"]', style: { 'background-color': '#fcd34d', 'border-color': '#d97706' }},
    { selector: 'node[status *= "unsupported"]', style: { 'background-color': '#fda4af', 'border-color': '#e11d48' }},
    { selector: 'node.module-group', style: {
      'background-color': '#f1f5f9', 'background-opacity': .3, 'border-style': 'dashed',
      'border-width': 1.5, 'padding': '18px', 'text-valign': 'top', 'font-size': '11px', 'font-weight': 700
    }},
    { selector: 'edge', style: {
      'width': 1.4, 'line-color': '#94a3b8', 'target-arrow-color': '#94a3b8',
      'target-arrow-shape': 'triangle', 'curve-style': 'bezier', 'opacity': .72
    }},
    { selector: '.dimmed', style: { 'opacity': .16 }},
    { selector: '.mapped', style: { 'opacity': 1, 'border-width': 4, 'border-color': '#7c3aed' }}
  ];

  function makeCy(container, elements) {
    const cy = cytoscape({ container, elements, style, minZoom: .05, maxZoom: 3, wheelSensitivity: .18 });
    try {
      cy.layout({ name: 'dagre', rankDir: 'LR', nodeSep: 24, rankSep: 46, edgeSep: 12, padding: 44, animate: false }).run();
    } catch (_layoutError) {
      cy.layout({ name: 'breadthfirst', directed: true, spacingFactor: 1.15, padding: 44, animate: false }).run();
    }
    cy.fit(undefined, 40);
    return cy;
  }
  const sourceCy = makeCy(document.getElementById('source-cy'), sourceElements);
  const targetCy = makeCy(document.getElementById('target-cy'), targetElements);

  function mappingFor(side, id) {
    return mappings.filter(m => (side === 'source' ? m.source_node_ids : m.target_node_ids).includes(id));
  }
  function labelFor(cy, id) {
    const node = cy.getElementById(id);
    return node && node.length ? node.data('label') : id;
  }
  function clearClasses() {
    sourceCy.elements().removeClass('dimmed mapped');
    targetCy.elements().removeClass('dimmed mapped');
  }
  function linePoint(cy, id) {
    const node = cy.getElementById(id);
    if (!node || !node.length || !node.visible()) return null;
    const panelRect = cy.container().getBoundingClientRect();
    const shellRect = shell.getBoundingClientRect();
    const pos = node.renderedPosition();
    return { x: panelRect.left - shellRect.left + pos.x, y: panelRect.top - shellRect.top + pos.y };
  }
  function drawLines() {
    svg.replaceChildren();
    if (!active) return;
    const selectedMappings = mappingFor(active.side, active.id);
    let emitted = 0;
    selectedMappings.forEach((mapping, mapIndex) => {
      mapping.source_node_ids.forEach(sourceId => mapping.target_node_ids.forEach(targetId => {
        if (emitted++ > 60) return;
        const a = linePoint(sourceCy, sourceId);
        const b = linePoint(targetCy, targetId);
        if (!a || !b) return;
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        const bend = Math.max(30, (b.x - a.x) * .45);
        path.setAttribute('d', `M ${a.x} ${a.y} C ${a.x + bend} ${a.y}, ${b.x - bend} ${b.y}, ${b.x} ${b.y}`);
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', mapIndex % 2 ? '#0ea5e9' : '#7c3aed');
        path.setAttribute('stroke-width', '2.2');
        path.setAttribute('stroke-dasharray', mapping.kind.includes('fallback') ? '6 5' : 'none');
        path.setAttribute('opacity', '.82');
        svg.appendChild(path);
      }));
    });
  }
  function select(side, node) {
    active = { side, id: node.id() };
    clearClasses();
    const selectedMappings = mappingFor(side, node.id());
    const sourceIds = new Set(selectedMappings.flatMap(m => m.source_node_ids));
    const targetIds = new Set(selectedMappings.flatMap(m => m.target_node_ids));
    if (selectedMappings.length) {
      sourceCy.nodes().addClass('dimmed'); targetCy.nodes().addClass('dimmed');
      sourceIds.forEach(id => sourceCy.getElementById(id).removeClass('dimmed').addClass('mapped'));
      targetIds.forEach(id => targetCy.getElementById(id).removeClass('dimmed').addClass('mapped'));
    } else {
      node.addClass('mapped');
    }
    const data = node.data();
    selectedEl.textContent = `${data.label} · ${data.op || data.module_type || 'node'}${data.module_path ? ` · ${data.module_path}` : ''}`;
    const reasons = selectedMappings.map(m => m.reason).filter(Boolean);
    reasonEl.textContent = data.reason || reasons.join(' · ') || `Status: ${data.status || 'structural'}`;
    const opposite = side === 'source'
      ? [...targetIds].map(id => labelFor(targetCy, id))
      : [...sourceIds].map(id => labelFor(sourceCy, id));
    mapsToEl.textContent = opposite.length ? opposite.join(' · ') : 'No quantized mapping; remains structural or unresolved.';
    requestAnimationFrame(drawLines);
  }
  function clearSelection() {
    active = null; clearClasses(); svg.replaceChildren();
    selectedEl.textContent = 'Click a node in either graph.'; selectedEl.className = 'detail-value hint';
    reasonEl.textContent = 'Mappings appear only for the current selection to keep large models readable.';
    mapsToEl.textContent = '—';
  }
  sourceCy.on('tap', 'node', event => select('source', event.target));
  targetCy.on('tap', 'node', event => select('target', event.target));
  sourceCy.on('tap', event => { if (event.target === sourceCy) clearSelection(); });
  targetCy.on('tap', event => { if (event.target === targetCy) clearSelection(); });
  [sourceCy, targetCy].forEach(cy => cy.on('pan zoom resize position', () => requestAnimationFrame(drawLines)));
  document.getElementById('fit').addEventListener('click', () => { sourceCy.fit(undefined, 40); targetCy.fit(undefined, 40); requestAnimationFrame(drawLines); });
  document.getElementById('clear').addEventListener('click', clearSelection);
  new ResizeObserver(() => { sourceCy.resize(); targetCy.resize(); requestAnimationFrame(drawLines); }).observe(shell);
})();
</script>
</body>
</html>
"""
        html = (
            html_template
            .replace("__SOURCE_ELEMENTS__", _safe_json(source_elements))
            .replace("__TARGET_ELEMENTS__", _safe_json(target_elements))
            .replace("__MAPPINGS__", _safe_json(mapping_payload))
        )
        # ``components.html`` has no explicit widget key. Include the analysis
        # key as an inert HTML attribute so Streamlit still sees new component
        # content whenever a fresh analysis is produced.
        html = html.replace('<div id="shell">', f'<div id="shell" data-analysis-key="{key}">', 1)
        _mw_components.html(html, height=816, scrolling=False)


    @st.cache_data(show_spinner=False)
    def _mw_cached_model_names(source):
        if _MW_IMPORT_ERROR is not None:
            return []
        return list(_mw_list_model_names(source))


    def _mw_clear_derived_state():
        for state_key in (
            "_mw_quantized_model",
            "_mw_conversion_plan",
            "_mw_conversion_result",
            "_mw_inference_result",
            "_mw_export_bundle",
            "_mw_build_fingerprint",
            "_mw_validation_state",
            "_mw_dataset_benchmark",
            "_mw_dataset_benchmark_config",
            "_mw_dataset_benchmark_error",
            "_mw_dataset_benchmark_fingerprint",
        ):
            st.session_state.pop(state_key, None)


    def _mw_summary_counts(analysis):
        summary = dict(_mw_field(analysis, "summary", {}) or {})
        status_counts = summary.get("status_counts")
        if not isinstance(status_counts, dict):
            source_graph = _mw_as_dict(_mw_field(analysis, "source_graph", {}))
            status_counts = dict(_MwCounter(
                str(_mw_field(node, "status", "structural"))
                for node in source_graph.get("nodes", [])
            ))
        return summary, status_counts


    def _mw_inference_view(result):
        """Keep comparison metadata without pinning potentially huge logits."""
        data = dict(_mw_as_dict(result))
        data.pop("reference_output", None)
        data.pop("quantized_output", None)
        return data


    def _mw_validation_state(result, require_allclose):
        comparison = dict(_mw_as_dict(result).get("comparison", {}) or {})
        structure_ok = bool(comparison.get("structure_match"))
        max_error = comparison.get("max_abs_error", 0.0)
        finite = True
        try:
            finite = _mw_math.isfinite(float(max_error))
        except (TypeError, ValueError):
            finite = False
        numerical_ok = bool(comparison.get("allclose")) if require_allclose else finite
        passed = structure_ok and numerical_ok
        if not structure_ok:
            reason = "Reference and converted outputs have different structures or shapes."
        elif require_allclose and not numerical_ok:
            reason = "FP32 structural conversion changed the sample output beyond tolerance."
        elif not finite:
            reason = "Converted inference produced a non-finite numerical error."
        else:
            reason = "Sample validation passed."
        return {"status": "passed" if passed else "failed", "reason": reason}


    def _mw_render_replacement_editor(
        reference_model,
        replacement_rows,
        analysis_token,
    ):
        if not replacement_rows:
            return {}, [], True, {}

        backend_ready = all(callable(callback) for callback in (
            _mw_list_replacement_targets,
            _mw_inspect_replacement_target,
            _mw_validate_replacement_spec,
        ))
        if not backend_ready:
            message = (
                "Explicit replacement was selected, but the loaded Model Workbench backend "
                "does not expose the safe replacement catalog and validator. Restart the "
                "dashboard after updating the backend."
            )
            st.error(message)
            return {}, [], False, {"backend_error": message}

        try:
            catalog = [
                dict(_mw_as_dict(entry))
                for entry in list(_mw_list_replacement_targets() or [])
            ]
        except Exception as exc:
            message = f"The safe replacement catalog could not be loaded: {exc}"
            st.error(message)
            return {}, [], False, {"catalog_error": message}
        catalog_by_id = {
            str(entry.get("id", "")): entry
            for entry in catalog
            if str(entry.get("id", ""))
        }
        if not catalog_by_id:
            message = "The backend replacement catalog is empty; explicit replacement is disabled."
            st.error(message)
            return {}, [], False, {"catalog_error": message}

        state = st.session_state.get("_mw_replacement_drafts")
        if not isinstance(state, dict) or state.get("analysis_token") != analysis_token:
            state = {"analysis_token": analysis_token, "drafts": {}}
            st.session_state["_mw_replacement_drafts"] = state
        drafts = state.setdefault("drafts", {})

        st.markdown("##### Explicit custom replacements")
        st.warning(
            "A catalog target is not proof of equivalence. Configure state transfer, inspect "
            "shape/dtype compatibility, and explicitly confirm the exact recipe. Conversion "
            "remains locked until the backend validates every concrete source path."
        )
        st.caption(
            "Keeping target state initialized is explicit and may preserve random values. "
            "The JSON recipe records that policy; only the converted state bundle captures "
            "the realized initialized tensors bit-for-bit. Unused source state is permitted "
            "and shown for review."
        )
        with st.expander("Safe backend replacement catalog", expanded=False):
            st.dataframe(
                [
                    {
                        "Catalog ID": target_id,
                        "Target": entry.get("target_name", ""),
                        "Target type": entry.get("target_type", ""),
                        "Native counterpart": entry.get("native_name", ""),
                        "Constructor": str(entry.get("constructor_parameters", "")),
                        "Realizable": entry.get("realizable", True),
                        "Backend warnings": " · ".join(
                            str(warning)
                            for warning in entry.get("warnings", []) or []
                        ),
                    }
                    for target_id, entry in sorted(catalog_by_id.items())
                ],
                hide_index=True,
                width='stretch',
            )

        rows_by_path = {
            str(row["Path"]): row
            for row in replacement_rows
        }
        groups = {}
        for path, row in rows_by_path.items():
            groups.setdefault(str(row.get("Type", "Unknown") or "Unknown"), []).append(path)

        inspected_by_path = {}
        bulk_confirm_paths = set()
        for group_index, (source_type, raw_paths) in enumerate(sorted(groups.items())):
            paths = sorted(raw_paths)
            count = len(paths)
            with st.expander(
                f"{source_type} · {count} concrete {'path' if count == 1 else 'paths'}",
                expanded=len(groups) == 1,
            ):
                group_key = _mw_stable_digest({"token": analysis_token, "type": source_type})
                active_path = st.selectbox(
                    "Edit concrete source path",
                    paths,
                    key=f"mw_replacement_active_{group_key}",
                    help=(
                        "Each path keeps its own recipe and backend validation. Use the bulk "
                        "control below to copy this exact recipe to repeated compatible layers."
                    ),
                )
                path_key = _mw_stable_digest({"token": analysis_token, "path": active_path})
                draft = drafts.setdefault(active_path, {
                    "target_id": "",
                    "constructor_args_text": "[]",
                    "constructor_kwargs_text": "{}",
                    "state_choices": {},
                    "constant_values": {},
                })

                target_options = [""] + sorted(catalog_by_id)

                def target_label(target_id):
                    if not target_id:
                        return "Choose a safe target…"
                    entry = catalog_by_id[target_id]
                    name = str(entry.get("target_name") or target_id)
                    native = str(entry.get("native_name") or "")
                    suffix = f" · native {native}" if native else ""
                    return f"{name}{suffix} · [{target_id}]"

                current_target = str(draft.get("target_id", "") or "")
                if current_target not in target_options:
                    current_target = ""
                target_id = st.selectbox(
                    "Safe replacement target",
                    target_options,
                    index=target_options.index(current_target),
                    format_func=target_label,
                    key=f"mw_replacement_target_{path_key}",
                    help=(
                        "Only opaque IDs supplied by the backend registry are accepted. "
                        "Displayed Python type names are informational and are never imported."
                    ),
                )
                if target_id != draft.get("target_id"):
                    draft.pop("confirmed_fingerprint", None)
                    draft.pop("validated_fingerprint", None)
                    draft.pop("normalized_spec", None)
                draft["target_id"] = target_id

                constructor_cols = st.columns(2)
                args_text = constructor_cols[0].text_area(
                    "Constructor positional arguments (JSON array)",
                    value=str(draft.get("constructor_args_text", "[]")),
                    key=f"mw_replacement_args_{path_key}",
                    height=92,
                ).strip()
                kwargs_text = constructor_cols[1].text_area(
                    "Constructor keyword arguments (JSON object)",
                    value=str(draft.get("constructor_kwargs_text", "{}")),
                    key=f"mw_replacement_kwargs_{path_key}",
                    height=92,
                ).strip()
                draft["constructor_args_text"] = args_text
                draft["constructor_kwargs_text"] = kwargs_text

                inspection = None
                editor_error = None
                try:
                    inspection, _, _ = _mw_inspect_replacement_draft(
                        reference_model,
                        active_path,
                        draft,
                    )
                except Exception as exc:
                    editor_error = str(exc)
                    st.error(f"Replacement inspection for `{active_path}` failed: {exc}")
                else:
                    inspected_by_path[active_path] = inspection
                    for inspection_warning in inspection.get("warnings", []) or []:
                        st.warning(str(inspection_warning))
                    target_info = dict(inspection.get("target", {}) or {})
                    st.caption(
                        "Inspecting backend target "
                        f"`{target_info.get('type', target_info.get('target_type', target_id))}` "
                        f"for concrete source path `{active_path}`."
                    )
                    source_fields = {
                        field["local_key"]: field
                        for field in _mw_replacement_state_fields(inspection.get("source"))
                    }
                    defaults = _mw_default_state_choices(inspection)
                    stored_choices = dict(draft.get("state_choices", {}) or {})
                    context_key = _mw_stable_digest({
                        "path": active_path,
                        "target_id": target_id,
                        "constructor_args": args_text,
                        "constructor_kwargs": kwargs_text,
                    })
                    updated_choices = {}
                    constant_values = dict(draft.get("constant_values", {}) or {})
                    target_fields = _mw_replacement_state_fields(inspection.get("target"))
                    if target_fields:
                        st.markdown("**Target-state transfer**")
                    else:
                        st.info("This target has no parameters or persistent buffers to transfer.")
                    for field_index, target_field in enumerate(target_fields):
                        target_key = target_field["local_key"]
                        choice_options = list(_MW_REPLACEMENT_INITIALIZERS) + [
                            f"source:{source_key}" for source_key in sorted(source_fields)
                        ]
                        requested_choice = str(
                            stored_choices.get(target_key, defaults.get(target_key, "")) or ""
                        )
                        if requested_choice not in choice_options:
                            requested_choice = defaults.get(
                                target_key,
                                "initializer:target_default",
                            )

                        def state_choice_label(choice):
                            if choice.startswith("source:"):
                                return f"Copy source state · {choice.split(':', 1)[1]}"
                            return _MW_REPLACEMENT_INITIALIZERS.get(choice, choice)

                        choice = st.selectbox(
                            f"Target `{target_key}`",
                            choice_options,
                            index=choice_options.index(requested_choice),
                            format_func=state_choice_label,
                            key=(
                                f"mw_replacement_state_{path_key}_{context_key}_"
                                f"{field_index}"
                            ),
                            help=(
                                "Every target parameter/buffer must explicitly copy one source "
                                "field or use an initializer. Keep target initialized is an "
                                "intentional choice, not an implicit fallback."
                            ),
                        )
                        updated_choices[target_key] = choice
                        if choice == "initializer:constant":
                            constant_values[target_key] = st.number_input(
                                f"Constant value for `{target_key}`",
                                value=float(constant_values.get(target_key, 0.0)),
                                key=(
                                    f"mw_replacement_constant_{path_key}_{context_key}_"
                                    f"{field_index}"
                                ),
                            )
                    draft["state_choices"] = updated_choices
                    draft["constant_values"] = {
                        key: value
                        for key, value in constant_values.items()
                        if key in updated_choices
                    }
                    used_source_keys = {
                        choice.split(":", 1)[1]
                        for choice in updated_choices.values()
                        if choice.startswith("source:")
                    }
                    unused_source_keys = sorted(set(source_fields) - used_source_keys)
                    if unused_source_keys:
                        st.caption(
                            "Unused source state (informational): "
                            + ", ".join(f"`{key}`" for key in unused_source_keys)
                        )
                    try:
                        active_spec, active_fingerprint = _mw_compile_replacement_spec(
                            active_path,
                            draft,
                            inspection,
                        )
                    except Exception as exc:
                        editor_error = str(exc)
                        st.error(f"Replacement recipe for `{active_path}` is incomplete: {exc}")
                    else:
                        compatibility_rows = _mw_replacement_compatibility_rows(
                            inspection,
                            draft,
                        )
                        if compatibility_rows:
                            st.markdown("**Shape and dtype compatibility preview**")
                            st.dataframe(
                                compatibility_rows,
                                hide_index=True,
                                width='stretch',
                            )
                        confirmed = st.checkbox(
                            (
                                "I explicitly confirm this exact target, constructor, and "
                                f"state-transfer recipe for `{active_path}`"
                            ),
                            value=(
                                draft.get("confirmed_fingerprint") == active_fingerprint
                            ),
                            key=(
                                f"mw_replacement_confirm_{path_key}_"
                                f"{active_fingerprint}"
                            ),
                        )
                        if confirmed:
                            draft["confirmed_fingerprint"] = active_fingerprint
                        elif draft.get("confirmed_fingerprint") == active_fingerprint:
                            draft.pop("confirmed_fingerprint", None)

                        other_paths = [path for path in paths if path != active_path]
                        if other_paths:
                            st.markdown("**Apply to repeated compatible paths**")
                            selected_paths = st.multiselect(
                                "Concrete paths that should receive this exact recipe",
                                other_paths,
                                default=other_paths,
                                key=(
                                    f"mw_replacement_bulk_paths_{group_key}_{path_key}_"
                                    f"{active_fingerprint}"
                                ),
                                help=(
                                    "Each selected path receives its own spec and is independently "
                                    "inspected and validated by the backend."
                                ),
                            )
                            selection_fingerprint = _mw_stable_digest({
                                "recipe": active_fingerprint,
                                "paths": sorted(selected_paths),
                            })
                            bulk_confirmed = st.checkbox(
                                (
                                    "I explicitly confirm applying this exact recipe to all "
                                    f"{len(selected_paths)} selected concrete paths"
                                ),
                                value=False,
                                key=(
                                    f"mw_replacement_bulk_confirm_{group_key}_"
                                    f"{selection_fingerprint}"
                                ),
                                disabled=not selected_paths,
                            )
                            if st.button(
                                "Apply recipe to selected paths",
                                key=(
                                    f"mw_replacement_bulk_apply_{group_key}_"
                                    f"{selection_fingerprint}"
                                ),
                                disabled=not selected_paths or not bulk_confirmed,
                                width='stretch',
                            ):
                                for destination_path in selected_paths:
                                    destination = _mw_copy.deepcopy(draft)
                                    for cache_key in (
                                        "confirmed_fingerprint",
                                        "validated_fingerprint",
                                        "normalized_spec",
                                    ):
                                        destination.pop(cache_key, None)
                                    drafts[destination_path] = destination
                                    bulk_confirm_paths.add(destination_path)
                                st.success(
                                    "Copied the exact recipe. Each selected path is being "
                                    "inspected and validated independently below."
                                )

                drafts[active_path] = draft
                if editor_error:
                    draft.pop("confirmed_fingerprint", None)

        normalized_specs = {}
        status_rows = []
        issues = []
        for path, row in sorted(rows_by_path.items()):
            draft = drafts.setdefault(path, {
                "target_id": "",
                "constructor_args_text": "[]",
                "constructor_kwargs_text": "{}",
                "state_choices": {},
                "constant_values": {},
            })
            try:
                inspection = inspected_by_path.get(path)
                if inspection is None:
                    inspection, _, _ = _mw_inspect_replacement_draft(
                        reference_model,
                        path,
                        draft,
                    )
                spec, fingerprint = _mw_compile_replacement_spec(path, draft, inspection)
                if path in bulk_confirm_paths:
                    draft["confirmed_fingerprint"] = fingerprint
                    spec["confirmed"] = True
                if not spec.get("confirmed"):
                    raise ValueError(
                        "Explicit confirmation is required for this exact replacement recipe."
                    )
                validation_fingerprint = _mw_stable_digest({
                    "path": path,
                    "spec": spec,
                })
                if (
                    draft.get("validated_fingerprint") == validation_fingerprint
                    and isinstance(draft.get("normalized_spec"), dict)
                ):
                    normalized = _mw_copy.deepcopy(draft["normalized_spec"])
                else:
                    normalized = dict(_mw_as_dict(
                        _mw_validate_replacement_spec(reference_model, path, spec)
                    ))
                    draft["validated_fingerprint"] = validation_fingerprint
                    draft["normalized_spec"] = _mw_copy.deepcopy(normalized)
                normalized_specs[path] = normalized
                for backend_warning in normalized.get("warnings", []) or []:
                    st.warning(f"Replacement `{path}`: {backend_warning}")
                status = "Validated and confirmed"
                detail = "Backend accepted the exact path-specific recipe."
            except Exception as exc:
                draft.pop("validated_fingerprint", None)
                draft.pop("normalized_spec", None)
                status = "Blocked"
                detail = str(exc)
                issues.append(f"{path}: {exc}")
            status_rows.append({
                "Concrete source path": path,
                "Source type": row.get("Type", ""),
                "Target catalog ID": draft.get("target_id", ""),
                "Status": status,
                "Detail": detail,
            })

        st.markdown("**Per-path backend validation**")
        st.dataframe(status_rows, hide_index=True, width='stretch')
        if issues:
            issue_preview = issues[:5]
            if len(issues) > len(issue_preview):
                issue_preview.append(f"… and {len(issues) - len(issue_preview)} more paths")
            st.error(
                "Build and validation are locked until every explicit replacement is valid "
                "and confirmed. " + " | ".join(issue_preview)
            )
        else:
            st.success(
                f"Backend validated {len(normalized_specs)} explicit replacement "
                f"{'path' if len(normalized_specs) == 1 else 'paths'}."
            )

        fingerprint_payload = {
            "drafts": {
                path: {
                    "target_id": draft.get("target_id", ""),
                    "constructor_args_text": draft.get("constructor_args_text", "[]"),
                    "constructor_kwargs_text": draft.get("constructor_kwargs_text", "{}"),
                    "state_choices": dict(draft.get("state_choices", {}) or {}),
                    "constant_values": dict(draft.get("constant_values", {}) or {}),
                    "confirmed_fingerprint": draft.get("confirmed_fingerprint"),
                }
                for path, draft in sorted(drafts.items())
                if path in rows_by_path
            },
            "normalized_specs": normalized_specs,
        }
        ready = not issues and len(normalized_specs) == len(rows_by_path)
        return normalized_specs, status_rows, ready, fingerprint_payload


    @st.fragment
    def _render_model_workbench_tab():
        st.markdown("""
        <div class="dashboard-hero">
            <div class="dashboard-hero__eyebrow">Model Onboarding · Guided Quantization</div>
            <h1>Model Quantization Workbench</h1>
            <p>Inspect the operations a model actually executes, resolve custom layers, build a quantized clone, and validate it on a sample input.</p>
        </div>
        """, unsafe_allow_html=True)

        if _MW_IMPORT_ERROR is not None:
            st.error(f"The model workbench backend could not be imported: {_MW_IMPORT_ERROR}")
            st.exception(_MW_IMPORT_ERROR)
            return
        if _MW_LOADED_ANALYSIS_SCHEMA != _MW_REQUIRED_ANALYSIS_SCHEMA:
            st.error(
                "The dashboard process has an older Model Workbench backend loaded "
                f"(schema {_MW_LOADED_ANALYSIS_SCHEMA}; expected "
                f"{_MW_REQUIRED_ANALYSIS_SCHEMA}). Restart `./dashboard.sh` once to load "
                "the updated ViT/MobileViT capture logic."
            )
            return
        benchmark_backend_available = not (
            _MW_LOADED_BENCHMARK_API != _MW_REQUIRED_BENCHMARK_API
            or _mw_build_classification_validation_loader is None
            or _mw_benchmark_classification_models is None
        )

        st.info(
            "A module name is never accepted as proof of equivalence. Custom subclasses that "
            "override `forward` are expanded into their executed operations and may map to "
            "several QBench operations."
        )

        setup_left, setup_right = st.columns([1.05, 1])
        with setup_left:
            st.markdown("#### 1. Choose a model")
            source_label = st.selectbox(
                "Model source",
                ["torchvision", "timm", "custom factory"],
                key="mw_model_source",
                help="Custom factories use an import path such as `my_package.models:build_model` and execute trusted local Python code.",
            )
            source = "custom" if source_label == "custom factory" else source_label
            custom_factory = None
            if source == "custom":
                custom_factory = st.text_input(
                    "Factory import path",
                    placeholder="my_package.models:build_model",
                    key="mw_custom_factory",
                ).strip()
                model_name = custom_factory.rsplit(":", 1)[-1] if custom_factory else "custom_model"
                st.warning("Only use factories from code you trust. Importing a custom factory executes Python in the dashboard process.")
            else:
                try:
                    model_names = _mw_cached_model_names(source)
                except Exception as exc:
                    st.error(f"Could not enumerate {source} models: {exc}")
                    model_names = []
                if not model_names:
                    st.warning(f"No models are available from `{source}` in this environment.")
                    return
                preferred = "resnet18" if "resnet18" in model_names else model_names[0]
                model_name = st.selectbox(
                    f"{source} model ({len(model_names):,} available)",
                    model_names,
                    index=model_names.index(preferred),
                    key=f"mw_{source}_model_name",
                )
            pretrained = st.checkbox(
                "Load default pretrained weights",
                value=False,
                key="mw_pretrained",
                help="May download weights when they are not already cached.",
            )

        with setup_right:
            st.markdown("#### 2. Describe a sample input")
            use_provider_shape = False
            if source != "custom":
                use_provider_shape = st.checkbox(
                    "Use the model's preferred image size",
                    value=True,
                    key="mw_use_provider_input_shape",
                    help="Uses provider metadata when available (for example, MobileViT-S uses 256×256).",
                )
            input_shape_text = st.text_input(
                "Input tensor shape",
                value="1, 3, 224, 224",
                key="mw_input_shape",
                help="Examples: `1, 3, 224, 224` for an image model or `1, 768` for an MLP.",
                disabled=use_provider_shape,
            )
            if use_provider_shape:
                input_shape = None
                st.code("Resolved from model metadata during analysis", language="text")
            else:
                try:
                    input_shape = _mw_parse_input_shape(input_shape_text)
                except ValueError as exc:
                    input_shape = None
                    st.error(str(exc))
                else:
                    st.code(f"torch.randn{input_shape}", language="python")
            st.caption(
                "The first version accepts one tensor input. Models requiring tokenizers, masks, "
                "or dictionary inputs are reported clearly and can be added through a provider plugin."
            )

        analyze_disabled = (
            (source == "custom" and not custom_factory)
            or (not use_provider_shape and input_shape is None)
        )
        analysis_fingerprint = _mw_json.dumps({
            "analysis_schema": _MW_REQUIRED_ANALYSIS_SCHEMA,
            "source": source,
            "model_name": model_name,
            "custom_factory": custom_factory or "",
            "pretrained": bool(pretrained),
            "input_shape": (
                "provider_default"
                if use_provider_shape
                else list(input_shape) if input_shape is not None else None
            ),
        }, sort_keys=True)
        if st.button(
            "Analyze model operations",
            key="mw_analyze",
            type="primary",
            width='stretch',
            disabled=analyze_disabled,
        ):
            for state_key in (
                "_mw_analysis",
                "_mw_reference_model",
                "_mw_sample_input",
                "_mw_analysis_fingerprint",
                "_mw_replacement_drafts",
            ):
                st.session_state.pop(state_key, None)
            _mw_clear_derived_state()
            _mw_torch.manual_seed(0)
            with st.spinner(f"Loading and tracing `{model_name}`..."):
                try:
                    reference_model = _mw_load_model(
                        source=source,
                        model_name=model_name,
                        pretrained=pretrained,
                        custom_factory=custom_factory,
                    )
                    reference_model.cpu().eval()
                    resolved_input_shape = (
                        tuple(_mw_resolve_model_input_size(reference_model, batch_size=1))
                        if use_provider_shape
                        else tuple(input_shape)
                    )
                    sample_input = _mw_torch.randn(*resolved_input_shape)
                    analysis = _mw_analyze_model(
                        reference_model,
                        model_name=model_name,
                        source=source,
                        sample_input=sample_input,
                    )
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
                    st.exception(exc)
                else:
                    st.session_state["_mw_reference_model"] = reference_model
                    st.session_state["_mw_sample_input"] = sample_input
                    st.session_state["_mw_analysis"] = analysis
                    st.session_state["_mw_analysis_token"] = int(st.session_state.get("_mw_analysis_token", 0)) + 1
                    st.session_state["_mw_model_name"] = model_name
                    st.session_state["_mw_model_source"] = source
                    st.session_state["_mw_resolved_input_shape"] = resolved_input_shape
                    st.session_state["_mw_analysis_fingerprint"] = analysis_fingerprint
                    st.success(
                        "Analysis complete at input shape "
                        f"{tuple(resolved_input_shape)}. Review the graph and conversion decisions below."
                    )

        analysis = st.session_state.get("_mw_analysis")
        reference_model = st.session_state.get("_mw_reference_model")
        sample_input = st.session_state.get("_mw_sample_input")
        if analysis is None or reference_model is None or sample_input is None:
            st.caption("Choose a model and click **Analyze model operations** to begin.")
            return
        analysis_schema = int(
            _mw_field(
                analysis,
                "schema_version",
                _mw_field(analysis, "summary", {}).get("schema_version", 0),
            )
        )
        if analysis_schema != _MW_REQUIRED_ANALYSIS_SCHEMA:
            st.warning(
                "This result was created by an older analysis engine and has been hidden. "
                "Click **Analyze model operations** to regenerate the ViT graph."
            )
            return
        if st.session_state.get("_mw_analysis_fingerprint") != analysis_fingerprint:
            st.info(
                "The model source, weights, or sample input changed. "
                "Click **Analyze model operations** to refresh the graph before building."
            )
            return

        st.markdown("---")
        st.markdown("#### 3. Inspect support and mappings")
        summary, status_counts = _mw_summary_counts(analysis)
        source_graph = _mw_as_dict(_mw_field(analysis, "source_graph", {}))
        target_graph = _mw_as_dict(_mw_field(analysis, "target_graph", {}))
        mappings = list(_mw_field(analysis, "mappings", []) or [])
        source_ops = int(summary.get("source_node_count", summary.get("source_nodes", len(source_graph.get("nodes", [])))))
        target_ops = int(summary.get("target_node_count", summary.get("target_nodes", len(target_graph.get("nodes", [])))))
        supported_count = int(summary.get(
            "supported_modules",
            summary.get("supported", sum(
                count for status, count in status_counts.items()
                if "support" in status or "quantized" in status or "custom_expanded" in status
            )),
        ))
        unresolved_count = int(summary.get(
            "unsupported_modules",
            summary.get("unsupported", sum(
                count for status, count in status_counts.items()
                if "unsupported" in status or "unresolved" in status or "fp32_fallback" in status
            )),
        ))
        capture_kind = str(
            _mw_field(analysis, "capture_kind", summary.get("capture_kind", "module_hierarchy"))
        )
        capture_details = dict(
            _mw_field(analysis, "capture_details", summary.get("capture_details", {})) or {}
        )
        operation_counts = dict(summary.get("operation_status_counts", {}) or {})
        operation_supported = sum(
            int(operation_counts.get(status, 0))
            for status in (
                "exact_native_support",
                "transparent_subclass",
                "custom_expanded",
                "functional_support",
            )
        )
        operation_fp32 = sum(
            int(operation_counts.get(status, 0))
            for status in ("fp32_fallback", "unsupported")
        )
        operation_total = operation_supported + operation_fp32
        module_total = supported_count + unresolved_count
        module_coverage = (
            f"{supported_count}/{module_total}"
            if module_total
            else "n/a"
        )
        operation_coverage = (
            f"{operation_supported}/{operation_total}"
            if operation_total
            else "n/a"
        )

        if capture_kind == "fx":
            st.success("Graph capture: full quantization-aware FX, validated on the selected sample.")
        elif capture_kind == "torch_export":
            shape = capture_details.get("sample_input_shape")
            shape_label = " × ".join(str(dim) for dim in shape) if shape else "the selected shape"
            st.info(
                "Graph capture: input-specialized `torch.export` at "
                f"{shape_label}. Conversion stays eager and disables unsafe whole-model FX rewriting."
            )
        else:
            st.error(
                "Only the module hierarchy could be captured. Operation-level support is unknown; "
                "conversion remains conservative until sample validation passes."
            )

        metric_cols = st.columns(5)
        metric_cols[0].metric("Convertible modules", module_coverage)
        metric_cols[1].metric("Quantizable operations", operation_coverage)
        metric_cols[2].metric("FP32 operations", operation_fp32)
        metric_cols[3].metric("Structural modules", int(summary.get("passthrough_modules", 0)))
        metric_cols[4].metric("Mappings", len(mappings))
        st.caption(
            f"Graph size: {source_ops:,} reference nodes → {target_ops:,} proposed nodes. "
            "Shape/control nodes are excluded from the operation-coverage denominator."
        )

        for warning in list(_mw_field(analysis, "warnings", []) or []):
            st.warning(str(warning))

        module_rows = [dict(_mw_as_dict(row)) for row in list(_mw_field(analysis, "module_rows", []) or [])]
        if module_rows:
            support_rows = []
            for row in module_rows:
                candidates = row.get("candidates", []) or []
                if isinstance(candidates, str):
                    candidates = [candidates]
                support_rows.append({
                    "Path": row.get("path", row.get("module_path", "<root>")),
                    "Type": row.get("type", row.get("module_type", "")),
                    "Status": row.get("status", ""),
                    "Reason": row.get("reason", ""),
                    "Candidates": ", ".join(str(candidate) for candidate in candidates),
                })
            with st.expander("Complete module support report", expanded=False):
                st.dataframe(_mw_pd.DataFrame(support_rows), width='stretch', hide_index=True)

        st.markdown("#### 4. Resolve conversion decisions")
        st.caption(
            "`decompose` keeps the custom forward visible and maps its operations separately. "
            "An explicit native alias replaces the whole custom module and is always validated "
            "against the reference output. `replace` opens a catalog-backed, path-specific "
            "recipe editor; it never assumes equivalence from a class name."
        )

        decision_rows = []
        allowed_by_node = {}
        recommended_decisions = {}
        replacement_eligible_nodes = set()
        for index, row in enumerate(module_rows):
            candidates = row.get("candidates", []) or []
            if isinstance(candidates, str):
                candidates = [candidates]
            candidates = [str(candidate) for candidate in candidates if str(candidate)]
            status = str(row.get("status", "") or "")
            replacement_eligible = bool(row.get("custom_type")) or status in {
                "custom_expanded",
                "fp32_fallback",
                "unsupported",
                "mixed_quantized_fp32",
            }
            if replacement_eligible and "replace" not in candidates:
                candidates.append("replace")
            recommended = str(row.get("recommended") or (candidates[0] if candidates else "keep_fp32"))
            if recommended not in candidates:
                candidates.insert(0, recommended)
            node_id = str(row.get("node_id") or row.get("id") or row.get("path") or f"module-{index}")
            if replacement_eligible:
                replacement_eligible_nodes.add(node_id)
            allowed_by_node[node_id] = candidates or [recommended]
            recommended_decisions[node_id] = recommended
            decision_rows.append({
                "Node ID": node_id,
                "Path": row.get("path", row.get("module_path", "<root>")),
                "Type": row.get("type", row.get("module_type", "")),
                "Status": row.get("status", ""),
                "Allowed choices": " · ".join(candidates),
                "Decision": recommended,
            })

        show_all_decisions = st.checkbox(
            "Show advanced override for every convertible module",
            value=False,
            key="mw_show_all_decisions",
            help="By default the editor shows custom, decomposed, and FP32 rows that need review.",
        )
        attention_statuses = {
            "custom_expanded",
            "fp32_fallback",
            "unsupported",
            "mixed_quantized_fp32",
        }
        visible_decision_rows = (
            decision_rows
            if show_all_decisions
            else [
                row
                for row in decision_rows
                if row["Status"] in attention_statuses
                or row["Node ID"] in replacement_eligible_nodes
            ]
        )

        if visible_decision_rows:
            all_decision_options = sorted({
                option for options in allowed_by_node.values() for option in options
            })
            decisions_df = st.data_editor(
                _mw_pd.DataFrame(visible_decision_rows),
                key=f"mw_decision_editor_{st.session_state.get('_mw_analysis_token', 0)}",
                width='stretch',
                hide_index=True,
                disabled=["Node ID", "Path", "Type", "Status", "Allowed choices"],
                column_config={
                    "Node ID": None,
                    "Path": st.column_config.TextColumn("Module / operation", width="large"),
                    "Type": st.column_config.TextColumn("Type", width="medium"),
                    "Status": st.column_config.TextColumn("Support", width="medium"),
                    "Allowed choices": st.column_config.TextColumn("Allowed choices", width="large"),
                    "Decision": st.column_config.SelectboxColumn(
                        "Decision",
                        options=all_decision_options,
                        required=True,
                        width="medium",
                    ),
                },
            )
            decisions = dict(recommended_decisions)
            decisions.update({
                str(row["Node ID"]): str(row["Decision"])
                for _, row in decisions_df.iterrows()
            })
        else:
            st.info(
                "No custom or FP32 decisions require attention. Enable the advanced override "
                "to keep any supported layer in FP32."
            )
            decisions = recommended_decisions

        invalid_decisions = {
            node_id: decision
            for node_id, decision in decisions.items()
            if decision not in allowed_by_node.get(node_id, [decision])
        }
        if invalid_decisions:
            st.error(
                "One or more decisions are not valid for that row: "
                + ", ".join(f"{node_id}={decision}" for node_id, decision in invalid_decisions.items())
            )

        replacement_rows = [
            row
            for row in decision_rows
            if decisions.get(str(row["Node ID"])) == "replace"
        ]
        (
            replacement_specs,
            _replacement_status_rows,
            replacements_ready,
            replacement_fingerprint_payload,
        ) = _mw_render_replacement_editor(
            reference_model,
            replacement_rows,
            int(st.session_state.get("_mw_analysis_token", 0)),
        )

        q_col1, q_col2, q_col3, q_col4 = st.columns(4)
        quantization_type = q_col1.selectbox(
            "Weight format",
            ["fp8_e4m3", "fp8_e5m2", "fp7_e3m3", "fp6_e3m2", "fp4_e2m1"],
            key="mw_quant_type",
        )
        weight_quantization = q_col2.checkbox(
            "Quantize weights",
            value=bool(_mw_torch.cuda.is_available()),
            key="mw_weight_quant",
            help="Disable for a CPU-safe structural conversion preview.",
        )
        quantize_first_layer = q_col3.checkbox(
            "Quantize first layer",
            value=False,
            key="mw_quant_first",
        )
        fx_rewrite_available = capture_kind == "fx"
        if not fx_rewrite_available:
            st.session_state["mw_enable_fx"] = False
        enable_fx = q_col4.checkbox(
            "Rewrite functional ops",
            value=fx_rewrite_available,
            key="mw_enable_fx",
            disabled=not fx_rewrite_available,
            help=(
                "Available only when the full symbolic graph is valid for the selected input. "
                "Export/module-hierarchy capture uses safe eager module conversion."
            ),
        )
        if weight_quantization and not _mw_torch.cuda.is_available():
            st.warning(
                "QBench's calibrated weight codec may require CUDA in this environment. "
                "Disable **Quantize weights** for a structural CPU preview."
            )
        st.caption(
            "Activation-boundary quantization is intentionally disabled in this preview. "
            "QBench activation inference must execute through the hardware transport runtime, not a raw `model(x)` call."
        )

        quant_options = {
            "quantization_type": quantization_type,
            "weight_quantization": bool(weight_quantization),
            "input_quantization": False,
            "output_quantization": False,
            "quantize_first_layer": bool(quantize_first_layer),
            "enable_fx_quantization": bool(enable_fx),
            "fold_layers": False,
            "fold_input_norm": False,
            "skip_calibration": not bool(weight_quantization),
        }
        build_fingerprint = _mw_json.dumps({
            "analysis": analysis_fingerprint,
            "decisions": decisions,
            "quant_options": quant_options,
            "replacement_specs": replacement_specs,
            "replacement_editor": replacement_fingerprint_payload,
        }, sort_keys=True)

        preview_ready = not invalid_decisions and replacements_ready
        if preview_ready:
            try:
                preview_kwargs = {}
                if replacement_rows:
                    preview_kwargs["replacement_specs"] = replacement_specs
                preview_plan = _mw_build_conversion_plan(
                    analysis,
                    decisions,
                    quant_options,
                    **preview_kwargs,
                )
                target_graph, mappings = _mw_preview_conversion_plan(analysis, preview_plan)
            except Exception as preview_exc:
                preview_ready = False
                st.warning(f"The edited graph preview could not be produced: {preview_exc}")
        else:
            st.warning(
                "The edited graph preview is withheld until every explicit replacement "
                "recipe is backend-valid and confirmed. The graph below remains the "
                "analysis-time proposal."
            )
        st.markdown("##### Current conversion-plan graph")
        st.caption(
            "The overview is a strict layer-type conversion map. Detailed views retain "
            "individual operations for inspecting decompositions and data flow."
        )
        graph_detail = st.selectbox(
            "Graph detail",
            [
                "Layer-type overview",
                "Quantization-relevant",
                "Modules only",
                "Full graph",
            ],
            key="mw_graph_detail",
            help=(
                "The overview groups hundreds of repeated layers by type. Detailed views retain "
                "individual nodes and hide or show shape/control bookkeeping as requested."
            ),
        )
        if graph_detail == "Layer-type overview":
            rendered_source_graph, rendered_target_graph, rendered_mappings = _mw_type_overview(
                source_graph,
                target_graph,
                mappings,
            )
            source_type_count = len(rendered_source_graph.get("nodes", []))
            target_type_count = len(rendered_target_graph.get("nodes", []))
            arrow_count = len(rendered_mappings)

            def _type_map_noun(count, singular, plural):
                return singular if count == 1 else plural

            st.caption(
                f"Layer map: {source_type_count:,} original "
                f"{_type_map_noun(source_type_count, 'layer type', 'layer types')} → "
                f"{target_type_count:,} converted "
                f"{_type_map_noun(target_type_count, 'layer type', 'layer types')}, with "
                f"{arrow_count:,} conversion "
                f"{_type_map_noun(arrow_count, 'arrow', 'arrows')}. "
                "Supported child types keep their intrinsic Quant mapping; any inherited "
                "FP32 work is attributed to the enclosing composite. These arrows describe "
                "conversion, not forward data flow."
            )
            _mw_render_type_mapping_overview(
                rendered_source_graph,
                rendered_target_graph,
                rendered_mappings,
                key=(
                    f"mw_type_mapping_{st.session_state.get('_mw_analysis_token', 0)}_"
                    f"{abs(hash(build_fingerprint))}"
                ),
            )
            with st.expander("Exact layer-type mappings", expanded=False):
                st.dataframe(
                    [
                        {
                            "Original layer type": mapping.get("source_type", ""),
                            "Converted layer type": mapping.get("target_type", ""),
                            "Mapped instances": int(mapping.get("count", 0) or 0),
                            "Conversion": str(mapping.get("kind", "mapped")).replace("_", " "),
                            "Why": str(mapping.get("reason", "") or ""),
                        }
                        for mapping in rendered_mappings
                    ],
                    hide_index=True,
                    width='stretch',
                )
                if replacement_specs:
                    st.markdown("**Explicit path-specific replacements in this strict map**")
                    st.dataframe(
                        [
                            {
                                "Concrete source path": path,
                                "Target catalog ID": spec.get("target_id", ""),
                                "Confirmed": bool(spec.get("confirmed")),
                                "Backend validation": "valid",
                            }
                            for path, spec in sorted(replacement_specs.items())
                        ],
                        hide_index=True,
                        width='stretch',
                    )
        else:
            rendered_source_graph = _mw_filter_graph(source_graph, graph_detail)
            rendered_target_graph = _mw_filter_graph(target_graph, graph_detail)
            rendered_mappings = _mw_filter_mappings(
                mappings,
                rendered_source_graph,
                rendered_target_graph,
            )
            st.caption(
                f"Rendered view: {len(rendered_source_graph.get('nodes', [])):,} source nodes and "
                f"{len(rendered_target_graph.get('nodes', [])):,} target nodes."
            )
            _mw_render_mapping_graph(
                rendered_source_graph,
                rendered_target_graph,
                rendered_mappings,
                key=(
                    f"mw_mapping_graph_{st.session_state.get('_mw_analysis_token', 0)}_"
                    f"{abs(hash(build_fingerprint))}_{graph_detail}"
                ),
            )

        if st.button(
            "Build quantized model",
            key="mw_build_quantized",
            type="primary",
            width='stretch',
            disabled=bool(invalid_decisions) or not replacements_ready or not preview_ready,
            help=(
                "Resolve, explicitly confirm, and pass backend validation for every custom "
                "replacement before building."
                if replacement_rows and (not replacements_ready or not preview_ready)
                else None
            ),
        ):
            _mw_clear_derived_state()
            with st.spinner("Applying the conversion plan to a cloned model..."):
                try:
                    build_kwargs = {}
                    if replacement_rows:
                        build_kwargs["replacement_specs"] = replacement_specs
                    plan = _mw_build_conversion_plan(
                        analysis,
                        decisions,
                        quant_options,
                        **build_kwargs,
                    )
                    conversion_result = _mw_convert_model(reference_model, plan)
                    quantized_model = _mw_field(conversion_result, "model")
                    if quantized_model is None:
                        raise RuntimeError("The converter did not return a model.")
                except Exception as exc:
                    st.error(f"Conversion failed: {exc}")
                    st.exception(exc)
                else:
                    st.session_state["_mw_conversion_plan"] = plan
                    st.session_state["_mw_conversion_result"] = conversion_result
                    st.session_state["_mw_quantized_model"] = quantized_model
                    st.session_state["_mw_build_fingerprint"] = build_fingerprint
                    st.session_state.pop("_mw_export_bundle", None)
                    st.session_state["_mw_validation_state"] = {
                        "status": "pending",
                        "reason": "The converted model has not completed sample validation.",
                    }
                    st.success("Quantized model built from a clone. The reference model was not modified.")
                    try:
                        initial_validation = _mw_run_sample_inference(
                            reference_model,
                            quantized_model,
                            sample_input,
                        )
                    except Exception as validation_exc:
                        st.session_state.pop("_mw_inference_result", None)
                        st.session_state["_mw_validation_state"] = {
                            "status": "failed",
                            "reason": (
                                "Automatic sample inference raised "
                                f"{type(validation_exc).__name__}: {validation_exc}"
                            ),
                        }
                    else:
                        inference_view = _mw_inference_view(initial_validation)
                        st.session_state["_mw_inference_result"] = inference_view
                        st.session_state["_mw_validation_state"] = _mw_validation_state(
                            inference_view,
                            require_allclose=not bool(weight_quantization),
                        )

        conversion_result = st.session_state.get("_mw_conversion_result")
        quantized_model = st.session_state.get("_mw_quantized_model")
        plan = st.session_state.get("_mw_conversion_plan")
        if conversion_result is None or quantized_model is None or plan is None:
            return
        if st.session_state.get("_mw_build_fingerprint") != build_fingerprint:
            st.info(
                "The conversion decisions or quantization settings changed. "
                "Click **Build quantized model** again before validating or exporting."
            )
            return

        for warning in list(_mw_field(conversion_result, "warnings", []) or []):
            st.warning(str(warning))

        realization = dict(_mw_field(conversion_result, "realization", {}) or {})
        realized_by_type = dict(realization.get("by_type", {}) or {})
        realized_total = int(realization.get("total", sum(realized_by_type.values())) or 0)
        separate_model = quantized_model is not reference_model
        if separate_model:
            st.success(
                "Real conversion: this is a separate cloned model with "
                f"{realized_total:,} materialized QBench "
                f"{'module' if realized_total == 1 else 'modules'}; the reference object "
                "remains unchanged."
            )
        else:
            st.error(
                "The converter returned the reference object itself. Validation and export "
                "are disabled because an independent converted model was not materialized."
            )
            return
        st.caption(
            "Validation uses temporary execution-audit hooks only to count which QBench "
            "modules ran. The hooks are removed immediately after inference; they do not "
            "implement quantization or alter either model."
        )
        with st.expander("What was actually rebuilt?", expanded=False):
            st.write(
                "Supported layers are concrete `Quant*` or `Decomposed*` modules in the "
                "converted clone. Native containers, control/shape operations, and any "
                "explicit FP32 fallback islands remain where they are required."
            )
            if realized_by_type:
                st.dataframe(
                    [
                        {"QBench module type": module_type, "Instances": int(count)}
                        for module_type, count in sorted(realized_by_type.items())
                    ],
                    hide_index=True,
                    width='stretch',
                )
            else:
                st.warning(
                    "No QBench modules were found in this converted model. Review the "
                    "conversion decisions before treating it as quantized."
                )

        validation_state = dict(st.session_state.get("_mw_validation_state", {}) or {})
        if validation_state.get("status") == "passed":
            st.success(validation_state.get("reason", "Sample validation passed."))
        elif validation_state.get("status") == "failed":
            st.error(
                "Converted-model validation failed. State export is disabled. "
                + str(validation_state.get("reason", ""))
            )
        else:
            st.warning(
                validation_state.get(
                    "reason",
                    "Run sample inference before exporting converted state.",
                )
            )

        st.markdown("---")
        st.markdown("#### 5. Validate and export")
        action_col1, action_col2 = st.columns(2)
        if action_col1.button("Run sample inference", key="mw_run_inference", width='stretch'):
            with st.spinner("Running reference and quantized inference..."):
                try:
                    inference_result = _mw_run_sample_inference(
                        reference_model,
                        quantized_model,
                        sample_input,
                    )
                except Exception as exc:
                    st.session_state.pop("_mw_export_bundle", None)
                    st.session_state["_mw_validation_state"] = {
                        "status": "failed",
                        "reason": f"Sample inference raised {type(exc).__name__}: {exc}",
                    }
                    st.error(f"Sample inference failed: {exc}")
                    st.exception(exc)
                else:
                    inference_view = _mw_inference_view(inference_result)
                    st.session_state["_mw_inference_result"] = inference_view
                    st.session_state["_mw_validation_state"] = _mw_validation_state(
                        inference_view,
                        require_allclose=not bool(weight_quantization),
                    )
                    if st.session_state["_mw_validation_state"]["status"] != "passed":
                        st.session_state.pop("_mw_export_bundle", None)

        if action_col2.button(
            "Prepare state bundle",
            key="mw_prepare_export",
            width='stretch',
            disabled=validation_state.get("status") != "passed",
            help="A converted state bundle is available only after successful sample validation.",
        ):
            with st.spinner("Serializing converted weights..."):
                try:
                    cpu_state = {
                        name: (
                            value.detach().cpu()
                            if _mw_torch.is_tensor(value)
                            else _mw_copy.deepcopy(value)
                        )
                        for name, value in quantized_model.state_dict().items()
                    }
                    buffer = _mw_io.BytesIO()
                    export_benchmark = st.session_state.get(
                        "_mw_dataset_benchmark"
                    )
                    if (
                        export_benchmark
                        and _mw_field(export_benchmark, "fingerprint")
                        != st.session_state.get(
                            "_mw_dataset_benchmark_fingerprint"
                        )
                    ):
                        export_benchmark = None
                    _mw_torch.save({
                        "state_dict": cpu_state,
                        "conversion_recipe": _mw_recipe_with_replacements(
                            conversion_result,
                            plan,
                            replacement_specs,
                        ),
                        "validation": {
                            "schema_version": 1,
                            "sample_status": _mw_copy.deepcopy(
                                st.session_state.get("_mw_validation_state")
                            ),
                            "sample": st.session_state.get("_mw_inference_result"),
                            "dataset_benchmark": export_benchmark,
                        },
                    }, buffer)
                    st.session_state["_mw_export_bundle"] = buffer.getvalue()
                except Exception as exc:
                    st.error(f"Could not serialize the converted state: {exc}")

        st.markdown("##### Dataset subset accuracy")
        st.caption(
            "Run both models on the exact same labeled samples. Image folders use the "
            "selected torchvision/timm model's evaluation preprocessing; a custom factory "
            "can provide any other classification Dataset or DataLoader."
        )
        if not benchmark_backend_available:
            st.error(
                "The running dashboard has an older dataset-validation backend loaded "
                f"(API {_MW_LOADED_BENCHMARK_API}; expected "
                f"{_MW_REQUIRED_BENCHMARK_API}). Restart `./dashboard.sh` once to enable "
                "dataset accuracy comparison; graph analysis and sample export remain available."
            )
        dataset_kind_label = st.selectbox(
            "Validation dataset",
            ["ImageNet / ImageFolder", "Custom dataset factory"],
            key="mw_dataset_kind",
        )
        dataset_kind = (
            "image_folder"
            if dataset_kind_label == "ImageNet / ImageFolder"
            else "custom_factory"
        )
        dataset_path = None
        dataset_factory = None
        dataset_factory_kwargs = None
        dataset_factory_kwargs_error = None
        dataset_split = "val"
        if dataset_kind == "image_folder":
            dataset_path_col, dataset_split_col = st.columns([3, 1])
            dataset_path = dataset_path_col.text_input(
                "Dataset root or split directory",
                value="/data/imagenet",
                key="mw_dataset_path",
                help=(
                    "Accepts either a dataset root containing the selected split or an "
                    "ImageFolder directory containing one subdirectory per class."
                ),
            ).strip()
            dataset_split = dataset_split_col.text_input(
                "Split",
                value="val",
                key="mw_dataset_split",
            ).strip() or "val"
            st.caption(
                "ImageNet WordNet directory names are remapped to canonical class indices. "
                "Other ImageFolder datasets retain their deterministic folder ordering."
            )
        else:
            dataset_factory = st.text_input(
                "Dataset factory import path",
                placeholder="my_package.datasets:build_validation_dataset",
                key="mw_dataset_factory",
                help=(
                    "The trusted callable may return a labeled torch Dataset or DataLoader. "
                    "It owns preprocessing for non-ImageNet/custom-model datasets."
                ),
            ).strip()
            st.warning(
                "Only use dataset factories from code you trust. Importing the factory "
                "executes local Python code in the dashboard process."
            )
            dataset_factory_kwargs_text = st.text_area(
                "Factory keyword arguments (JSON object)",
                value="{}",
                key="mw_dataset_factory_kwargs",
                help="Passed directly to the trusted dataset factory.",
            ).strip()
            try:
                dataset_factory_kwargs = _mw_json.loads(
                    dataset_factory_kwargs_text or "{}"
                )
                if not isinstance(dataset_factory_kwargs, dict):
                    raise ValueError("Factory arguments must be a JSON object.")
            except (ValueError, TypeError, _mw_json.JSONDecodeError) as exc:
                dataset_factory_kwargs = None
                dataset_factory_kwargs_error = str(exc)
                st.error(f"Invalid factory arguments: {exc}")

        dataset_control_cols = st.columns(5)
        dataset_samples = int(dataset_control_cols[0].number_input(
            "Subset samples",
            min_value=1,
            max_value=50_000,
            value=128,
            step=1,
            key="mw_dataset_samples",
        ))
        dataset_batch_size = int(dataset_control_cols[1].number_input(
            "Batch size",
            min_value=1,
            max_value=512,
            value=8,
            step=1,
            key="mw_dataset_batch_size",
        ))
        dataset_workers = int(dataset_control_cols[2].number_input(
            "Loader workers",
            min_value=0,
            max_value=32,
            value=0,
            step=1,
            key="mw_dataset_workers",
        ))
        dataset_seed = int(dataset_control_cols[3].number_input(
            "Subset seed",
            min_value=0,
            max_value=2_147_483_647,
            value=42,
            step=1,
            key="mw_dataset_seed",
        ))
        device_options = ["auto", "cpu"]
        if _mw_torch.cuda.is_available():
            device_options.append("cuda")
        benchmark_device = dataset_control_cols[4].selectbox(
            "Device",
            device_options,
            key="mw_dataset_device",
            help="Auto uses CUDA when available and otherwise CPU.",
        )

        if source != "custom" and not pretrained:
            st.warning(
                "This model was loaded without pretrained weights. Accuracy deltas are still "
                "computed, but absolute ImageNet accuracy is not meaningful until trained "
                "weights are loaded."
            )

        benchmark_config = {
            "dataset_kind": dataset_kind,
            "path": dataset_path,
            "custom_factory": dataset_factory,
            "factory_kwargs": dataset_factory_kwargs,
            "split": dataset_split,
            "max_samples": dataset_samples,
            "batch_size": dataset_batch_size,
            "num_workers": dataset_workers,
            "seed": dataset_seed,
            "device": benchmark_device,
            "model_source": source,
            "model_name": model_name,
            "pretrained": bool(pretrained),
        }
        benchmark_fingerprint = _mw_json.dumps(
            {"build": build_fingerprint, "dataset": benchmark_config},
            sort_keys=True,
        )
        stored_benchmark_fingerprint = st.session_state.get(
            "_mw_dataset_benchmark_fingerprint"
        )
        if (
            stored_benchmark_fingerprint
            and stored_benchmark_fingerprint != benchmark_fingerprint
        ):
            for state_key in (
                "_mw_dataset_benchmark",
                "_mw_dataset_benchmark_config",
                "_mw_dataset_benchmark_error",
                "_mw_dataset_benchmark_fingerprint",
                "_mw_export_bundle",
            ):
                st.session_state.pop(state_key, None)
            st.info(
                "The dataset, subset, or execution settings changed. Run the accuracy "
                "comparison again before including it in an export."
            )

        benchmark_missing_input = (
            dataset_kind == "image_folder" and not dataset_path
        ) or (
            dataset_kind == "custom_factory" and not dataset_factory
        ) or dataset_factory_kwargs_error is not None
        benchmark_disabled = (
            not benchmark_backend_available
            or
            validation_state.get("status") != "passed"
            or benchmark_missing_input
        )
        if st.button(
            "Compare reference and converted accuracy",
            key="mw_run_dataset_benchmark",
            type="primary",
            width='stretch',
            disabled=benchmark_disabled,
            help=(
                "The converted model must first pass sample structure validation."
                if validation_state.get("status") != "passed"
                else None
            ),
        ):
            for state_key in (
                "_mw_dataset_benchmark",
                "_mw_dataset_benchmark_error",
                "_mw_export_bundle",
            ):
                st.session_state.pop(state_key, None)
            progress_bar = st.progress(0.0)
            progress_text = st.empty()

            def update_benchmark_progress(processed_samples, total_samples):
                processed = max(int(processed_samples), 0)
                total = max(int(total_samples), 1)
                progress_bar.progress(min(processed / total, 1.0))
                progress_text.caption(
                    f"Evaluated {min(processed, total):,} / {total:,} samples"
                )

            try:
                with st.spinner("Loading the labeled subset and evaluating both models..."):
                    loader_result = _mw_build_classification_validation_loader(
                        dataset_kind=dataset_kind,
                        model=reference_model,
                        source=source,
                        model_name=model_name,
                        path=dataset_path,
                        custom_factory=dataset_factory,
                        factory_kwargs=dataset_factory_kwargs,
                        batch_size=dataset_batch_size,
                        max_samples=dataset_samples,
                        seed=dataset_seed,
                        num_workers=dataset_workers,
                        split=dataset_split,
                    )
                    if isinstance(loader_result, tuple) and len(loader_result) == 2:
                        data_loader, dataset_metadata = loader_result
                    else:
                        data_loader = loader_result
                        dataset_metadata = getattr(
                            data_loader,
                            "qbench_metadata",
                            {},
                        )
                    benchmark_result = _mw_benchmark_classification_models(
                        reference_model,
                        conversion_result,
                        data_loader,
                        max_samples=dataset_samples,
                        device=benchmark_device,
                        progress_callback=update_benchmark_progress,
                    )
            except Exception as exc:
                st.session_state["_mw_dataset_benchmark_error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                st.session_state["_mw_dataset_benchmark_fingerprint"] = (
                    benchmark_fingerprint
                )
            else:
                st.session_state["_mw_dataset_benchmark"] = {
                    "schema_version": 1,
                    "fingerprint": benchmark_fingerprint,
                    "config": _mw_copy.deepcopy(benchmark_config),
                    "loader_metadata": _mw_as_dict(dataset_metadata),
                    "result": _mw_as_dict(benchmark_result),
                }
                st.session_state["_mw_dataset_benchmark_config"] = (
                    _mw_copy.deepcopy(benchmark_config)
                )
                st.session_state["_mw_dataset_benchmark_fingerprint"] = (
                    benchmark_fingerprint
                )
            finally:
                progress_bar.empty()
                progress_text.empty()

        benchmark_error = st.session_state.get("_mw_dataset_benchmark_error")
        if (
            benchmark_error
            and st.session_state.get("_mw_dataset_benchmark_fingerprint")
            == benchmark_fingerprint
        ):
            benchmark_error_data = _mw_as_dict(benchmark_error)
            st.error(
                "Dataset accuracy comparison failed: "
                f"{benchmark_error_data.get('type', 'Error')}: "
                f"{benchmark_error_data.get('message', benchmark_error)}"
            )

        benchmark_record = st.session_state.get("_mw_dataset_benchmark")
        if (
            benchmark_record
            and st.session_state.get("_mw_dataset_benchmark_fingerprint")
            == benchmark_fingerprint
        ):
            benchmark_data = _mw_as_dict(benchmark_record).get("result", {})
            reference_accuracy = dict(benchmark_data.get("reference", {}) or {})
            quantized_accuracy = dict(benchmark_data.get("quantized", {}) or {})
            accuracy_delta = dict(benchmark_data.get("delta", {}) or {})
            top1_delta = float(
                accuracy_delta.get("top1_accuracy_percentage_points", 0.0)
            )
            top5_delta = float(
                accuracy_delta.get("top5_accuracy_percentage_points", 0.0)
            )
            effective_top5 = int(benchmark_data.get("effective_top5_k", 5))
            accuracy_cols = st.columns(5)
            accuracy_cols[0].metric(
                "Reference top-1",
                f"{float(reference_accuracy.get('top1_accuracy_percent', 0.0)):.2f}%",
            )
            accuracy_cols[1].metric(
                "Converted top-1",
                f"{float(quantized_accuracy.get('top1_accuracy_percent', 0.0)):.2f}%",
                delta=f"{top1_delta:+.2f} pp",
            )
            accuracy_cols[2].metric(
                f"Reference top-{effective_top5}",
                f"{float(reference_accuracy.get('top5_accuracy_percent', 0.0)):.2f}%",
            )
            accuracy_cols[3].metric(
                f"Converted top-{effective_top5}",
                f"{float(quantized_accuracy.get('top5_accuracy_percent', 0.0)):.2f}%",
                delta=f"{top5_delta:+.2f} pp",
            )
            accuracy_cols[4].metric(
                "Prediction agreement",
                f"{float(benchmark_data.get('prediction_agreement_percent', 0.0)):.2f}%",
            )
            benchmark_detail_cols = st.columns(4)
            benchmark_detail_cols[0].metric(
                "Evaluated samples",
                f"{int(benchmark_data.get('samples', 0)):,}",
            )
            benchmark_detail_cols[1].metric(
                "Reference throughput",
                f"{float(reference_accuracy.get('throughput_samples_per_second', 0.0)):.1f} img/s",
            )
            benchmark_detail_cols[2].metric(
                "Converted throughput",
                f"{float(quantized_accuracy.get('throughput_samples_per_second', 0.0)):.1f} img/s",
            )
            benchmark_detail_cols[3].metric(
                "Output classes",
                f"{int(benchmark_data.get('class_dimension', 0)):,}",
            )
            loader_metadata = dict(
                _mw_as_dict(benchmark_record).get("loader_metadata", {}) or {}
            )
            dataset_class_count = loader_metadata.get("class_count")
            output_class_count = int(benchmark_data.get("class_dimension", 0))
            canonical_mapped = int(
                loader_metadata.get("canonical_imagenet_labels_mapped", 0) or 0
            )
            if (
                dataset_class_count is not None
                and int(dataset_class_count) != output_class_count
                and canonical_mapped == 0
            ):
                st.warning(
                    f"The dataset reports {int(dataset_class_count):,} classes, but the "
                    f"model returns {output_class_count:,} logits. Confirm that labels and "
                    "the loaded model weights use the same class index mapping."
                )
            for dataset_warning in loader_metadata.get("warnings", []) or []:
                st.warning(str(dataset_warning))
            if effective_top5 < 5:
                st.caption(
                    f"The model exposes only {effective_top5} classes, so the top-5 card "
                    f"uses top-{effective_top5}."
                )
            with st.expander("Complete dataset benchmark report", expanded=False):
                st.json(_mw_as_dict(benchmark_record))

        inference_result = st.session_state.get("_mw_inference_result")
        if inference_result is not None:
            inference_data = _mw_as_dict(inference_result)
            comparison_data = dict(inference_data.get("comparison", {}) or {})
            runtime_audit = dict(inference_data.get("runtime_audit", {}) or {})
            if runtime_audit:
                audit_cols = st.columns(3)
                audit_cols[0].metric(
                    "Realized QBench modules",
                    int(runtime_audit.get("quantized_modules_total", 0)),
                )
                audit_cols[1].metric(
                    "Executed QBench modules",
                    int(runtime_audit.get("executed_quantized_modules", 0)),
                )
                audit_cols[2].metric(
                    "QBench runtime calls",
                    sum(int(value) for value in runtime_audit.get("runtime_calls_by_type", {}).values()),
                )
            metric_candidates = [
                ("Mean abs. error", "mean_abs_error", ".4e"),
                ("Max abs. error", "max_abs_error", ".4e"),
                ("Cosine similarity", "cosine_similarity", ".6f"),
                ("Compared tensors", "compared_tensors", "d"),
            ]
            available = [item for item in metric_candidates if item[1] in comparison_data]
            if available:
                cols = st.columns(len(available))
                for col, (label, field, fmt) in zip(cols, available):
                    value = comparison_data[field]
                    try:
                        rendered = format(value, fmt)
                    except (TypeError, ValueError):
                        rendered = str(value)
                    col.metric(label, rendered)
            if comparison_data.get("structure_match") is False:
                st.error("Reference and converted outputs do not have the same structure or tensor shapes.")
            elif "allclose" in comparison_data:
                if comparison_data["allclose"]:
                    st.success("Reference and converted outputs are within the configured tolerance.")
                else:
                    st.warning("The outputs have matching structure but are outside the configured numerical tolerance.")
            if not weight_quantization:
                st.caption(
                    "Validation mode: FP32 structural equivalence. QBench wrappers execute, "
                    "but weight and activation quantization are disabled."
                )
            with st.expander("Complete inference result", expanded=not bool(available)):
                # Raw output tensors remain available in session state but are
                # intentionally omitted here; Streamlit's JSON renderer cannot
                # serialize arbitrary tensors and large logits would swamp the UI.
                st.json({
                    "comparison": comparison_data,
                    "runtime_audit": runtime_audit,
                    "reference_summary": inference_data.get("reference_summary"),
                    "quantized_summary": inference_data.get("quantized_summary"),
                })

        plan_data = _mw_recipe_with_replacements(
            conversion_result,
            plan,
            replacement_specs,
        )
        export_name = _mw_safe_filename(st.session_state.get("_mw_model_name", "model"))
        dl_col1, report_col, state_col = st.columns(3)
        dl_col1.download_button(
            "Download conversion recipe",
            data=_mw_json.dumps(plan_data, indent=2, sort_keys=True, default=str),
            file_name=f"{export_name}.qbench-recipe.json",
            mime="application/json",
            key="mw_download_recipe",
            width='stretch',
        )
        current_benchmark_report = st.session_state.get("_mw_dataset_benchmark")
        if (
            current_benchmark_report
            and _mw_field(current_benchmark_report, "fingerprint")
            == st.session_state.get("_mw_dataset_benchmark_fingerprint")
        ):
            report_col.download_button(
                "Download validation report",
                data=_mw_json.dumps(
                    current_benchmark_report,
                    indent=2,
                    sort_keys=True,
                    default=str,
                ),
                file_name=f"{export_name}.qbench-validation.json",
                mime="application/json",
                key="mw_download_validation_report",
                width='stretch',
            )
        else:
            report_col.caption("Run a dataset comparison to download an accuracy report.")
        export_bundle = st.session_state.get("_mw_export_bundle")
        latest_validation = dict(st.session_state.get("_mw_validation_state", {}) or {})
        if export_bundle and latest_validation.get("status") == "passed":
            state_col.download_button(
                "Download converted state bundle",
                data=export_bundle,
                file_name=f"{export_name}.qbench-state.pt",
                mime="application/octet-stream",
                key="mw_download_bundle",
                width='stretch',
            )
        else:
            state_col.caption("Click **Prepare state bundle** before downloading converted weights.")


    _render_model_workbench_tab()
