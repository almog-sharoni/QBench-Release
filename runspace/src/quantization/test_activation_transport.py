from __future__ import annotations

import pytest
import torch

from runspace.src.quantization.activation_transport import (
    DEFAULT_ACTIVATION_TRANSPORT,
    ActivationPacket,
    ActivationTransport,
    decode_activation_packet,
    encode_dynamic_packet,
    encode_uniform_packet,
    normalize_activation_transport,
)
from runspace.src.quantization.chunking import (
    chunk_tensor_by_context,
    unchunk_tensor_by_context,
)


def test_transport_mode_normalization_and_default():
    assert DEFAULT_ACTIVATION_TRANSPORT == "encoded"
    assert normalize_activation_transport() == "encoded"
    assert normalize_activation_transport(" REFERENCE ") == "reference"
    with pytest.raises(ValueError, match="Unsupported activation transport"):
        normalize_activation_transport("legacy")


def test_cpu_reference_uniform_and_dynamic_use_context_chunks():
    torch.manual_seed(7)
    tensor = torch.randn(2, 3, 17, dtype=torch.float32)
    transport = ActivationTransport("reference")

    uniform = transport.transmit_uniform(tensor, "fp8_e4m3", producer_id="relu")
    assert isinstance(uniform, torch.Tensor)
    assert uniform.shape == tensor.shape
    assert uniform.dtype == torch.float32

    # Shape [2, 3, 17] merges three contexts per batch into two 128-wide chunks.
    format_ids = torch.tensor([0, 1], dtype=torch.int64)
    dynamic = transport.transmit_dynamic(
        tensor,
        format_ids,
        ("fp4_e2m1", "fp8_e4m3"),
        producer_id="softmax",
    )
    assert isinstance(dynamic, torch.Tensor)
    assert dynamic.shape == tensor.shape
    assert dynamic.dtype == torch.float32
    assert torch.isfinite(dynamic).all()
    assert not torch.equal(dynamic, tensor)


def test_encoded_transport_rejects_cpu_and_bad_format_ids():
    tensor = torch.randn(2, 128, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="requires a CUDA tensor"):
        encode_uniform_packet(tensor, "fp8_e4m3")

    reference = ActivationTransport("reference")
    with pytest.raises(ValueError, match="must be in"):
        reference.transmit_dynamic(
            tensor,
            [0, 2],
            ("fp4_e2m1", "fp8_e4m3"),
        )
    with pytest.raises(ValueError, match="must be in"):
        reference.transmit_dynamic(tensor[:1], [2**32], ("fp8_e4m3",))
    with pytest.raises(ValueError, match="must be unique"):
        reference.transmit_dynamic(
            tensor,
            [0, 0],
            ("fp8_e4m3", "fp8_e4m3"),
        )


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for encoded activation transport")


def test_uniform_packet_metadata_and_reference_parity_cuda():
    _require_cuda()
    torch.manual_seed(11)
    tensor = torch.randn(1, 3, 7, 7, device="cuda", dtype=torch.float32)

    packet = encode_uniform_packet(tensor, "fp8_e4m3", producer_id="features.relu")
    reference = ActivationTransport("reference").transmit_uniform(
        tensor,
        "fp8_e4m3",
        producer_id="features.relu",
    )
    decoded = decode_activation_packet(packet)

    assert isinstance(packet, ActivationPacket)
    assert packet.producer_id == "features.relu"
    assert packet.version == 2
    assert packet.layout.original_shape == tuple(tensor.shape)
    assert packet.layout.kind == "packed_spatial_contexts"
    assert packet.layout.num_chunks == packet.scales.numel()
    assert packet.original_shape == tuple(tensor.shape)
    assert packet.chunk_size == 128
    assert packet.num_chunks == packet.layout.num_chunks
    assert packet.candidate_formats == ("fp8_e4m3",)
    assert packet.payload.dtype == torch.int32
    assert packet.scales.dtype == torch.float32
    assert packet.format_ids.dtype == torch.int32
    assert packet.word_offsets.dtype == torch.int64
    assert packet.signedness == "signed"
    assert packet.encoded_nbytes > packet.payload.numel() * packet.payload.element_size()
    assert torch.equal(decoded, reference)
    assert torch.equal(packet.decode(), reference)


def test_mixed_width_packet_offsets_and_reference_parity_cuda():
    _require_cuda()
    torch.manual_seed(13)
    tensor = torch.randn(4, 128, device="cuda", dtype=torch.float32)
    candidates = ("fp4_e2m1", "fp8_e4m3")
    format_ids = torch.tensor([0, 1, 0, 1], device="cuda", dtype=torch.int64)

    packet = encode_dynamic_packet(
        tensor,
        format_ids,
        candidates,
        producer_id="stage.output",
    )
    decoded = packet.decode()
    reference = ActivationTransport("reference").transmit_dynamic(
        tensor,
        format_ids,
        candidates,
        producer_id="stage.output",
    )

    assert tuple(packet.word_offsets.cpu().tolist()) == (0, 16, 48, 64, 96)
    assert tuple(fmt.words_per_chunk for fmt in packet.formats) == (16, 32)
    assert packet.payload.numel() == 96
    assert packet.layout.num_chunks == 4
    assert packet.signedness == "signed"
    assert torch.equal(decoded, reference)


def test_packet_matches_existing_dynamic_selector_output_cuda():
    _require_cuda()
    from runspace.src.quantization.cuda import search_best_chunk_format

    torch.manual_seed(17)
    tensor = torch.randn(8, 128, device="cuda", dtype=torch.float32).contiguous()
    candidates = ("fp4_e2m1", "fp6_e3m2", "fp8_e4m3")
    candidate_e = torch.tensor([2, 3, 4], device="cuda", dtype=torch.int32)
    candidate_m = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int32)
    candidate_signed = torch.ones(3, device="cuda", dtype=torch.int32)
    best_ids, _scales, best_values, _capture = search_best_chunk_format(
        tensor.reshape(-1),
        candidate_e,
        candidate_m,
        candidate_signed,
        False,
        0,
        0.0625,
        0,
        0,
        0,
    )

    packet = encode_dynamic_packet(tensor, best_ids, candidates)
    decoded = packet.decode()
    assert torch.equal(decoded, best_values.reshape_as(tensor))


@pytest.mark.parametrize("shape", [(1, 2, 14, 14), (1, 4, 7, 7)])
def test_dynamic_selector_packet_parity_for_spatial_layouts_cuda(shape):
    _require_cuda()
    from runspace.src.quantization.cuda import search_best_chunk_format

    torch.manual_seed(19)
    tensor = torch.randn(*shape, device="cuda", dtype=torch.float32).contiguous()
    chunks, original_shape, padding = chunk_tensor_by_context(tensor, 128)
    candidates = ("fp4_e2m1", "fp8_e4m3")
    best_ids, _scales, best_values, _capture = search_best_chunk_format(
        chunks.reshape(-1).contiguous(),
        torch.tensor([2, 4], device="cuda", dtype=torch.int32),
        torch.tensor([1, 3], device="cuda", dtype=torch.int32),
        torch.ones(2, device="cuda", dtype=torch.int32),
        False,
        0,
        0.0625,
        0,
        0,
        0,
    )
    expected = unchunk_tensor_by_context(
        best_values.reshape(chunks.shape),
        original_shape,
        padding,
    )

    packet = encode_dynamic_packet(tensor, best_ids, candidates)
    assert torch.equal(packet.decode(), expected)


def test_dynamic_selector_packet_parity_for_merged_transformer_layout_cuda():
    _require_cuda()
    from runspace.src.quantization.cuda import search_best_chunk_format

    torch.manual_seed(23)
    tensor = torch.randn(1, 256, 192, device="cuda", dtype=torch.float32)
    chunks, original_shape, padding = chunk_tensor_by_context(tensor, 128)
    candidates = ("fp8_e1m6", "fp8_e2m5")
    best_ids, _scales, best_values, _capture = search_best_chunk_format(
        chunks.reshape(-1).contiguous(),
        torch.tensor([1, 2], device="cuda", dtype=torch.int32),
        torch.tensor([6, 5], device="cuda", dtype=torch.int32),
        torch.ones(2, device="cuda", dtype=torch.int32),
        False,
        0,
        0.0625,
        0,
        0,
        0,
    )
    expected = unchunk_tensor_by_context(
        best_values.reshape(chunks.shape),
        original_shape,
        padding,
    )

    packet = encode_dynamic_packet(tensor, best_ids, candidates)

    assert packet.layout.algorithm == "qbench_context_v2"
    assert packet.layout.chunked_shape == tuple(chunks.shape)
    assert packet.num_chunks == best_ids.numel()
    assert torch.equal(packet.decode(), expected)


def test_unsigned_dynamic_packet_preserves_softmax_zeros_cuda():
    _require_cuda()
    logits = torch.tensor(
        [[0.0, -float("inf"), 1.0, -float("inf")]],
        device="cuda",
        dtype=torch.float32,
    )
    probabilities = torch.softmax(logits, dim=-1).repeat(1, 32).contiguous()
    packet = encode_dynamic_packet(
        probabilities,
        [0],
        ("ufp4_e2m2", "ufp8_e4m4"),
        producer_id="attention.softmax",
    )
    decoded = packet.decode()

    assert packet.signedness == "unsigned"
    assert torch.equal(decoded == 0, probabilities == 0)
    assert torch.all(decoded >= 0)


def _legacy_selected_packet_parts(tensor, format_ids, candidates):
    from runspace.src.quantization.cuda import decode_chunk, encode_chunk, resolve_format

    chunks, original_shape, padding = chunk_tensor_by_context(tensor, 128)
    flat_chunks = chunks.reshape(-1, 128).contiguous()
    num_chunks = flat_chunks.shape[0]
    ids_cpu = [int(value) for value in format_ids.cpu().tolist()]
    encoded_by_format = []
    decoded_by_format = []
    scales_by_format = []
    words_per_chunk = []

    for candidate in candidates:
        exponent_bits, mantissa_bits, is_signed = resolve_format(candidate)
        payload, scales = encode_chunk(
            flat_chunks,
            exponent_bits,
            mantissa_bits,
            is_signed,
        )
        words = payload.numel() // max(num_chunks, 1)
        encoded_by_format.append(payload.reshape(num_chunks, words))
        decoded_by_format.append(
            decode_chunk(
                payload,
                scales,
                [num_chunks, 128],
                exponent_bits,
                mantissa_bits,
                is_signed,
            ).reshape(num_chunks, 128)
        )
        scales_by_format.append(scales)
        words_per_chunk.append(words)

    offsets = [0]
    payload_rows = []
    decoded = torch.empty_like(flat_chunks)
    for chunk_index, format_id in enumerate(ids_cpu):
        payload_rows.append(encoded_by_format[format_id][chunk_index])
        decoded[chunk_index] = decoded_by_format[format_id][chunk_index]
        offsets.append(offsets[-1] + words_per_chunk[format_id])

    payload = torch.cat(payload_rows) if payload_rows else torch.empty(
        0,
        dtype=torch.int32,
        device=tensor.device,
    )
    decoded = unchunk_tensor_by_context(
        decoded.reshape(chunks.shape),
        original_shape,
        padding,
    )
    return payload, scales_by_format, offsets, decoded


def test_fused_uniform_packet_is_bitwise_identical_to_legacy_codec_cuda():
    _require_cuda()
    from runspace.src.quantization.cuda import decode_chunk, encode_chunk

    torch.manual_seed(101)
    tensor = torch.randn(2, 3, 14, 14, device="cuda", dtype=torch.float32)
    chunks, original_shape, padding = chunk_tensor_by_context(tensor, 128)
    flat_chunks = chunks.reshape(-1, 128).contiguous()
    legacy_payload, legacy_scales = encode_chunk(flat_chunks, 4, 3, True)
    legacy_decoded = decode_chunk(
        legacy_payload,
        legacy_scales,
        [flat_chunks.shape[0], 128],
        4,
        3,
        True,
    )
    legacy_decoded = unchunk_tensor_by_context(
        legacy_decoded.reshape(chunks.shape),
        original_shape,
        padding,
    )

    packet = encode_uniform_packet(tensor, "fp8_e4m3")

    assert torch.equal(packet.payload, legacy_payload)
    assert torch.equal(packet.scales, legacy_scales)
    assert torch.equal(packet.decode(), legacy_decoded)
    packet.validate()


@pytest.mark.parametrize(
    "candidates",
    [
        ("fp8_e1m6", "fp8_e2m5"),
        ("ufp8_e1m7", "ufp8_e2m6"),
        ("fp8_e4m3", "ufp8_e4m4"),
        ("fp4_e2m1", "fp8_e4m3"),
    ],
)
def test_selected_format_packer_is_byte_exact_to_legacy_cuda(candidates):
    _require_cuda()
    torch.manual_seed(103)
    tensor = torch.randn(7, 128, device="cuda", dtype=torch.float32)
    tensor[0, :8] = torch.tensor(
        [0.0, -0.0, 2.0**-126, -(2.0**-126), 0.5, -0.5, 1.0, -1.0],
        device="cuda",
    )
    format_ids = torch.tensor([0, 1, 1, 0, 1, 0, 1], device="cuda", dtype=torch.int64)
    expected_payload, expected_scales, expected_offsets, expected_decoded = (
        _legacy_selected_packet_parts(tensor, format_ids, candidates)
    )

    packet = encode_dynamic_packet(tensor, format_ids, candidates)

    assert torch.equal(packet.payload, expected_payload)
    assert all(torch.equal(packet.scales, scales) for scales in expected_scales)
    assert packet.word_offsets.cpu().tolist() == expected_offsets
    assert torch.equal(packet.decode(), expected_decoded)
    packet.validate()


def test_selected_format_packer_uses_current_cuda_stream_cuda():
    _require_cuda()
    tensor = torch.randn(9, 128, device="cuda", dtype=torch.float32)
    format_ids = torch.arange(9, device="cuda", dtype=torch.int64) % 2
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        packet = encode_dynamic_packet(
            tensor,
            format_ids,
            ("fp8_e1m6", "fp8_e2m5"),
        )
        decoded = packet.decode()
    stream.synchronize()

    expected_payload, expected_scales, expected_offsets, expected_decoded = (
        _legacy_selected_packet_parts(
            tensor,
            format_ids,
            ("fp8_e1m6", "fp8_e2m5"),
        )
    )
    assert torch.equal(packet.payload, expected_payload)
    assert all(torch.equal(packet.scales, scales) for scales in expected_scales)
    assert packet.word_offsets.cpu().tolist() == expected_offsets
    assert torch.equal(decoded, expected_decoded)


def test_selected_format_packer_is_byte_exact_for_every_supported_width_cuda():
    _require_cuda()
    torch.manual_seed(107)
    tensor = torch.randn(5, 128, device="cuda", dtype=torch.float32)
    format_ids = torch.tensor([0, 1, 0, 1, 1], device="cuda", dtype=torch.int64)

    for bit_width in range(2, 17):
        if bit_width == 2:
            candidates = ("fp2_e1m0", "ufp2_e1m1")
        else:
            candidates = (
                f"fp{bit_width}_e1m{bit_width - 2}",
                f"fp{bit_width}_e2m{bit_width - 3}",
            )
        expected_payload, expected_scales, expected_offsets, expected_decoded = (
            _legacy_selected_packet_parts(tensor, format_ids, candidates)
        )
        packet = encode_dynamic_packet(tensor, format_ids, candidates)

        assert torch.equal(packet.payload, expected_payload), candidates
        assert all(
            torch.equal(packet.scales, scales) for scales in expected_scales
        ), candidates
        assert packet.word_offsets.cpu().tolist() == expected_offsets
        assert torch.equal(packet.decode(), expected_decoded), candidates
