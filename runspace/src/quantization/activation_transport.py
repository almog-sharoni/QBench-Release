"""Hardware-oriented activation transport primitives.

Encoded transport stores each activation chunk in the format selected for that
chunk.  The payload is densely packed: mixed-width chunks are located through
``word_offsets`` rather than padded to the widest candidate.  Reference
transport returns the same decoded FP32 values without exposing a packet.

The current CUDA codec fixes activation chunks at 128 elements.  This module
keeps that constraint explicit so a caller cannot silently simulate a layout
that hardware and the codec do not share.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import re
from typing import Sequence

import torch

from .chunking import chunk_tensor_by_context, unchunk_tensor_by_context
from .quantizer import quantize


ENCODED_TRANSPORT = "encoded"
REFERENCE_TRANSPORT = "reference"
DEFAULT_ACTIVATION_TRANSPORT = ENCODED_TRANSPORT
ACTIVATION_PACKET_VERSION = 2
HARDWARE_CHUNK_SIZE = 128

_FORMAT_RE = re.compile(
    r"^(?P<prefix>u?fp)(?P<bits>\d+)_e(?P<exp>\d+)m(?P<mant>\d+)$"
)


def normalize_activation_transport(mode: str | None = None) -> str:
    """Return the canonical activation transport mode."""
    normalized = str(mode or DEFAULT_ACTIVATION_TRANSPORT).strip().lower()
    if normalized not in (ENCODED_TRANSPORT, REFERENCE_TRANSPORT):
        raise ValueError(
            f"Unsupported activation transport {mode!r}; expected "
            f"{ENCODED_TRANSPORT!r} or {REFERENCE_TRANSPORT!r}."
        )
    return normalized


@dataclass(frozen=True)
class ActivationFormat:
    """One entry in an activation packet's candidate format table."""

    format_id: int
    name: str
    exponent_bits: int
    mantissa_bits: int
    is_signed: bool
    bit_width: int
    words_per_chunk: int


@dataclass(frozen=True)
class ActivationLayout:
    """Deterministic layout information needed to reconstruct an activation."""

    original_shape: tuple[int, ...]
    chunked_shape: tuple[int, ...]
    padding: int | dict[str, int | str]
    chunk_size: int
    num_chunks: int
    kind: str
    algorithm: str = "qbench_context_v2"


@dataclass(frozen=True)
class ActivationPacket:
    """Packed activation payload plus all metadata needed for decoding."""

    payload: torch.Tensor = field(repr=False)
    scales: torch.Tensor = field(repr=False)
    format_ids: torch.Tensor = field(repr=False)
    word_offsets: torch.Tensor = field(repr=False)
    formats: tuple[ActivationFormat, ...]
    layout: ActivationLayout
    producer_id: str = ""
    version: int = ACTIVATION_PACKET_VERSION
    _decoded_cache: torch.Tensor | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _trusted_contents: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_packet(self, check_contents=not self._trusted_contents)

    @property
    def device(self) -> torch.device:
        return self.payload.device

    @property
    def candidate_formats(self) -> tuple[str, ...]:
        return tuple(fmt.name for fmt in self.formats)

    @property
    def original_shape(self) -> tuple[int, ...]:
        return self.layout.original_shape

    @property
    def chunk_size(self) -> int:
        return self.layout.chunk_size

    @property
    def num_chunks(self) -> int:
        return self.layout.num_chunks

    @property
    def encoded_nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.payload,
                self.scales,
                self.format_ids,
                self.word_offsets,
            )
        )

    @property
    def signedness(self) -> str:
        signed_values = {fmt.is_signed for fmt in self.formats}
        if signed_values == {True}:
            return "signed"
        if signed_values == {False}:
            return "unsigned"
        return "mixed"

    def decode(self) -> torch.Tensor:
        return decode_activation_packet(self)

    def validate(self) -> None:
        """Synchronously validate packet IDs, offsets, and payload bounds."""
        _validate_packet(self, check_contents=True)


def _parse_activation_format(name: str, format_id: int) -> ActivationFormat:
    match = _FORMAT_RE.fullmatch(str(name))
    if match is None:
        raise ValueError(
            f"Unsupported activation format {name!r}; encoded transport supports "
            "fp/ufp formats named like 'fp8_e4m3' or 'ufp8_e4m4'."
        )

    is_signed = match.group("prefix") == "fp"
    bit_width = int(match.group("bits"))
    exponent_bits = int(match.group("exp"))
    mantissa_bits = int(match.group("mant"))
    expected_width = int(is_signed) + exponent_bits + mantissa_bits
    if bit_width != expected_width:
        raise ValueError(
            f"Invalid activation format {name!r}: its name declares {bit_width} "
            f"bits but sign+exponent+mantissa requires {expected_width}."
        )
    if not 2 <= bit_width <= 16:
        raise ValueError(
            f"Unsupported activation width in {name!r}: the CUDA codec supports "
            "element widths from 2 through 16 bits."
        )
    if exponent_bits < 1:
        raise ValueError(f"Activation format {name!r} must have at least one exponent bit.")

    values_per_word = 32 // bit_width
    words_per_chunk = (HARDWARE_CHUNK_SIZE + values_per_word - 1) // values_per_word
    return ActivationFormat(
        format_id=int(format_id),
        name=str(name),
        exponent_bits=exponent_bits,
        mantissa_bits=mantissa_bits,
        is_signed=is_signed,
        bit_width=bit_width,
        words_per_chunk=words_per_chunk,
    )


@lru_cache(maxsize=128)
def _cached_format_table(names: tuple[str, ...]) -> tuple[ActivationFormat, ...]:
    if not names:
        raise ValueError("At least one activation candidate format is required.")
    if len(set(names)) != len(names):
        raise ValueError(f"Activation candidate formats must be unique; got {names!r}.")
    return tuple(_parse_activation_format(name, idx) for idx, name in enumerate(names))


def _format_table(candidate_formats: Sequence[str]) -> tuple[ActivationFormat, ...]:
    return _cached_format_table(tuple(str(fmt) for fmt in candidate_formats))


def _layout_kind(shape: tuple[int, ...], chunk_size: int) -> str:
    if len(shape) < 4:
        return "greedy_context"
    context_width = 1
    for dim in shape[2:]:
        context_width *= dim
    if context_width <= chunk_size:
        return "packed_spatial_contexts"
    return "spatial_rows"


def _make_layout(
    original_shape: Sequence[int],
    chunked_shape: Sequence[int],
    padding: int | dict[str, int | str],
    num_chunks: int,
    chunk_size: int,
) -> ActivationLayout:
    shape = tuple(int(dim) for dim in original_shape)
    if not shape:
        raise ValueError("Activation transport does not support scalar tensors.")
    if any(dim < 0 for dim in shape):
        raise ValueError(f"Activation shape must be non-negative; got {shape!r}.")
    return ActivationLayout(
        original_shape=shape,
        chunked_shape=tuple(int(dim) for dim in chunked_shape),
        padding=padding,
        chunk_size=int(chunk_size),
        num_chunks=int(num_chunks),
        kind=_layout_kind(shape, chunk_size),
    )


def _activation_chunks(tensor: torch.Tensor, chunk_size: int):
    chunked, original_shape, padding = chunk_tensor_by_context(tensor, chunk_size)
    flat_chunks = chunked.reshape(-1, chunk_size).contiguous()
    return flat_chunks, original_shape, chunked.shape, padding


def _require_hardware_tensor(tensor: torch.Tensor, chunk_size: int) -> torch.Tensor:
    if chunk_size != HARDWARE_CHUNK_SIZE:
        raise ValueError(
            f"Encoded activation transport requires chunk_size={HARDWARE_CHUNK_SIZE}; "
            f"got {chunk_size}."
        )
    if not tensor.is_cuda:
        raise RuntimeError(
            "Encoded activation transport requires a CUDA tensor. Use "
            "transport='reference' for CPU execution."
        )
    if tensor.dtype != torch.float32:
        raise TypeError(
            f"Activation transport requires float32 input; got {tensor.dtype}."
        )
    if tensor.ndim == 0:
        raise ValueError("Activation transport does not support scalar tensors.")
    return tensor.contiguous()


def _cuda_codec():
    # Import lazily so CPU reference transport does not compile/load the CUDA extension.
    from .cuda import (
        decode_chunk,
        encode_chunk,
        encode_decode_chunk_rows,
        encode_selected_chunk_rows,
    )

    return (
        encode_chunk,
        decode_chunk,
        encode_decode_chunk_rows,
        encode_selected_chunk_rows,
    )


def _normalize_format_ids(
    format_ids: torch.Tensor | Sequence[int],
    *,
    device: torch.device,
    num_chunks: int,
    num_formats: int,
    check_bounds: bool = True,
) -> torch.Tensor:
    ids = torch.as_tensor(format_ids, device=device)
    if ids.dtype == torch.bool or ids.dtype.is_floating_point or ids.dtype.is_complex:
        raise TypeError("best_format_ids must contain integer candidate indices.")
    ids = ids.reshape(-1)
    if ids.numel() != num_chunks:
        raise ValueError(
            f"best_format_ids has {ids.numel()} entries, but the activation layout "
            f"contains {num_chunks} chunks."
        )
    if check_bounds and ids.numel() > 0:
        minimum = int(ids.min().item())
        maximum = int(ids.max().item())
        if minimum < 0 or maximum >= num_formats:
            raise ValueError(
                f"best_format_ids must be in [0, {num_formats - 1}]; "
                f"observed [{minimum}, {maximum}]."
            )
    return ids.to(dtype=torch.int32).contiguous()


def _word_offsets(
    format_ids: torch.Tensor,
    formats: tuple[ActivationFormat, ...],
) -> torch.Tensor:
    words_by_format = torch.tensor(
        [fmt.words_per_chunk for fmt in formats],
        dtype=torch.int64,
        device=format_ids.device,
    )
    counts = words_by_format[format_ids.to(torch.long)]
    return torch.cat(
        (
            torch.zeros(1, dtype=torch.int64, device=format_ids.device),
            counts.cumsum(dim=0),
        )
    ).contiguous()


def encode_uniform_packet(
    tensor: torch.Tensor,
    q_type: str,
    *,
    producer_id: str = "",
    chunk_size: int = HARDWARE_CHUNK_SIZE,
) -> ActivationPacket:
    """Encode an activation using one format for all context chunks."""
    tensor = _require_hardware_tensor(tensor, chunk_size)
    formats = _format_table((q_type,))
    fmt = formats[0]
    _, _, encode_decode_chunk_rows, _ = _cuda_codec()
    chunks, original_shape, chunked_shape, padding = _activation_chunks(
        tensor,
        chunk_size,
    )
    payload, scales, decoded_chunks = encode_decode_chunk_rows(
        chunks,
        fmt.exponent_bits,
        fmt.mantissa_bits,
        fmt.is_signed,
    )
    num_chunks = int(scales.numel())
    format_ids = torch.zeros(num_chunks, dtype=torch.int32, device=tensor.device)
    offsets = torch.arange(
        num_chunks + 1,
        dtype=torch.int64,
        device=tensor.device,
    ) * fmt.words_per_chunk
    return ActivationPacket(
        payload=payload.contiguous(),
        scales=scales.contiguous(),
        format_ids=format_ids,
        word_offsets=offsets.contiguous(),
        formats=formats,
        layout=_make_layout(
            original_shape,
            chunked_shape,
            padding,
            num_chunks,
            chunk_size,
        ),
        producer_id=str(producer_id),
        _decoded_cache=unchunk_tensor_by_context(
            decoded_chunks.reshape(chunked_shape),
            original_shape,
            padding,
        ),
        _trusted_contents=True,
    )


def encode_dynamic_packet(
    tensor: torch.Tensor,
    best_format_ids: torch.Tensor | Sequence[int],
    candidate_formats: Sequence[str],
    *,
    producer_id: str = "",
    chunk_size: int = HARDWARE_CHUNK_SIZE,
    _candidate_params: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    _trusted_format_ids: bool = False,
) -> ActivationPacket:
    """Encode chunks using format IDs produced by the dynamic selector.

    Candidate formats may have different element widths.  Each selected chunk
    occupies exactly its format's packed word count; ``word_offsets`` locates
    those variable-size records in the flat payload.
    """
    tensor = _require_hardware_tensor(tensor, chunk_size)
    formats = _format_table(candidate_formats)
    _, _, _, encode_selected_chunk_rows = _cuda_codec()

    chunks, original_shape, chunked_shape, padding = _activation_chunks(
        tensor,
        chunk_size,
    )
    num_chunks = int(chunks.shape[0])
    format_ids = _normalize_format_ids(
        best_format_ids,
        device=tensor.device,
        num_chunks=num_chunks,
        num_formats=len(formats),
        check_bounds=not _trusted_format_ids,
    )
    common_width = formats[0].bit_width
    same_width = all(fmt.bit_width == common_width for fmt in formats)
    if _candidate_params is None:
        cands_e = torch.tensor(
            [fmt.exponent_bits for fmt in formats],
            dtype=torch.int32,
            device=tensor.device,
        )
        cands_m = torch.tensor(
            [fmt.mantissa_bits for fmt in formats],
            dtype=torch.int32,
            device=tensor.device,
        )
        cands_sgn = torch.tensor(
            [int(fmt.is_signed) for fmt in formats],
            dtype=torch.int32,
            device=tensor.device,
        )
    else:
        cands_e, cands_m, cands_sgn = _candidate_params

    if same_width:
        offsets = torch.arange(
            num_chunks + 1,
            dtype=torch.int64,
            device=tensor.device,
        ) * formats[0].words_per_chunk
        payload, scales, decoded_chunks = encode_selected_chunk_rows(
            chunks,
            format_ids,
            cands_e,
            cands_m,
            cands_sgn,
            offsets,
            num_chunks * formats[0].words_per_chunk,
        )
    else:
        # A dense mixed-width packet needs its terminal offset on the host to
        # allocate the exact payload size. Packing itself remains one CUDA pass.
        offsets = _word_offsets(format_ids, formats)
        payload_words = int(offsets[-1].item())
        payload, scales, decoded_chunks = encode_selected_chunk_rows(
            chunks,
            format_ids,
            cands_e,
            cands_m,
            cands_sgn,
            offsets,
            payload_words,
        )

    decoded_cache = unchunk_tensor_by_context(
        decoded_chunks.reshape(chunked_shape),
        original_shape,
        padding,
    )

    return ActivationPacket(
        payload=payload.contiguous(),
        scales=scales.contiguous(),
        format_ids=format_ids,
        word_offsets=offsets,
        formats=formats,
        layout=_make_layout(
            original_shape,
            chunked_shape,
            padding,
            num_chunks,
            chunk_size,
        ),
        producer_id=str(producer_id),
        _decoded_cache=decoded_cache,
        _trusted_contents=True,
    )


def _validate_packet(
    packet: ActivationPacket,
    *,
    check_contents: bool,
) -> None:
    if packet.version != ACTIVATION_PACKET_VERSION:
        raise ValueError(
            f"Unsupported activation packet version {packet.version}; "
            f"expected {ACTIVATION_PACKET_VERSION}."
        )
    if packet.layout.chunk_size != HARDWARE_CHUNK_SIZE:
        raise ValueError(
            f"Activation packet chunk_size must be {HARDWARE_CHUNK_SIZE}; "
            f"got {packet.layout.chunk_size}."
        )
    if not packet.formats:
        raise ValueError("Activation packet must contain at least one format entry.")
    if tuple(fmt.format_id for fmt in packet.formats) != tuple(range(len(packet.formats))):
        raise ValueError("Activation packet format IDs must be dense and ordered from zero.")

    tensors = (packet.payload, packet.scales, packet.format_ids, packet.word_offsets)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("Encoded activation packet tensors must all reside on CUDA.")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("Encoded activation packet tensors must share one CUDA device.")
    if packet.payload.dtype != torch.int32:
        raise TypeError(f"Activation payload must be int32; got {packet.payload.dtype}.")
    if packet.scales.dtype != torch.float32:
        raise TypeError(f"Activation scales must be float32; got {packet.scales.dtype}.")
    if packet.format_ids.dtype != torch.int32:
        raise TypeError(f"Activation format_ids must be int32; got {packet.format_ids.dtype}.")
    if packet.word_offsets.dtype != torch.int64:
        raise TypeError(f"Activation word_offsets must be int64; got {packet.word_offsets.dtype}.")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("Encoded activation packet tensors must be contiguous.")
    if packet._decoded_cache is not None:
        decoded = packet._decoded_cache
        if not decoded.is_cuda or decoded.device != packet.device:
            raise ValueError("Cached decoded activation must share the packet CUDA device.")
        if decoded.dtype != torch.float32:
            raise TypeError(
                f"Cached decoded activation must be float32; got {decoded.dtype}."
            )
        if tuple(decoded.shape) != packet.layout.original_shape:
            raise ValueError(
                "Cached decoded activation shape must match layout.original_shape."
            )

    num_chunks = packet.layout.num_chunks
    chunked_shape = packet.layout.chunked_shape
    if not chunked_shape or chunked_shape[-1] != packet.layout.chunk_size:
        raise ValueError(
            "Activation packet chunked_shape must end in layout.chunk_size."
        )
    chunked_elements = 1
    for dim in chunked_shape:
        chunked_elements *= dim
    if chunked_elements != num_chunks * packet.layout.chunk_size:
        raise ValueError(
            "Activation packet chunked_shape does not match layout.num_chunks."
        )
    if packet.scales.numel() != num_chunks or packet.format_ids.numel() != num_chunks:
        raise ValueError(
            "Activation packet scale and format-ID counts must equal "
            f"layout.num_chunks={num_chunks}."
        )
    if packet.word_offsets.numel() != num_chunks + 1:
        raise ValueError(
            "Activation packet word_offsets must contain one entry per chunk plus "
            "the terminal payload offset."
        )

    if check_contents and num_chunks:
        minimum = int(packet.format_ids.min().item())
        maximum = int(packet.format_ids.max().item())
        if minimum < 0 or maximum >= len(packet.formats):
            raise ValueError(
                f"Activation packet format IDs are outside [0, {len(packet.formats) - 1}]."
            )
    if check_contents:
        if int(packet.word_offsets[0].item()) != 0:
            raise ValueError("Activation packet word_offsets must start at zero.")
        if int(packet.word_offsets[-1].item()) != packet.payload.numel():
            raise ValueError("Activation packet terminal word offset must equal payload length.")

        expected_offsets = _word_offsets(packet.format_ids, packet.formats)
        if not torch.equal(packet.word_offsets, expected_offsets):
            raise ValueError(
                "Activation packet word offsets do not match the selected format widths."
            )


def decode_activation_packet(packet: ActivationPacket) -> torch.Tensor:
    """Decode a uniform or mixed-format activation packet to FP32."""
    _validate_packet(packet, check_contents=not packet._trusted_contents)
    if packet._decoded_cache is not None:
        return packet._decoded_cache
    _, decode_chunk, _, _ = _cuda_codec()
    shape = packet.layout.original_shape
    num_chunks = packet.layout.num_chunks
    if num_chunks == 0:
        return torch.empty(shape, dtype=torch.float32, device=packet.device)

    selected_ids = [int(value) for value in torch.unique(packet.format_ids, sorted=True).tolist()]
    if len(selected_ids) == 1:
        fmt = packet.formats[selected_ids[0]]
        decoded_chunks = decode_chunk(
            packet.payload,
            packet.scales,
            [num_chunks, packet.layout.chunk_size],
            fmt.exponent_bits,
            fmt.mantissa_bits,
            fmt.is_signed,
        )
        return unchunk_tensor_by_context(
            decoded_chunks.reshape(packet.layout.chunked_shape),
            shape,
            packet.layout.padding,
        )

    decoded_chunks = None
    for format_id in selected_ids:
        fmt = packet.formats[format_id]
        chunk_indices = torch.nonzero(
            packet.format_ids == format_id,
            as_tuple=False,
        ).flatten()
        source_indices = packet.word_offsets[chunk_indices, None] + torch.arange(
            fmt.words_per_chunk,
            dtype=torch.int64,
            device=packet.device,
        )
        format_payload = torch.zeros(
            (num_chunks, fmt.words_per_chunk),
            dtype=torch.int32,
            device=packet.device,
        )
        format_payload[chunk_indices] = packet.payload[source_indices]
        decoded = decode_chunk(
            format_payload.reshape(-1).contiguous(),
            packet.scales,
            [num_chunks, packet.layout.chunk_size],
            fmt.exponent_bits,
            fmt.mantissa_bits,
            fmt.is_signed,
        )
        flat_chunks = decoded.reshape(num_chunks, packet.layout.chunk_size)
        if decoded_chunks is None:
            decoded_chunks = torch.empty_like(flat_chunks)
        decoded_chunks[chunk_indices] = flat_chunks[chunk_indices]

    assert decoded_chunks is not None
    return unchunk_tensor_by_context(
        decoded_chunks.reshape(packet.layout.chunked_shape),
        shape,
        packet.layout.padding,
    )


def _chunk_scales(chunks: torch.Tensor) -> torch.Tensor:
    amax = chunks.abs().amax(dim=1, keepdim=True)
    scales = torch.ones_like(amax)
    nonzero = amax != 0
    if nonzero.any():
        values = amax[nonzero].contiguous()
        bits = values.view(torch.int32)
        exponent_mask = torch.tensor(-8388608, dtype=torch.int32, device=chunks.device)
        scales[nonzero] = torch.bitwise_and(bits, exponent_mask).view(torch.float32)
    return scales


def _reference_dynamic(
    tensor: torch.Tensor,
    best_format_ids: torch.Tensor | Sequence[int],
    candidate_formats: Sequence[str],
    chunk_size: int,
) -> torch.Tensor:
    if tensor.dtype != torch.float32:
        raise TypeError(
            f"Activation transport requires float32 input; got {tensor.dtype}."
        )
    if tensor.ndim == 0:
        raise ValueError("Activation transport does not support scalar tensors.")
    formats = _format_table(candidate_formats)
    chunked, original_shape, padding = chunk_tensor_by_context(tensor, chunk_size)
    chunked_shape = chunked.shape
    chunks = chunked.reshape(-1, chunk_size)
    format_ids = _normalize_format_ids(
        best_format_ids,
        device=tensor.device,
        num_chunks=chunks.shape[0],
        num_formats=len(formats),
    )
    if chunks.shape[0] == 0:
        return tensor.clone()

    scales = _chunk_scales(chunks)
    scaled = chunks / scales
    result = torch.empty_like(chunks)
    for format_id in [int(value) for value in torch.unique(format_ids, sorted=True).tolist()]:
        mask = format_ids == format_id
        result[mask] = quantize(scaled[mask], q_type=formats[format_id].name) * scales[mask]
    return unchunk_tensor_by_context(
        result.reshape(chunked_shape),
        original_shape,
        padding,
    )


class ActivationTransport:
    """Transmit quantized activations as hardware packets or FP32 references."""

    def __init__(
        self,
        mode: str = DEFAULT_ACTIVATION_TRANSPORT,
        chunk_size: int = HARDWARE_CHUNK_SIZE,
    ) -> None:
        self.mode = normalize_activation_transport(mode)
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError(f"Activation transport chunk_size must be positive; got {self.chunk_size}.")
        if self.mode == ENCODED_TRANSPORT and self.chunk_size != HARDWARE_CHUNK_SIZE:
            raise ValueError(
                f"Encoded activation transport requires chunk_size={HARDWARE_CHUNK_SIZE}; "
                f"got {self.chunk_size}."
            )

    def transmit_uniform(
        self,
        tensor: torch.Tensor,
        q_type: str,
        *,
        producer_id: str = "",
    ) -> ActivationPacket | torch.Tensor:
        if self.mode == REFERENCE_TRANSPORT and (
            not tensor.is_cuda or self.chunk_size != HARDWARE_CHUNK_SIZE
        ):
            num_chunks = chunk_tensor_by_context(tensor, self.chunk_size)[0].numel()
            num_chunks //= self.chunk_size
            ids = torch.zeros(num_chunks, dtype=torch.int32, device=tensor.device)
            return _reference_dynamic(tensor, ids, (q_type,), self.chunk_size)

        packet = encode_uniform_packet(
            tensor,
            q_type,
            producer_id=producer_id,
            chunk_size=self.chunk_size,
        )
        if self.mode == REFERENCE_TRANSPORT:
            return decode_activation_packet(packet)
        return packet

    def transmit_dynamic(
        self,
        tensor: torch.Tensor,
        best_format_ids: torch.Tensor | Sequence[int],
        candidate_formats: Sequence[str],
        *,
        producer_id: str = "",
        candidate_params: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        trusted_format_ids: bool = False,
    ) -> ActivationPacket | torch.Tensor:
        if self.mode == REFERENCE_TRANSPORT:
            return _reference_dynamic(
                tensor,
                best_format_ids,
                candidate_formats,
                self.chunk_size,
            )

        packet = encode_dynamic_packet(
            tensor,
            best_format_ids,
            candidate_formats,
            producer_id=producer_id,
            chunk_size=self.chunk_size,
            _candidate_params=candidate_params,
            _trusted_format_ids=trusted_format_ids,
        )
        return packet

    @staticmethod
    def decode(value: ActivationPacket | torch.Tensor) -> torch.Tensor:
        if isinstance(value, ActivationPacket):
            return decode_activation_packet(value)
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                "ActivationTransport.decode expects an ActivationPacket or torch.Tensor; "
                f"got {type(value).__name__}."
            )
        return value


__all__ = [
    "ACTIVATION_PACKET_VERSION",
    "DEFAULT_ACTIVATION_TRANSPORT",
    "ENCODED_TRANSPORT",
    "HARDWARE_CHUNK_SIZE",
    "REFERENCE_TRANSPORT",
    "ActivationFormat",
    "ActivationLayout",
    "ActivationPacket",
    "ActivationTransport",
    "decode_activation_packet",
    "encode_dynamic_packet",
    "encode_uniform_packet",
    "normalize_activation_transport",
]
