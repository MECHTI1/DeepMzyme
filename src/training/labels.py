from __future__ import annotations

from pathlib import Path
import re
from typing import Optional, Tuple

from data_structures import PocketRecord
from label_schemes import map_site_metal_symbols


EC_TOP_LEVEL_RE = re.compile(r"__EC_(\d+)")
STRUCTURE_ID_RE = re.compile(r"^(?P<pdbid>[^_]+)__chain_(?P<chain>[^_]+)__EC_(?P<ec>.+)$")


def normalize_ec_number_list(value: str) -> str:
    values = []
    seen = set()
    for ec in re.split(r"[;,]", value):
        ec = ec.strip()
        if not ec or ec in seen:
            continue
        seen.add(ec)
        values.append(ec)
    return ";".join(values)


def parse_structure_identity(structure_id: str) -> Tuple[str, str, str]:
    match = STRUCTURE_ID_RE.match(structure_id.strip())
    if match is None:
        raise ValueError(f"Could not parse structure identity from {structure_id!r}")
    return (
        match.group("pdbid").strip().lower(),
        match.group("chain").strip(),
        normalize_ec_number_list(match.group("ec")),
    )


def parse_structure_ec_numbers(structure_id: str) -> tuple[str, ...]:
    _pdbid, _chain, ec_values = parse_structure_identity(structure_id)
    return tuple(ec for ec in normalize_ec_number_list(ec_values).split(";") if ec)


def ec_label_token_from_numbers(
    ec_numbers: tuple[str, ...],
    *,
    depth: int,
) -> Optional[str]:
    if depth < 1:
        raise ValueError(f"EC label depth must be at least 1, got {depth}.")
    prefixes = {
        ".".join(ec_number.split(".")[:depth])
        for ec_number in ec_numbers
        if len(ec_number.split(".")) >= depth
    }
    if not prefixes:
        return None
    if len(prefixes) > 1:
        return None
    return next(iter(prefixes))


def ec_label_token_from_structure_id(
    structure_id: str,
    *,
    depth: int,
) -> Optional[str]:
    return ec_label_token_from_numbers(parse_structure_ec_numbers(structure_id), depth=depth)


def parse_ec_label_token_from_structure_path(
    path: Path,
    *,
    depth: int,
) -> Optional[str]:
    match = EC_TOP_LEVEL_RE.search(path.stem)
    if match is None:
        return None
    return ec_label_token_from_structure_id(path.stem, depth=depth)


def assign_ec_targets(
    pockets: list[PocketRecord],
    *,
    depth: int,
    token_to_index: dict[str, int] | None = None,
) -> tuple[dict[str, int], dict[int, str]]:
    if depth < 1:
        raise ValueError(f"EC label depth must be at least 1, got {depth}.")

    if token_to_index is None:
        tokens = sorted(
            {
                str(token)
                for pocket in pockets
                for token in [pocket.metadata.get("ec_label_token")]
                if token is not None
            }
        )
        token_to_index = {token: idx for idx, token in enumerate(tokens)}
    else:
        token_to_index = dict(token_to_index)

    for pocket in pockets:
        token = pocket.metadata.get("ec_label_token")
        pocket.y_ec = token_to_index.get(str(token)) if token is not None else None

    index_to_token = {idx: token for token, idx in token_to_index.items()}
    return token_to_index, index_to_token


def ec_prefix_from_label_token(label_token: str, *, level: int) -> Optional[str]:
    if level < 1:
        raise ValueError(f"EC prefix level must be at least 1, got {level}.")
    parts = [part for part in str(label_token).split(".") if part]
    if len(parts) < level:
        return None
    return ".".join(parts[:level])


def infer_metal_target_class_from_pocket(
    pocket: PocketRecord,
    *,
    unsupported_metal_policy: str = "error",
) -> Optional[int]:
    observed_symbols = pocket.metadata.get("metal_symbols_observed")
    if isinstance(observed_symbols, list) and observed_symbols:
        return map_site_metal_symbols(
            observed_symbols,
            unsupported_metal_policy=unsupported_metal_policy,
        )

    metal_element = pocket.metal_element.strip()
    if not metal_element:
        return None

    return map_site_metal_symbols(
        metal_element,
        unsupported_metal_policy=unsupported_metal_policy,
    )
