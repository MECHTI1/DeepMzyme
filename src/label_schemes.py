from __future__ import annotations

import os

VALID_UNSUPPORTED_METAL_POLICY_CHOICES = ("error", "skip")


def normalize_metal_label_scheme_name(raw_name: str) -> str:
    normalized = raw_name.strip().lower()
    try:
        return METAL_LABEL_SCHEME_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported DEEPGM metal label scheme {raw_name!r}. "
            f"Expected one of {sorted(METAL_LABEL_SCHEME_ALIASES)}."
        ) from exc

EC_TOP_LEVEL_LABELS = {
    1: "Oxidoreductase",
    2: "Transferase",
    3: "Hydrolase",
    4: "Lyase",
    5: "Isomerase",
    6: "Ligase",
    7: "Translocase",
}
N_EC_CLASSES = len(EC_TOP_LEVEL_LABELS)

METAL_LABEL_SCHEME_ALIASES = {
    "split_fe": "split_fe",
    "split_fe_co_ni": "split_fe",
    "split_all_metals": "split_all_metals",
    "split_by_metal": "split_all_metals",
    "split_each_metal": "split_all_metals",
    "merge_fe_class_viii": "merge_fe_class_viii",
    "merged_fe": "merge_fe_class_viii",
}

METAL_LABEL_SCHEMES = {
    "split_fe": (
        {
            0: "Mn",
            1: "Cu",
            2: "Zn",
            3: "Fe",
            4: "Class VIII",
        },
        {
            "MN": 0,
            "CU": 1,
            "ZN": 2,
            "FE": 3,
            "CO": 4,
            "NI": 4,
        },
    ),
    "split_all_metals": (
        {
            0: "Mn",
            1: "Cu",
            2: "Zn",
            3: "Fe",
            4: "Co",
            5: "Ni",
        },
        {
            "MN": 0,
            "CU": 1,
            "ZN": 2,
            "FE": 3,
            "CO": 4,
            "NI": 5,
        },
    ),
    "merge_fe_class_viii": (
        {
            0: "Mn",
            1: "Cu",
            2: "Zn",
            3: "Class VIII",
        },
        {
            "MN": 0,
            "CU": 1,
            "ZN": 2,
            "FE": 3,
            "CO": 3,
            "NI": 3,
        },
    ),
}


def _normalize_unsupported_metal_policy(policy: str) -> str:
    normalized = policy.strip().lower()
    if normalized not in VALID_UNSUPPORTED_METAL_POLICY_CHOICES:
        raise ValueError(
            f"Unsupported metal policy {policy!r}. "
            f"Expected one of {list(VALID_UNSUPPORTED_METAL_POLICY_CHOICES)}."
        )
    return normalized


def _normalize_site_metal_symbols(symbols: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(symbols, str):
        symbols = (symbols,)
    return tuple(symbol.strip().upper() for symbol in symbols)


def _validate_metal_label_schemes() -> None:
    for alias, scheme_name in METAL_LABEL_SCHEME_ALIASES.items():
        if scheme_name not in METAL_LABEL_SCHEMES:
            raise ValueError(f"Metal label scheme alias {alias!r} points to unknown scheme {scheme_name!r}.")

    for scheme_name, (labels, symbol_to_target) in METAL_LABEL_SCHEMES.items():
        expected_ids = set(range(len(labels)))
        actual_ids = set(labels)
        if actual_ids != expected_ids:
            raise ValueError(
                f"Metal label scheme {scheme_name!r} has non-contiguous label ids: "
                f"expected {sorted(expected_ids)}, got {sorted(actual_ids)}."
            )

        unknown_target_ids = sorted(set(symbol_to_target.values()) - actual_ids)
        if unknown_target_ids:
            raise ValueError(
                f"Metal label scheme {scheme_name!r} maps symbols to unknown target ids "
                f"{unknown_target_ids}."
            )


def active_metal_label_scheme_name() -> str:
    return normalize_metal_label_scheme_name(os.environ.get("DEEPGM_METAL_LABEL_SCHEME", "split_fe"))


def metal_labels_for_scheme(scheme_name: str) -> dict[int, str]:
    normalized = normalize_metal_label_scheme_name(scheme_name)
    labels, _symbol_to_target = METAL_LABEL_SCHEMES[normalized]
    return dict(labels)


def metal_symbol_to_target_for_scheme(scheme_name: str) -> dict[str, int]:
    normalized = normalize_metal_label_scheme_name(scheme_name)
    _labels, symbol_to_target = METAL_LABEL_SCHEMES[normalized]
    return dict(symbol_to_target)


def map_site_metal_symbols_with_mapping(
    symbols: str | tuple[str, ...] | list[str],
    *,
    symbol_to_target: dict[str, int],
    unsupported_metal_policy: str = "error",
) -> int | None:
    policy = _normalize_unsupported_metal_policy(unsupported_metal_policy)
    normalized_symbols = _normalize_site_metal_symbols(symbols)

    target_ids = set()
    for normalized in normalized_symbols:
        try:
            target_ids.add(symbol_to_target[normalized])
        except KeyError as exc:
            if policy == "skip":
                return None
            raise ValueError(
                f"Unsupported site metal symbol {normalized!r}. "
                f"Expected one of {sorted(symbol_to_target)}."
            ) from exc

    return next(iter(target_ids)) if len(target_ids) == 1 else None


def map_site_metal_symbols_for_scheme(
    symbols: str | tuple[str, ...] | list[str],
    *,
    scheme_name: str,
    unsupported_metal_policy: str = "error",
) -> int | None:
    return map_site_metal_symbols_with_mapping(
        symbols,
        symbol_to_target=metal_symbol_to_target_for_scheme(scheme_name),
        unsupported_metal_policy=unsupported_metal_policy,
    )

_validate_metal_label_schemes()
ACTIVE_METAL_LABEL_SCHEME = active_metal_label_scheme_name()
METAL_TARGET_LABELS = metal_labels_for_scheme(ACTIVE_METAL_LABEL_SCHEME)
METAL_SYMBOL_TO_TARGET = metal_symbol_to_target_for_scheme(ACTIVE_METAL_LABEL_SCHEME)
N_METAL_CLASSES = len(METAL_TARGET_LABELS)


def map_site_metal_symbols(
    symbols: str | tuple[str, ...] | list[str],
    *,
    unsupported_metal_policy: str = "error",
) -> int | None:
    return map_site_metal_symbols_with_mapping(
        symbols,
        symbol_to_target=METAL_SYMBOL_TO_TARGET,
        unsupported_metal_policy=unsupported_metal_policy,
    )
