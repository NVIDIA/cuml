# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define and validate typed benchmark suite manifest schemas."""

from __future__ import annotations

import sys
from typing import Annotated, Any, Literal

import msgspec

NonEmptyString = Annotated[str, msgspec.Meta(min_length=1)]
PositiveCount = Annotated[int, msgspec.Meta(ge=1)]
# The upper bound rejects infinity as well as nonpositive and NaN timeouts.
PositiveTimeout = Annotated[float, msgspec.Meta(gt=0, le=sys.float_info.max)]
ElementType = Literal["float32", "float64", "int32", "int64"]
Providers = Annotated[list[NonEmptyString], msgspec.Meta(min_length=1)]
InputSelection = Annotated[
    list[Literal["X", "y"]], msgspec.Meta(min_length=1, max_length=2)
]


class DtypeMapping(msgspec.Struct, forbid_unknown_fields=True):
    """Specify separate feature and target element types."""

    X: ElementType
    y: ElementType


class ShapeManifest(msgspec.Struct, forbid_unknown_fields=True):
    """Specify measured dimensions and training rows required for inference."""

    rows: PositiveCount
    features: PositiveCount
    train_rows: PositiveCount | msgspec.UnsetType = msgspec.UNSET


class DatasetManifest(msgspec.Struct, forbid_unknown_fields=True):
    """Specify a generated dataset's shape, types, and representation."""

    generator: NonEmptyString
    shape: ShapeManifest
    parameters: dict[str, Any] = msgspec.field(default_factory=dict)
    dtype: ElementType | DtypeMapping = "float32"
    format: Literal["dense", "csr"] = "dense"


class CaseManifest(msgspec.Struct, forbid_unknown_fields=True):
    """Specify an estimator operation and its dataset inputs."""

    estimator: NonEmptyString
    dataset: DatasetManifest
    operation: NonEmptyString
    parameters: dict[str, Any]
    input_selection: InputSelection
    # Omission inherits the suite providers; a list narrows applicability.
    providers: Providers | None = None
    # Required for inference, forbidden for fitting (validated by the loader).
    fit_input_selection: InputSelection | None = None
    # Omission inherits the profile timeout; explicit null disables it.
    timeout_sec: PositiveTimeout | None | msgspec.UnsetType = msgspec.UNSET


class ProfileManifest(msgspec.Struct, forbid_unknown_fields=True):
    """Specify execution counts, dataset scaling, and case timeout defaults."""

    warmups: Annotated[int, msgspec.Meta(ge=0)]
    repetitions: PositiveCount
    size_scale: Annotated[float, msgspec.Meta(gt=0, le=1)]
    timeout_sec: PositiveTimeout | None = None


ProfileMapping = Annotated[
    dict[NonEmptyString, ProfileManifest], msgspec.Meta(min_length=1)
]


class SuiteManifest(msgspec.Struct, forbid_unknown_fields=True):
    """Specify supported providers, profiles, and shared workloads."""

    version: Literal[2]
    name: NonEmptyString
    providers: Providers
    profiles: ProfileMapping
    cases: Annotated[list[CaseManifest], msgspec.Meta(min_length=1)]


def convert_manifest(document: Any) -> SuiteManifest:
    """Validate and convert a suite document to a typed manifest.

    Parameters
    ----------
    document : Any
        Parsed suite manifest data.
    """
    return msgspec.convert(document, type=SuiteManifest, strict=True)


def convert_case(document: Any) -> CaseManifest:
    """Validate and convert a case document to a typed manifest.

    Parameters
    ----------
    document : Any
        Parsed case manifest data.
    """
    return msgspec.convert(document, type=CaseManifest, strict=True)


def convert_profile(document: Any) -> ProfileManifest:
    """Validate and convert profile settings to a typed manifest.

    Parameters
    ----------
    document : Any
        Parsed execution profile data.
    """
    return msgspec.convert(document, type=ProfileManifest, strict=True)


def convert_profiles(document: Any) -> dict[str, ProfileManifest]:
    """Validate and convert named execution profiles.

    Parameters
    ----------
    document : Any
        Mapping of profile names to parsed settings.
    """
    return msgspec.convert(document, type=ProfileMapping, strict=True)


def suite_manifest_json_schema() -> dict[str, Any]:
    """Return the JSON Schema for typed suite manifests."""
    return msgspec.json.schema(SuiteManifest)
