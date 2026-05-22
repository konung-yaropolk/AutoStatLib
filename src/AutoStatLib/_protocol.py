"""
_protocol.py
Shared structural type that declares all attributes the mixin classes
(Helpers, TextFormatting, NormalityTests, StatisticalTests) rely on.

Mypy cannot see cross-mixin attribute access — it only checks each class
in isolation.  By having every mixin inherit from StatAnalysisProtocol
(which declares every shared attribute as abstract) mypy is satisfied
without any runtime cost or real inheritance change.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Optional
import numpy as np


class StatAnalysisProtocol:
    """
    Structural contract for the mixin classes.

    Every attribute listed here is actually set in StatisticalAnalysis.__init__.
    The mixins reference this class only so that mypy can resolve cross-mixin
    attribute access; it adds no runtime behaviour.
    """

    # ── data ──────────────────────────────────────────────────────────────
    data:                           list[list[float]]
    groups_list:                    list[list]
    groups_name:                    list[str]
    n_groups:                       int
    subgrouping:                    list

    # ── test configuration ────────────────────────────────────────────────
    paired:                         bool
    tails:                          int
    popmean:                        Optional[float]
    posthoc:                        bool
    verbose:                        bool
    raise_errors:                   bool
    parametric:                     Optional[bool]

    # ── state ─────────────────────────────────────────────────────────────
    results:                        Optional[dict]
    error:                          bool
    warnings:                       list[str]
    normals:                        list[bool]
    test_name:                      str
    test_id:                        Optional[str]
    test_stat:                      Optional[np.float64]
    p_value:                        Optional[np.float64]
    paired_test_applied:            bool
    posthoc_matrix_df:              Optional[object]
    posthoc_matrix:                 list[list[float]]
    posthoc_name:                   str
    summary:                        str
    warning_flag_non_numeric_data:  bool
    successfull:                    bool
    stars_int:                      Optional[int]
    stars_str:                      str

    # ── test ID lists ─────────────────────────────────────────────────────
    test_ids_all:                   list[str]
    test_ids_parametric:            list[str]
    test_ids_dependent:             list[str]
    test_ids_3sample:               list[str]
    test_ids_2sample:               list[str]
    test_ids_1sample:               list[str]
    warning_ids_all:                dict[str, str]

    # ── cross-mixin methods ───────────────────────────────────────────────
    @abstractmethod
    def log(self, *args: object, **kwargs: object) -> None: ...

    @abstractmethod
    def AddWarning(self, warning_id: str) -> None: ...

    @abstractmethod
    def make_stars(self, p: Optional[float]) -> int: ...

    @abstractmethod
    def make_stars_printed(self, n: int) -> str: ...

    @abstractmethod
    def make_p_value_printed(self, p: Optional[float]) -> str: ...

    @abstractmethod
    def list_to_matrix(self, values: list[float], n: int) -> list[list[float]]: ...

    @abstractmethod
    def matrix_to_dataframe(self, matrix: list[list[float]]): ...

    @abstractmethod
    def check_normality(self, data: list[float]) -> tuple: ...

    @abstractmethod
    def run_test_by_id(self, test_id: str) -> None: ...

    @abstractmethod
    def run_test_auto(self) -> None: ...

    @abstractmethod
    def print_groups(self, space: int = 24, max_length: int = 15) -> None: ...

    @abstractmethod
    def print_results(self) -> None: ...

    @abstractmethod
    def create_results_dict(self) -> dict: ...

    @abstractmethod
    def floatify_recursive(self, data) -> list: ...