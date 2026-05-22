from __future__ import annotations

from AutoStatLib._protocol import StatAnalysisProtocol

from typing import Optional

import numpy as np
import pandas as pd


class Helpers(StatAnalysisProtocol):

    def matrix_to_dataframe(self, matrix: list[list[float]]) -> pd.DataFrame:
        data: list[float] = []
        cols: list[int] = []
        rows: list[int] = []

        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                data.append(value)
                cols.append(i)
                rows.append(j)

        return pd.DataFrame({"Row": rows, "Col": cols, "Value": data})

    def list_to_matrix(self, values: list[float], n: int) -> list[list[float]]:
        i = 0
        matrix: list[list[float]] = [[1.0 for _ in range(n)] for _ in range(n)]
        for ax0 in range(n):
            for ax1 in range(ax0 + 1, n):
                matrix[ax0][ax1] = values[i]
                matrix[ax1][ax0] = values[i]
                i += 1
        return matrix

    def floatify_recursive(self, data) -> list:  # type: ignore[override]
        """
        Recursively convert all items in a nested list to np.float64.
        Non-convertible values are replaced with None and then dropped.
        Returns a flat or nested list of np.float64.
        """
        if isinstance(data, list):
            processed_list = [self.floatify_recursive(item) for item in data]
            return [item for item in processed_list if item is not None]
        else:
            try:
                # Try to convert the item to float
                return np.float64(data)  # type: ignore[return-value]
            except (ValueError, TypeError):
                # If conversion fails, replace with None
                self.warning_flag_non_numeric_data = True
                return None  # type: ignore[return-value]

    def create_results_dict(self) -> dict:
        if self.p_value is not None:
            self.successfull = True
        else:
            self.successfull = False
            self.error = True

        self.stars_int: Optional[int] = (
            self.make_stars(self.p_value.item()) if self.successfull else None
        )
        self.stars_str: str = (
            self.make_stars_printed(self.stars_int) if self.successfull else ""
        )

        return {
            "p_value": (
                self.make_p_value_printed(self.p_value.item())
                if self.successfull
                else None
            ),
            "Significance(p<0.05)": (
                True if self.successfull and self.p_value.item() < 0.05 else False
            ),
            "Stars_Printed": self.stars_str,
            "Test_Name": self.test_name,
            "Groups_Compared": self.n_groups,
            "Population_Mean": self.popmean if self.n_groups == 1 else "N/A",
            "Data_Normaly_Distributed": self.parametric if self.successfull else None,
            "Parametric_Test_Applied": (
                True if self.test_id in self.test_ids_parametric else False
            ),
            "Paired_Test_Applied": (
                self.paired_test_applied if self.successfull else None
            ),
            "Tails": self.tails,
            "p_value_exact": self.p_value.item() if self.successfull else None,
            "Stars": self.stars_int,
            "Warnings": self.warnings,
            "Successfull_Test": (self.successfull and not self.error),
            "Groups_Name": self.groups_name,
            "Groups_N": [len(self.data[i]) for i in range(len(self.data))],
            "Groups_Median": [
                np.median(self.data[i]).item() for i in range(len(self.data))
            ],
            "Groups_Mean": [
                np.mean(self.data[i]).item() for i in range(len(self.data))
            ],
            "Groups_SD": [
                np.std(self.data[i], ddof=1).item() for i in range(len(self.data))
            ],
            "Groups_SE": [
                np.std(self.data[i], ddof=1).item() / np.sqrt(len(self.data[i]))
                for i in range(len(self.data))
            ],
            "subgrouping": self.subgrouping,
            # actually returns list of lists of numpy dtypes of float64, next make it return regular floats:
            "Samples": self.data,
            "Posthoc_Tests_Name": (
                self.posthoc_name if self.posthoc_name is not None else ""
            ),
            "Posthoc_Matrix": self.posthoc_matrix if self.posthoc_matrix else [],
            "Posthoc_Matrix_bool": (
                [[bool(element) for element in row] for row in self.posthoc_matrix]
                if self.posthoc_matrix
                else []
            ),
            "Posthoc_Matrix_printed": (
                [
                    [self.make_p_value_printed(element) for element in row]
                    for row in self.posthoc_matrix
                ]
                if self.posthoc_matrix
                else []
            ),
            "Posthoc_Matrix_stars": (
                [
                    [
                        self.make_stars_printed(self.make_stars(element))
                        for element in row
                    ]
                    for row in self.posthoc_matrix
                ]
                if self.posthoc_matrix
                else []
            ),
        }

    def log(self, *args: object, **kwargs: object) -> None:
        message = " ".join(map(str, args))
        self.summary += "\n" + message

    def AddWarning(self, warning_id: str) -> None:
        message: str = self.warning_ids_all[warning_id]
        self.log(message)
        self.warnings.append(message)
