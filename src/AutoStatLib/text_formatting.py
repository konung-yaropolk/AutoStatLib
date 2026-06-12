from __future__ import annotations

from AutoStatLib._protocol import StatAnalysisProtocol

from typing import Optional


class TextFormatting(StatAnalysisProtocol):
    """Text formatting mixin."""

    def _fmt_row(self, elements: list[str], width: int, fill: str = " ") -> str:
        """
        Format a list of strings into a fixed-width columnar row.

        Each element is left-justified to ``width`` characters using ``fill``
        as the pad character.  The last element is appended without trailing
        padding (matches terminal/log output intent).

        Replaces the hand-rolled ``autospace()`` loop with Python's built-in
        ``str.ljust`` and ``str.join``.
        """
        if not elements:
            return ""
        # All but the last element are padded to `width`; last is bare.
        return "".join(e.ljust(width, fill) for e in elements[:-1]) + elements[-1]

    def print_groups(self, space: int = 24, max_length: int = 15) -> None:
        self.log("")
        data: list[list[float]] = self.data
        group_longest: int = max(len(row) for row in data)

        self.log(self._fmt_row(self.groups_name, space))
        self.log(self._fmt_row(["" * 7], space))

        for i in range(group_longest):
            row_values: list[str] = []
            all_values_empty: bool = True
            for row in data:
                if len(row) > max_length:
                    if i < max_length:
                        row_values.append(str(row[i]))
                        all_values_empty = False
                    elif i == max_length:
                        row_values.append(f"[{len(row) - max_length} more]")
                        all_values_empty = False
                    else:
                        continue
                else:
                    if i < len(row):
                        row_values.append(str(row[i]))
                        all_values_empty = False
                    else:
                        row_values.append("")
            if all_values_empty:
                break
            self.log(self._fmt_row(row_values, space))

    def print_results(self) -> None:
        self.log("\n\nResults: \n")
        for i in self.results:
            shift: int = 27 - len(i)
            if i == "Warnings":
                self.log(i, ":", " " * shift, len(self.results[i]))
            elif i == "Posthoc_Tests_Name":
                (
                    self.log(i, ":", " " * shift, self.results[i])
                    if self.results[i] != ""
                    else "N/A"
                )
            elif i == "Posthoc_Matrix":
                self.log(
                    i,
                    ":",
                    " " * shift,
                    (
                        "{0}x{0} matrix".format(len(self.results[i]))
                        if self.results[i]
                        else "N/A"
                    ),
                )
            elif i in (
                "Samples",
                "Posthoc_Matrix_bool",
                "Posthoc_Matrix_printed",
                "Posthoc_Matrix_stars",
            ):
                pass
            else:
                self.log(i, ":", " " * shift, self.results[i])

    def make_p_value_printed(self, p: Optional[float]) -> str:
        if p is not None:
            if p > 0.99:
                return "p>0.99"
            elif p >= 0.01:
                return f"p={p:.2g}"
            elif p >= 0.001:
                return f"p={p:.2g}"
            elif p >= 0.0001:
                return f"p={p:.1g}"
            elif p < 0.0001:
                return "p<0.0001"
            else:
                return "N/A"
        return "N/A"

    def make_stars(self, p: Optional[float]) -> int:
        if p is not None:
            if p < 0.0001:
                return 4
            elif p < 0.001:
                return 3
            elif p < 0.01:
                return 2
            elif p < 0.05:
                return 1
            else:
                return 0
        return 0

    def make_stars_printed(self, n: int) -> str:
        return "*" * n if n else "ns"
