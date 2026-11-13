from __future__ import annotations
from typing import Callable, Literal, Optional

import matplotlib.artist
import matplotlib.axes
import matplotlib.collections
import matplotlib.colors as mcolors
import matplotlib.colors as color
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Dot overlay layouts accepted by the ``overlay`` argument of the plot classes.
OVERLAY_LAYOUTS: tuple[str, ...] = ("swarm", "honeycomb", "scatter")


class _HoneycombLayout:
    """
    Packs the dots of every group onto a honeycomb (hexagonal) lattice.

    Every value is snapped to the nearest lattice row; dots that share a row
    are placed side by side around the group centre, and neighbouring rows are
    shifted by half a cell so the hexagons interlock and never overlap.  All
    dots start at the standard size; if the widest row of any group does not
    fit into ``max_width`` the size of *all* dots is reduced until it does.

    The packing depends on the final axes size and limits, which are only
    known when the figure is rendered, so it runs lazily at draw time and is
    recomputed whenever the axes transform changes (resize, zoom, save dpi).
    """

    # Pointy-top hexagon with circumradius R: rows are 1.5 R apart and
    # neighbours within a row are sqrt(3) R (flat-to-flat width) apart.
    ROW_STEP: float = 1.5
    COL_STEP: float = float(np.sqrt(3))

    def __init__(
        self,
        ax: matplotlib.axes.Axes,
        groups: list[np.ndarray],
        centers: list[float],
        radius_pt: float,
        max_width: float,
        gap: float,
        hexes: matplotlib.collections.PathCollection,
        lines: Optional[matplotlib.collections.LineCollection] = None,
    ) -> None:
        self.ax = ax
        self.groups = groups
        self.centers = centers
        self.max_radius_pt = radius_pt
        self.max_width = max_width
        self.gap = gap
        self.hexes = hexes
        self.lines = lines
        # Lattice circumradius (in points) chosen by the last layout pass.
        self.radius_pt: float = radius_pt
        self._key: Optional[tuple] = None

    # -- geometry helpers -------------------------------------------------- #

    @staticmethod
    def _extent(counts: np.ndarray, parity: np.ndarray) -> np.ndarray:
        """Largest |offset| (in columns) of a row holding ``counts`` dots."""
        return np.where(parity == 0, counts // 2, (counts - 1) // 2 + 0.5)

    def _rows(self, y_px: np.ndarray, y_ref: float, r_px: float) -> tuple:
        """
        Snap values to lattice rows and choose each row's half-cell shift.

        Directly adjacent rows must alternate their shift to interlock.  Each
        block of adjacent occupied rows may start with either shift, so the
        one giving the narrower (more centred) block is chosen.
        """
        k = np.rint((y_px - y_ref) / (self.ROW_STEP * r_px)).astype(np.int64)
        rows, inverse, counts = np.unique(k, return_inverse=True, return_counts=True)
        new_block = np.r_[True, np.diff(rows) > 1]
        block_id = np.cumsum(new_block) - 1
        block_start = np.flatnonzero(new_block)
        pos_in_block = rows - rows[block_start][block_id]
        worst = [
            np.maximum.reduceat(self._extent(counts, (pos_in_block + q) % 2), block_start)
            for q in (0, 1)
        ]
        shift = (worst[1] < worst[0]).astype(np.int64)
        parity = (pos_in_block + shift[block_id]) % 2
        return k, rows, inverse.ravel(), self._extent(counts, parity), parity, block_id

    def _fits(self, packs: list[tuple], y_ref: float, r_px: float) -> bool:
        for y_px, _, half_px in packs:
            if y_px.size:
                extent = self._rows(y_px, y_ref, r_px)[3]
                if (extent.max() + 0.5) * self.COL_STEP * r_px > half_px:
                    return False
        return True

    def _place(
        self, y_px: np.ndarray, xc_px: float, y_ref: float, r_px: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return display coordinates of the hexagon centres of one group."""
        k, rows, inverse, _, parity, block_id = self._rows(y_px, y_ref, r_px)
        row_px = self.ROW_STEP * r_px

        # Rank of every dot inside its row (stable: keeps the input order).
        order = np.argsort(inverse, kind="stable")
        sorted_rows = inverse[order]
        rank = np.empty_like(inverse)
        rank[order] = np.arange(inverse.size) - np.searchsorted(sorted_rows, sorted_rows)

        # Fill rows from the centre outwards: 0, +1, -1, +2 ... for unshifted
        # rows and +.5, -.5, +1.5 ... for shifted ones.  The side that gets the
        # odd dot out alternates every two rows to avoid a lopsided swarm.
        p = parity[inverse]
        side = np.where((rows[inverse] // 2) % 2 == 0, 1.0, -1.0)
        magnitude = np.where(p == 0, (rank + 1) // 2, rank // 2 + 0.5)
        first_side = np.where(p == 0, rank % 2 == 1, rank % 2 == 0)
        offset = magnitude * np.where(first_side, side, -side)

        x_px = xc_px + offset * self.COL_STEP * r_px

        # Blocks of rows separated by empty rows need not share the lattice:
        # centre each block on the range of its snapping errors (so isolated
        # values, e.g. discrete levels, sit exactly at their value and no dot
        # moves more than 0.75 R) while keeping >= 2 R between the facing rows
        # of consecutive blocks, so they cannot touch.
        y_lattice = y_ref + k * row_px
        residual = y_px - y_lattice
        point_block = block_id[inverse]
        n_blocks = int(block_id[-1]) + 1
        res_lo = np.full(n_blocks, np.inf)
        res_hi = np.full(n_blocks, -np.inf)
        np.minimum.at(res_lo, point_block, residual)
        np.maximum.at(res_hi, point_block, residual)
        block_shift = (res_lo + res_hi) / 2
        block_first = np.flatnonzero(np.r_[True, np.diff(block_id) > 0])
        block_last = np.r_[block_first[1:] - 1, rows.size - 1]
        for b in range(1, n_blocks):
            gap_px = (rows[block_first[b]] - rows[block_last[b - 1]]) * row_px
            block_shift[b] = max(block_shift[b], block_shift[b - 1] + 2 * r_px - gap_px)

        return x_px, y_lattice + block_shift[point_block]

    # -- draw-time entry point ---------------------------------------------- #

    def update(self, renderer) -> None:
        trans = self.ax.transData
        px_per_pt = float(renderer.points_to_pixels(1.0))
        key = (
            tuple(trans.transform([[1.0, 1.0], [2.0, 2.0]]).ravel()),
            self.ax.get_xscale(),
            self.ax.get_yscale(),
            px_per_pt,
        )
        if key == self._key:
            return

        packs: list[tuple] = []
        for xc, vals in zip(self.centers, self.groups):
            if not vals.size:
                packs.append((vals, 0.0, 0.0))
                continue
            y_px = trans.transform(np.column_stack([np.full(vals.size, xc), vals]))[:, 1]
            left, mid, right = trans.transform(
                [[xc - self.max_width / 2, vals[0]], [xc, vals[0]], [xc + self.max_width / 2, vals[0]]]
            )[:, 0]
            packs.append((y_px, mid, min(abs(mid - left), abs(right - mid))))

        y_all = np.concatenate([p[0] for p in packs])
        half_all = [p[2] for p in packs if p[0].size]
        if (
            not y_all.size
            or px_per_pt <= 0
            or not np.all(np.isfinite(y_all))
            or min(half_all) <= 0
            or not np.all(np.isfinite(half_all))
        ):
            return  # degenerate axes (e.g. zero size) — keep the previous layout
        self._key = key
        y_ref = float(y_all.min())

        # Largest circumradius not above the standard one that fits every group.
        r_px = self.max_radius_pt * px_per_pt
        if not self._fits(packs, y_ref, r_px):
            n_max = max(p[0].size for p in packs)
            # Even with every dot in one single row this radius fits:
            lo = min(half_all) / (self.COL_STEP * (n_max / 2 + 0.5))
            hi = r_px
            while hi - lo > 1e-3 * hi:
                mid_r = 0.5 * (lo + hi)
                if self._fits(packs, y_ref, mid_r):
                    lo = mid_r
                else:
                    hi = mid_r
            r_px = lo
        self.radius_pt = r_px / px_per_pt

        placed_px = [
            np.column_stack(self._place(y_px, xc_px, y_ref, r_px))
            if y_px.size
            else np.empty((0, 2))
            for y_px, xc_px, _ in packs
        ]
        placed = [trans.inverted().transform(xy) if xy.size else xy for xy in placed_px]

        self.hexes.set_offsets(np.concatenate(placed))
        self.hexes.set_sizes(np.array([(2 * self.radius_pt * (1 - self.gap)) ** 2]))
        if self.lines is not None:
            n_pairs = min(len(xy) for xy in placed)
            self.lines.set_segments(
                [np.array([xy[j] for xy in placed]) for j in range(n_pairs)]
            )


def _run_before_draw(
    artist: matplotlib.artist.Artist,
    callback: Callable,
) -> None:
    """Call ``callback(renderer)`` every time ``artist`` is about to be drawn."""
    original_draw = artist.draw

    def draw(renderer) -> None:
        callback(renderer)
        original_draw(renderer)

    setattr(artist, "draw", draw)


class Helpers:

    def colors_to_rgba(
        self,
        colors: list[str | tuple],
        alpha: float = 0.35,
    ) -> list[tuple[float, float, float, float]]:
        # mcolors.to_rgba returns a 4-tuple; replace only the alpha channel.
        return [(*mcolors.to_rgba(c)[:3], alpha) for c in colors]

    def get_colors(
        self,
        colormap: list[str | tuple] | None,
    ) -> tuple[list, list]:
        """
        Return (edge_colors, fill_colors) for all groups.

        If ``colormap`` is provided each entry is validated; invalid entries
        fall back to black.  Otherwise the ``'Set1'`` matplotlib colormap is
        used with 9 colours.
        """
        # If a colormap is provided, use it;
        # else generate default one with n_colors colors
        # (the best color combination is 9 imho)
        # but we can change it later
        if colormap:
            colors_edge: list = [c if color.is_color_like(c) else "k" for c in colormap]
            colors_fill: list = self.colors_to_rgba(colors_edge)
        else:
            n_colors = 9
            cmap = plt.get_cmap("Set1")
            colors_edge = [cmap(i / n_colors) for i in range(n_colors)]
            colors_edge.insert(0, "k")
            colors_fill = self.colors_to_rgba(colors_edge)
        return colors_edge, colors_fill

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
            if p < 0.001:
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

    def transpose(self, data: list[list]) -> list[list]:
        return list(map(list, zip(*data)))

    def expand_counts(self, counts: list[int]) -> list[int]:
        """
        The input is a list of integers.
        Output is list of matrices.
        Each int represents each output matrix and defines
        how many columns to include in the matrix.
        Eg: input:  [3,2,1]
            output: [0,0,0,1,1,2]
        """
        output: list[int] = []
        counts = list(filter(None, counts))
        for n, c in enumerate(counts, start=0):
            output.extend([n] * c)
        if not output:
            output = [0]
        return output


class BaseStatPlot(Helpers):

    def __init__(
        self,
        data_groups: list[list[float]],
        p_value_exact: Optional[float] = None,
        Test_Name: str = "",
        plot_title: str = "",
        x_label: str = "",
        y_label: str = "",
        print_x_labels: bool = True,
        Paired_Test_Applied: bool = False,
        Paired_Samples: bool = False,
        Groups_Name: Optional[list[str]] = None,
        subgrouping: Optional[list] = None,
        Posthoc_Matrix: Optional[list[list[float]]] = None,
        Posthoc_Tests_Name: str = "",
        colormap: Optional[list] = None,
        print_p_label: bool = True,
        print_stars: bool = True,
        figure_scale_factor: float = 1,
        figure_h: float = 4,
        figure_w: float = 0,  # 0 means auto
        overlay: Optional[str] = None,  # "swarm", "honeycomb", "scatter"; None = plot default
        **kwargs,
    ) -> None:

        if overlay is not None and overlay not in OVERLAY_LAYOUTS:
            raise ValueError(
                f"Unknown overlay {overlay!r}; expected one of {OVERLAY_LAYOUTS}"
            )
        self.overlay: Optional[str] = overlay

        # Sanitise input data — replace empty groups with None; fall back to
        # a dummy two-group dataset so the object is always in a valid state.
        self.data_groups: list[list[float]] = (
            [group if group else None for group in data_groups]  # type: ignore[misc]
            if any(data_groups)
            else [[0], [0]]
        )
        self.n_groups: int = len(self.data_groups)
        self.p: Optional[float] = p_value_exact
        self.testname: str = Test_Name
        self.posthoc_name: str = Posthoc_Tests_Name
        self.posthoc_matrix: list[list[float]] = (
            Posthoc_Matrix if Posthoc_Matrix is not None else []
        )
        self.n_significance_bars: int = 1
        self.dependent: bool = Paired_Test_Applied or Paired_Samples
        self.plot_title: str = plot_title
        self.x_label: str = x_label
        self.y_label: str = y_label
        self.print_p_label: bool = print_p_label
        self.print_stars: bool = print_stars
        self.print_x_labels: bool = print_x_labels
        self.figure_scale_factor: float = figure_scale_factor
        self.figure_h: float = figure_h
        self.figure_w: float = figure_w
        self.error: bool = False

        try:
            assert any(self.data_groups), "There is no input data"
        except AssertionError as error:
            self.error = True
            print("AutoStatLib.StatPlots Error :", error)
            return

        # sd sem mean and median calculation if they are not provided.
        # Convert each group to a float array once; reuse for all four stats.
        # This avoids calling np.std twice per group (old code recomputed it
        # from scratch for sem after already computing it for sd).
        _arrs       = [np.asarray(g, dtype=float) for g in self.data_groups]
        self.n      = [len(a) for a in _arrs]
        self.mean   = [float(a.mean())      for a in _arrs]
        self.median = [float(np.median(a))  for a in _arrs]
        self.sd     = [float(a.std(ddof=1)) for a in _arrs]
        self.sem    = [sd / np.sqrt(n) for sd, n in zip(self.sd, self.n)]
        self.p_printed: str = self.make_p_value_printed(self.p)
        self.stars_printed: str = self.make_stars_printed(self.make_stars(self.p))

        # Pre-compute posthoc matrix string representations once here so that
        # add_significance_bars() doesn't rebuild them on every call.
        if self.posthoc_matrix:
            self._posthoc_printed: list[list[str]] = [
                [self.make_p_value_printed(e) for e in row]
                for row in self.posthoc_matrix
            ]
            self._posthoc_stars: list[list[str]] = [
                [self.make_stars_printed(self.make_stars(e)) for e in row]
                for row in self.posthoc_matrix
            ]
        else:
            self._posthoc_printed = []
            self._posthoc_stars   = []

        self.groups_name: list[str] = Groups_Name if Groups_Name is not None else [""]
        self.subgrouping: list = subgrouping if subgrouping else [0]
        self.subgrouping_arrange: list[int] = self.expand_counts(self.subgrouping)

        _colormap: list = []
        if colormap is not None and colormap != [""]:
            _colormap = colormap
            self.colormap_default: bool = False
        else:
            self.colormap_default = True
        self.colors_edge, self.colors_fill = self.get_colors(_colormap or None)

        self.y_max: float = max(max(data) for data in self.data_groups)

    # ------------------------------------------------------------------ #
    # Figure / axes helpers                                              #
    # ------------------------------------------------------------------ #

    def setup_figure(self) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        fig, ax = plt.subplots(
            dpi=100,
            figsize=(
                (0.5 + 0.9 * self.n_groups) if not self.figure_w else self.figure_w,
                self.figure_h,
            ),
        )
        figure_size = plt.gcf().get_size_inches()
        scaled = self.figure_scale_factor * figure_size
        plt.gcf().set_size_inches((float(scaled[0]), float(scaled[1])))
        return fig, ax

    def add_barplot(
        self,
        ax: matplotlib.axes.Axes,
        x: int,
        fill: bool = True,
        linewidth: float = 2,
        zorder: int = 1,
    ) -> None:
        ax.bar(
            x,
            self.mean[x],
            width=0.75,
            facecolor=self.colors_fill[x % len(self.colors_fill)],
            edgecolor=self.colors_edge[x % len(self.colors_edge)],
            fill=fill,
            linewidth=linewidth * self.figure_scale_factor,
            zorder=zorder,
        )

    def add_violinplot(
        self,
        ax: matplotlib.axes.Axes,
        x: int,
        linewidth: float = 2,
        widths: float = 0.85,
        orientation: Literal["vertical", "horizontal"] = "vertical",
        showmeans: bool = False,
        showmedians: bool = False,
        showextrema: bool = False,
        points: int = 200,
        bw_method: float = 0.5,
    ) -> None:
        vp = ax.violinplot(
            self.data_groups[x],
            positions=[x],
            widths=widths,
            orientation=orientation,
            showmeans=showmeans,
            showmedians=showmedians,
            showextrema=showextrema,
            points=points,
            bw_method=bw_method,
        )
        for pc in vp["bodies"]:  # type: ignore[attr-defined]
            pc.set_facecolor(self.colors_fill[x % len(self.colors_fill)])
            pc.set_edgecolor(self.colors_edge[x % len(self.colors_edge)])
            pc.set_linewidth(linewidth * self.figure_scale_factor)

    def add_boxplot(
        # positions of boxes, defaults to range(1,n+1)
        self,
        ax: matplotlib.axes.Axes,
        positions: Optional[list[int]] = None,
        widths: float = 0.6,
        tickLabels: Optional[list[str]] = None,
        notch: bool = False,
        confidences: Optional[list] = None,
        fliers: bool = False,
        fliersMarker: str = "",
        flierFillColor: Optional[str] = None,
        flierEdgeColor: Optional[str] = None,
        flierLineWidth: float = 2,
        flierLineStyle: Optional[str] = None,
        orientation: Literal["vertical", "horizontal"] = "vertical",
        # whiskers when one float is tukeys parameter, when a pair of percentages,
        # defines the percentiles where the whiskers should be If a float,
        # the lower whisker is at the lowest datum above Q1 - whis*(Q3-Q1),
        # and the upper whisker at the highest datum below Q3 + whis*(Q3-Q1),
        # where Q1 and Q3 are the first and third quartiles. The default value of whis = 1.5
        # corresponds to Tukey's original definition of boxplots.
        whiskers: float = 1.5,
        bootstrap: Optional[int] = None,
        whiskersColor: Optional[str] = None,
        whiskersLineWidth: float = 2,
        whiskersLineStyle: Optional[str] = None,
        showWhiskersCaps: bool = True,
        whiskersCapsWidths: Optional[float] = None,
        whiskersCapsColor: Optional[str] = None,
        whiskersCapsLineWidth: float = 2,
        whiskersCapsLineStyle: Optional[str] = None,
        boxFill: Optional[str] = None,
        boxBorderColor: Optional[str] = None,
        boxBorderWidth: float = 2,
        userMedians: Optional[list[float]] = None,
        medianColor: Optional[str] = None,
        medianLineStyle: Optional[str] = None,
        medianLineWidth: float = 2,
        showMeans: bool = False,
        meanMarker: Optional[str] = None,
        meanFillColor: Optional[str] = None,
        meanEdgeColor: Optional[str] = None,
        meanLine: bool = False,
        meanLineColor: Optional[str] = None,
        meanLineStyle: Optional[str] = None,
        meanLineWidth: float = 2,
        autorange: bool = False,
    ) -> None:

        positions = list(range(self.n_groups))

        fliersMarker = "" if not fliers else (fliersMarker or "b+")

        whiskersCapsStyles: dict = {}
        if whiskersCapsColor is not None:
            whiskersCapsStyles["color"] = whiskersCapsColor
        if whiskersCapsLineWidth is not None:
            whiskersCapsStyles["linewidth"] = whiskersCapsLineWidth
        if whiskersCapsLineStyle is not None:
            whiskersCapsStyles["linestyle"] = whiskersCapsLineStyle

        boxProps: dict = {
            "facecolor": (0, 0, 0, 0),
            "edgecolor": "black",
            "linewidth": 1,
        }
        if boxFill is not None:
            boxProps["facecolor"] = boxFill
        if boxBorderColor is not None:
            boxProps["edgecolor"] = boxBorderColor
        if boxBorderWidth is not None:
            boxProps["linewidth"] = boxBorderWidth

        whiskersProps: dict = {"color": "black", "linestyle": "solid", "linewidth": 1}
        if whiskersColor is not None:
            whiskersProps["color"] = whiskersColor
        if whiskersLineStyle is not None:
            whiskersProps["linestyle"] = whiskersLineStyle
        if whiskersLineWidth is not None:
            whiskersProps["linewidth"] = whiskersLineWidth

        flierProps: dict = {
            "markerfacecolor": [0, 0, 0, 0],
            "markeredgecolor": "black",
            "linestyle": "solid",
            "markeredgewidth": 1,
        }
        if flierFillColor is not None:
            flierProps["markerfacecolor"] = flierFillColor
        if flierEdgeColor is not None:
            flierProps["markeredgecolor"] = flierEdgeColor
        if flierLineWidth is not None:
            flierProps["markeredgewidth"] = flierLineWidth
        if flierLineStyle is not None:
            flierProps["linestyle"] = flierLineStyle

        medianProps: dict = {"linestyle": "solid", "linewidth": 1, "color": "red"}
        if medianColor is not None:
            medianProps["color"] = medianColor
        if medianLineStyle is not None:
            medianProps["linestyle"] = medianLineStyle
        if medianLineWidth is not None:
            medianProps["linewidth"] = medianLineWidth

        meanProps: dict = {
            "color": "black",
            "marker": "o",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "linestyle": "solid",
            "linewidth": 1,
        }
        if meanMarker is not None:
            meanProps["marker"] = meanMarker
        if meanFillColor is not None:
            meanProps["markerfacecolor"] = meanFillColor
        if meanEdgeColor is not None:
            meanProps["markeredgecolor"] = meanEdgeColor
        if meanLineColor is not None:
            meanProps["color"] = meanLineColor
        if meanLineStyle is not None:
            meanProps["linestyle"] = meanLineStyle
        if meanLineWidth is not None:
            meanProps["linewidth"] = meanLineWidth

        bplot = ax.boxplot(
            self.data_groups,
            positions=positions,
            widths=widths,
            notch=notch,
            conf_intervals=confidences,
            sym=fliersMarker,
            flierprops=flierProps,
            orientation=orientation,
            whis=whiskers,
            whiskerprops=whiskersProps,
            showcaps=showWhiskersCaps,
            capwidths=whiskersCapsWidths,
            capprops=whiskersCapsStyles,
            boxprops=boxProps,
            usermedians=userMedians,
            medianprops=medianProps,
            bootstrap=bootstrap,
            showmeans=showMeans,
            meanline=meanLine,
            meanprops=meanProps,
            autorange=autorange,
            patch_artist=True,
        )

        if not self.colormap_default:
            for x, patch in enumerate(bplot["boxes"]):
                patch.set_facecolor(self.colors_fill[x % len(self.colors_fill)])

    def add_scatter(
        self,
        ax: matplotlib.axes.Axes,
        color: str = "k",
        alpha: float = 0.5,
        marker: str = "o",
        markersize: float = 8,
        linewidth: float = 1.2,
        zorder: int = 2,
    ) -> None:
        # Generate all jitter offsets with NumPy at once instead of Python loops.
        rng = np.random.default_rng()
        spread_pool: list[np.ndarray] = [
            i + rng.uniform(-0.10, 0.10, size=len(g))
            for i, g in enumerate(self.data_groups)
        ]

        for i, data in enumerate(self.transpose(self.data_groups)):
            ax.plot(
                self.transpose([list(t) for t in spread_pool])[i],
                data,
                color=color,
                alpha=alpha,
                marker=marker,
                markersize=markersize * self.figure_scale_factor,
                linewidth=linewidth * self.figure_scale_factor,
                linestyle="-" if self.dependent else "",
                zorder=zorder - 1,
            )

    def add_swarm(
        self,
        ax: matplotlib.axes.Axes,
        color: str = "dimgrey",
        default_color: str = "dimgrey",
        alpha: float = 1,
        marker: str = "o",
        markersize: float = 8,
        linewidth: float = 1.4,
        zorder: int = 2,
    ) -> None:
        """
        Add a swarmplot (scatter-like plot with non-overlapping points)
        to the provided Axes. Automatically reduce point size if overcrowded.
        Automatically assigns colors using sns.color_palette("tab10")
        to all unique non-missing group labels.
        Missing labels → default_color.
        """
        values: list[float] = [v for group in self.data_groups for v in group]
        groups: list[int] = [
            i for i, group in enumerate(self.data_groups) for _ in group
        ]

        group_counts: list[int] = [len(g) for g in self.data_groups]
        max_points: int = max(group_counts) if group_counts else 1
        num_groups: int = len(self.data_groups)

        xlim = ax.get_xlim()
        width_per_group: float = (xlim[1] - xlim[0]) / max(num_groups, 1)
        density: float = max_points / (width_per_group + 1e-6)
        size_scale: float = max(0.1, min(1, 3.5 / (density**0.5)))

        sns.swarmplot(
            x=groups,
            y=values,
            ax=ax,
            color=color,
            alpha=alpha,
            size=markersize * self.figure_scale_factor * size_scale,
            marker=marker,
            linewidth=linewidth * self.figure_scale_factor * size_scale,
            zorder=zorder,
            warn_thresh = 1, # threshold for warning about too many points; set to 0 to always warn, or 1 to never warn
        )

        if self.dependent:
            for i, data in enumerate(self.transpose(self.data_groups)):
                ax.plot(
                    range(len(data)),
                    data,
                    color=color,
                    alpha=alpha * 0.25,
                    linewidth=linewidth * self.figure_scale_factor,
                    zorder=zorder - 1,
                )

    def add_swarm_with_alternate_colors(
        self,
        ax: matplotlib.axes.Axes,
        color: str = "dimgrey",
        default_color: str = "dimgrey",
        palette_name: str = "tab10",
        subgrouping: Optional[list] = None,
        alpha: float = 1,
        marker: str = "o",
        markersize: float = 8,
        linewidth: float = 1.4,
        zorder: int = 2,
    ) -> None:
        """
        Add a swarmplot (scatter-like plot with non-overlapping points)
        to the provided Axes. Automatically reduce point size if overcrowded.
        Automatically assigns colors using sns.color_palette("tab10")
        to all unique non-missing group labels.
        Missing labels → default_color.
        """
        if subgrouping is None:
            subgrouping = [0]

        values_flat: list[float] = [v for group in self.data_groups for v in group]
        groups_flat: list[int] = [
            i for i, group in enumerate(self.data_groups) for _ in group
        ]
        values_arr: np.ndarray = np.array(values_flat)

        group_counts: list[int] = [len(g) for g in self.data_groups]
        max_points: int = max(group_counts) if group_counts else 1
        num_groups: int = len(self.data_groups)

        xlim = ax.get_xlim()
        width_per_group: float = (xlim[1] - xlim[0]) / max(num_groups, 1)
        density: float = max_points / (width_per_group + 1e-6)
        size_scale: float = max(0.1, min(1, 3.5 / (density**0.5)))

        normalized_labels: list
        if set(subgrouping) != {0}:
            normalized_labels = [
                lbl if lbl not in (None, "", np.nan, 0) else "_" for lbl in subgrouping
            ]
            len_data = int(len(values_flat) / 2)
            len_lbl = len(normalized_labels)
            if len_lbl < len_data:
                normalized_labels.extend(["last"] * (len_data - len_lbl))
            elif len_lbl > len_data:
                normalized_labels = normalized_labels[:len_data]
        else:
            normalized_labels = ["_" for _ in self.data_groups[0]]

        # Construct row-by-row long-form DataFrame for seaborn
        # df_list = []
        # for col in range(num_groups):
        #     df_list.append(pd.DataFrame({
        #         "value": values,
        #         "x": groups,
        #         "subgroup": normalized_labels[col],
        #     }))
        # df = pd.concat(df_list, ignore_index=True)

        # Extract unique non-default labels
        # unique_subgroups = [g for g in df["subgroup"].unique() if g != "__default__"]

        unique_subgroups: list = list(set(normalized_labels))
        colors = sns.color_palette(palette_name, len(unique_subgroups))
        palette: dict = {g: c for g, c in zip(unique_subgroups, colors)}
        palette["_"] = default_color

        # debugging prints
        # print(values_flat)
        # print(groups_flat)
        # print(subgrouping)
        # print(normalized_labels)

        sns.swarmplot(
            y=values_arr,
            x=groups_flat,
            hue=normalized_labels * num_groups,
            ax=ax,
            palette=palette,
            dodge=False,
            legend=False,
            alpha=alpha,
            size=markersize * self.figure_scale_factor * size_scale,
            marker=marker,
            linewidth=linewidth * self.figure_scale_factor * size_scale,
            zorder=zorder,
        )

    def add_honeycomb(
        self,
        ax: matplotlib.axes.Axes,
        color: str = "dimgrey",
        alpha: float = 1,
        markersize: float = 8,
        max_width: float = 0.8,
        gap: float = 0.12,
        linewidth: float = 1.4,
        zorder: int = 2,
    ) -> _HoneycombLayout:
        """
        Add a honeycomb overlay: every value is drawn as a hexagon packed
        onto a hexagonal lattice, so the dots tile tightly and never overlap.

        Values are snapped to the nearest lattice row (at most 3/4 of a
        hexagon away from their true value).  Hexagons keep the standard
        ``markersize`` (vertex-to-vertex, in points) unless the widest row of
        a group would exceed ``max_width`` (in x-axis units, the distance
        between group centres is 1); then all hexagons shrink equally until
        every group fits.  ``gap`` is the fraction of the size left empty
        between neighbouring hexagons.

        The layout is computed at draw time, so it stays correct after the
        axes are resized, re-limited or saved at a different dpi.
        Returns the layout object; its ``radius_pt`` holds the size in use
        after the figure has been drawn.
        """
        groups = [
            np.asarray(g if g is not None else [], dtype=float).ravel()
            for g in self.data_groups
        ]
        groups = [g[np.isfinite(g)] for g in groups]
        centers = [float(i) for i in range(len(groups))]
        start_xy = np.concatenate(
            [np.column_stack([np.full(g.size, c), g]) for c, g in zip(centers, groups)]
        )

        radius_pt = markersize * self.figure_scale_factor / 2
        hexes = ax.scatter(
            start_xy[:, 0],
            start_xy[:, 1],
            s=(2 * radius_pt * (1 - gap)) ** 2,
            marker="h",  # pointy-top hexagon: tiles in horizontal rows
            color=color,
            alpha=alpha,
            linewidths=0,
            zorder=zorder,
        )

        lines = None
        if self.dependent:
            lines = matplotlib.collections.LineCollection(
                [],
                colors=color,
                alpha=alpha * 0.25,
                linewidths=linewidth * self.figure_scale_factor,
                zorder=zorder - 1,
            )
            ax.add_collection(lines, autolim=False)

        layout = _HoneycombLayout(
            ax, groups, centers, radius_pt, max_width, gap, hexes, lines
        )
        _run_before_draw(hexes, layout.update)
        if lines is not None:
            _run_before_draw(lines, layout.update)

        # Same categorical x-range a seaborn swarm overlay would set.
        ax.set_xlim(-0.5, len(groups) - 0.5)
        return layout

    def add_overlay(self, ax: matplotlib.axes.Axes, default: str = "swarm") -> None:
        """Draw the individual values using the layout chosen by ``overlay``."""
        match self.overlay or default:
            case "honeycomb":
                self.add_honeycomb(ax)
            case "scatter":
                self.add_scatter(ax)
            case _:
                self.add_swarm(ax)

    def add_errorbar_sd(
        self,
        ax: matplotlib.axes.Axes,
        x: int,
        capsize: float = 4,
        ecolor: str = "r",
        linewidth: float = 2,
        zorder: int = 3,
    ) -> None:
        ax.errorbar(
            x,
            self.mean[x],
            yerr=self.sd[x],
            fmt="none",
            capsize=capsize * self.figure_scale_factor,
            ecolor=ecolor,
            linewidth=linewidth * self.figure_scale_factor,
            elinewidth=linewidth * self.figure_scale_factor,
            capthick=linewidth * self.figure_scale_factor,
            zorder=zorder,
        )

    def add_errorbar_sem(
        self,
        ax: matplotlib.axes.Axes,
        x: int,
        capsize: float = 5,
        ecolor: str = "r",
        linewidth: float = 2,
        zorder: int = 3,
    ) -> None:
        ax.errorbar(
            x,
            self.mean[x],
            yerr=self.sem[x],
            fmt="none",
            capsize=capsize * self.figure_scale_factor,
            ecolor=ecolor,
            linewidth=linewidth * self.figure_scale_factor,
            elinewidth=linewidth * self.figure_scale_factor,
            capthick=linewidth * self.figure_scale_factor,
            zorder=zorder,
        )

    def add_mean_marker(
        self,
        ax: matplotlib.axes.Axes,
        x: int,
        marker: str = "_",
        markerfacecolor: str = "#00000000",
        markeredgecolor: str = "r",
        markersize: float = 20,
        linewidth: float = 2,
        zorder: int = 3,
    ) -> None:
        ax.plot(
            x,
            self.mean[x],
            marker=marker,
            markerfacecolor=markerfacecolor,
            markeredgecolor=markeredgecolor,
            markersize=markersize * self.figure_scale_factor,
            markeredgewidth=linewidth * self.figure_scale_factor,
            zorder=zorder,
        )

    def add_median_marker(
        self,
        ax: matplotlib.axes.Axes,
        x: int,
        marker: str = "o",
        markerfacecolor: str = "#FFFFFFFF",
        markeredgecolor: str = "r",
        markersize: float = 6,
        linewidth: float = 2,
        zorder: int = 4,
    ) -> None:
        ax.plot(
            x,
            self.median[x],
            marker=marker,
            markerfacecolor=markerfacecolor,
            markeredgecolor=markeredgecolor,
            markersize=markersize * self.figure_scale_factor,
            markeredgewidth=linewidth * self.figure_scale_factor,
            zorder=zorder,
        )

    def add_significance_bars(
        self,
        ax: matplotlib.axes.Axes,
        linewidth: float = 2,
        capsize: float = 0.01,
        col: str = "k",
    ) -> None:

        # Use the pre-computed representations cached in __init__ rather than
        # rebuilding them on every call to add_significance_bars().
        posthoc_matrix_printed: list[list[str]] = self._posthoc_printed
        posthoc_matrix_stars:   list[list[str]] = self._posthoc_stars

        def draw_bar(
            p: str,
            stars: str,
            order: int = 0,
            x1: int = 0,
            x2: int = self.n_groups - 1,
            capsize: float = capsize,
            linewidth: float = linewidth,
            col: str = col,
        ) -> None:
            label: str
            vspace: float
            match (self.print_p_label, self.print_stars):
                case (True, True):
                    vspace = (capsize + 0.06) * self.figure_scale_factor
                    label = "{}\n{}".format(p, stars)
                case (True, False):
                    vspace = (capsize + 0.03) * self.figure_scale_factor
                    label = "{}".format(p)
                case (False, True):
                    vspace = (capsize + 0.03) * self.figure_scale_factor
                    label = "{}".format(stars)
                case _:
                    return

            if self.print_p_label or self.print_stars:
                y: float = (1.05 + order * vspace) * self.y_max
                h: float = capsize * self.y_max
                ax.plot(
                    [x1, x1, x2, x2],
                    [y, y + h, y + h, y],
                    lw=linewidth * self.figure_scale_factor,
                    c=col,
                )
                ax.text(
                    (x1 + x2) * 0.5,
                    y + h,
                    label,
                    ha="center",
                    va="bottom",
                    color=col,
                    fontweight="bold",
                    fontsize=8 * self.figure_scale_factor,
                )

        def draw_bar_from_posthoc_matrix(x1: int, x2: int, o: int) -> None:
            draw_bar(
                posthoc_matrix_printed[x1][x2],
                posthoc_matrix_stars[x1][x2],
                order=o,
                x1=x1,
                x2=x2,
            )

        # bars_args= []
        # vshift=[0 for _ in self.data_groups]

        # for i in range(len(self.posthoc_matrix)):
        #     for j in range(i+1, len(self.posthoc_matrix[i])):
        #         bars_args.append((i, j, j*3-i*3))
        # for i in bars_args:
        #     draw_bar(i[0], i[1], i[2])

        if (self.p is not None) or self.posthoc_matrix:
            if not self.posthoc_matrix:
                draw_bar(self.p_printed, self.stars_printed)
            elif len(self.posthoc_matrix) == 3:
                draw_bar_from_posthoc_matrix(0, 1, 0)
                draw_bar_from_posthoc_matrix(1, 2, 1)
                draw_bar_from_posthoc_matrix(0, 2, 3)
            elif len(self.posthoc_matrix) == 4:
                draw_bar_from_posthoc_matrix(0, 1, 0)
                draw_bar_from_posthoc_matrix(2, 3, 0)
                draw_bar_from_posthoc_matrix(1, 2, 1)
                draw_bar_from_posthoc_matrix(0, 2, 3)
                draw_bar_from_posthoc_matrix(1, 3, 5)
                draw_bar_from_posthoc_matrix(0, 3, 7)
            elif len(self.posthoc_matrix) == 5:
                draw_bar_from_posthoc_matrix(0, 1, 0)
                draw_bar_from_posthoc_matrix(2, 3, 0)
                draw_bar_from_posthoc_matrix(1, 2, 1)
                draw_bar_from_posthoc_matrix(3, 4, 1)
                draw_bar_from_posthoc_matrix(0, 2, 4)
                draw_bar_from_posthoc_matrix(2, 4, 5)
                draw_bar_from_posthoc_matrix(1, 3, 8)
                draw_bar_from_posthoc_matrix(0, 3, 11)
                draw_bar_from_posthoc_matrix(1, 4, 14)
                draw_bar_from_posthoc_matrix(0, 4, 17)
            else:
                draw_bar(self.p_printed, self.stars_printed)

    def axes_formatting(
        self,
        ax: matplotlib.axes.Axes,
        linewidth: float = 2,
    ) -> None:
        
        # Add a 3% buffer to the y-axis to fit title and p-value
        ymin, ymax = plt.ylim()
        buffer = (ymax - ymin) * 0.03
        plt.ylim(ymin, ymax + buffer)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.xaxis.set_visible(bool(self.x_label or self.print_x_labels))
        plt.tight_layout()

        if self.print_x_labels:
            plt.subplots_adjust(bottom=0.11)
            if self.groups_name != [""]:
                ax.set_xticks(range(self.n_groups))
                ax.set_xticklabels(
                    [
                        self.groups_name[i % len(self.groups_name)]
                        for i in range(self.n_groups)
                    ],
                    fontweight="regular",
                    fontsize=8 * self.figure_scale_factor,
                )
            else:
                ax.set_xticks(range(self.n_groups))
                ax.set_xticklabels(
                    [f"Group {i + 1}" for i in range(self.n_groups)],
                    fontweight="regular",
                    fontsize=8 * self.figure_scale_factor,
                )
        else:
            plt.subplots_adjust(bottom=0.08)
            ax.tick_params(axis="x", which="both", labeltop=False, labelbottom=False)

        for ytick in ax.get_yticklabels():
            ytick.set_fontweight("bold")
        ax.tick_params(width=linewidth * self.figure_scale_factor)
        ax.xaxis.set_tick_params(labelsize=10 * self.figure_scale_factor)
        ax.yaxis.set_tick_params(labelsize=12 * self.figure_scale_factor)
        ax.spines["left"].set_linewidth(linewidth * self.figure_scale_factor)
        ax.tick_params(
            axis="y",
            which="both",
            length=linewidth * 2 * self.figure_scale_factor,
            width=linewidth * self.figure_scale_factor,
        )
        ax.tick_params(axis="x", which="both", length=0)

    def add_titles_and_labels(
        self,
        fig: matplotlib.figure.Figure,
        ax: matplotlib.axes.Axes,
    ) -> None:
        if self.plot_title:
            ax.set_title(
                self.plot_title,
                fontsize=10 * self.figure_scale_factor,
                fontweight="bold",
            )
        if self.x_label:
            ax.set_xlabel(
                self.x_label, fontsize=10 * self.figure_scale_factor, fontweight="bold"
            )
        if self.y_label:
            ax.set_ylabel(
                self.y_label, fontsize=10 * self.figure_scale_factor, fontweight="bold"
            )
        fig.text(
            0.95,
            0.0,
            "{}{}\nn={}".format(
                self.testname,
                (", " + self.posthoc_name) if self.posthoc_name else "",
                str(self.n)[1:-1] if not self.dependent else str(self.n[0]),
            ),
            ha="right",
            va="bottom",
            fontsize=8 * self.figure_scale_factor,
            fontweight="regular",
        )

    def show(self) -> None:
        if not self.error:
            plt.show()

    def save(
        self,
        path: str,
        format: str = "png",
        dpi: int = 150,
        transparent: bool = True,
    ) -> None:
        if not self.error:
            plt.savefig(
                path,
                pad_inches=0.1 * self.figure_scale_factor,
                format=format,
                dpi=dpi,
                transparent=transparent,
            )

    def close(self) -> None:
        if not self.error:
            plt.close()

    def plot(self) -> None:
        """Abstract — each subclass must implement its own plot method."""
        if not self.error:
            raise NotImplementedError("Implement the plot() method in the subclass")


# ─────────────────────────────────────────────────────────────────────────────
# Concrete plot classes
# ─────────────────────────────────────────────────────────────────────────────


class BarStatPlot(BaseStatPlot):

    def plot(self, linewidth: float = 1.8) -> None:  # type: ignore[override]
        if not self.error:
            fig, ax = self.setup_figure()
            for x in range(len(self.data_groups)):
                self.add_barplot(ax, x, linewidth=linewidth)
                self.add_median_marker(ax, x, linewidth=linewidth)
                self.add_mean_marker(ax, x, linewidth=linewidth)
                self.add_errorbar_sd(ax, x, linewidth=linewidth)
            self.add_overlay(ax, default="swarm")
            self.add_significance_bars(ax, linewidth)
            self.add_titles_and_labels(fig, ax)
            self.axes_formatting(ax, linewidth)


class ViolinStatPlot(BaseStatPlot):
    """
    Violin plot, for adjusting see
    https://matplotlib.org/stable/gallery/statistics/customized_violin.html#sphx-glr-gallery-statistics-customized-violin-py
    https://medium.com/@mohammadaryayi/anything-about-violin-plots-in-matplotlib-ffd58a62bbb5

    Kernel Density Estimation (violin shape prediction approach)
    https://scikit-learn.org/stable/modules/density.html

    SeaBorn violins:
    https://seaborn.pydata.org/archive/0.11/generated/seaborn.violinplot.html
    """

    def plot(self, linewidth: float = 1.8) -> None:  # type: ignore[override]
        if not self.error:
            fig, ax = self.setup_figure()
            for x in range(len(self.data_groups)):
                self.add_violinplot(ax, x)
                self.add_median_marker(ax, x, linewidth=linewidth)
                self.add_mean_marker(ax, x, linewidth=linewidth)
                self.add_errorbar_sd(ax, x, linewidth=linewidth)
            self.add_overlay(ax, default="swarm")
            self.add_significance_bars(ax, linewidth)
            self.add_titles_and_labels(fig, ax)
            self.axes_formatting(ax, linewidth)
            xmin, xmax = ax.get_xlim()
            ax.set_xlim(xmin - 0.3, xmax + 0.3)


class BoxStatPlot(BaseStatPlot):

    def plot(self, linewidth: float = 1.8) -> None:  # type: ignore[override]
        if not self.error:
            fig, ax = self.setup_figure()
            self.add_boxplot(ax)
            self.add_overlay(ax, default="swarm")
            self.add_significance_bars(ax, linewidth)
            self.add_titles_and_labels(fig, ax)
            self.axes_formatting(ax, linewidth)


class ScatterStatPlot(BaseStatPlot):

    def plot(self, linewidth: float = 1.8) -> None:  # type: ignore[override]
        if not self.error:
            fig, ax = self.setup_figure()
            for x in range(len(self.data_groups)):
                self.add_median_marker(ax, x, linewidth=linewidth)
                self.add_mean_marker(ax, x, linewidth=linewidth)
                self.add_errorbar_sd(ax, x, linewidth=linewidth)
            self.add_overlay(ax, default="scatter")
            self.add_significance_bars(ax, linewidth)
            self.add_titles_and_labels(fig, ax)
            self.axes_formatting(ax, linewidth)
            xmin, xmax = ax.get_xlim()
            ax.set_xlim(xmin - 0.3, xmax + 0.3)


class SwarmStatPlot(BaseStatPlot):

    def plot(self, linewidth: float = 1.8) -> None:  # type: ignore[override]
        if not self.error:
            fig, ax = self.setup_figure()
            for x in range(len(self.data_groups)):
                self.add_median_marker(ax, x, linewidth=linewidth)
                self.add_mean_marker(ax, x, linewidth=linewidth)
                self.add_errorbar_sd(ax, x, linewidth=linewidth)
            self.add_overlay(ax, default="swarm")
            self.add_significance_bars(ax, linewidth)
            self.add_titles_and_labels(fig, ax)
            self.axes_formatting(ax, linewidth)
            xmin, xmax = ax.get_xlim()
            ax.set_xlim(xmin - 0.3, xmax + 0.3)


class HoneycombStatPlot(BaseStatPlot):
    """
    Mean ± SD and median over a honeycomb overlay: each value is a hexagon
    packed onto a hexagonal lattice.  Hexagons have a standard size and
    shrink uniformly only when needed to fit the widest group.
    The honeycomb overlay is also available on every other plot class via
    ``overlay="honeycomb"``.
    """

    def plot(self, linewidth: float = 1.8) -> None:  # type: ignore[override]
        if not self.error:
            fig, ax = self.setup_figure()
            for x in range(len(self.data_groups)):
                self.add_median_marker(ax, x, linewidth=linewidth)
                self.add_mean_marker(ax, x, linewidth=linewidth)
                self.add_errorbar_sd(ax, x, linewidth=linewidth)
            self.add_overlay(ax, default="honeycomb")
            self.add_significance_bars(ax, linewidth)
            self.add_titles_and_labels(fig, ax)
            self.axes_formatting(ax, linewidth)
            xmin, xmax = ax.get_xlim()
            ax.set_xlim(xmin - 0.3, xmax + 0.3)


class SwarmStatPlot_subgrouping_betta(BaseStatPlot):
    """
    Swarm plot with subgrouping support. Subgrouping is defined by the user as a list of labels (one per data point)
    that indicate which subgroup each data point belongs to.
    The plot will automatically assign different colors to each unique subgroup label,
    and add a legend to indicate which color corresponds to which subgroup.
    Not tested well, use with caution.
    For now, only supports one subgrouping across all groups,
    so the subgrouping list should have the same length as the total number of data points across all groups.
    """

    def plot(self, linewidth: float = 1.8) -> None:  # type: ignore[override]
        if not self.error:
            fig, ax = self.setup_figure()
            for x in range(len(self.data_groups)):
                self.add_median_marker(ax, x, linewidth=linewidth)
                self.add_mean_marker(ax, x, linewidth=linewidth)
                self.add_errorbar_sd(ax, x, linewidth=linewidth)
            self.add_swarm_with_alternate_colors(
                ax, subgrouping=self.subgrouping_arrange
            )
            self.add_significance_bars(ax, linewidth)
            self.add_titles_and_labels(fig, ax)
            self.axes_formatting(ax, linewidth)
            xmin, xmax = ax.get_xlim()
            ax.set_xlim(xmin - 0.3, xmax + 0.3)
