import seaborn as sns
import random
# from math import comb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.colors as color


class Helpers():

    def colors_to_rgba(self, colors, alpha=0.35):
        rgba_colors = []
        for col in colors:
            rgba = list(mcolors.to_rgba(col))
            rgba[3] = alpha
            rgba_colors.append(tuple(rgba))
        return rgba_colors

    def get_colors(self, colormap):
        # If a colormap is provided, use it;
        # else generate default one with n_colors colors
        # (the best color combination is 9 imho)
        # but we can change it later
        if colormap:
            colors_edge = [c if color.is_color_like(
                c) else 'k' for c in colormap]
            colors_fill = self.colors_to_rgba(colors_edge)
        else:
            n_colors = 9  # len(self.data_groups)
            cmap = plt.get_cmap('Set1')
            colors_edge = [cmap(i / n_colors) for i in range(n_colors)]
            colors_edge.insert(0, 'k')
            colors_fill = self.colors_to_rgba(colors_edge)
        return colors_edge, colors_fill

    def make_p_value_printed(self, p) -> str:
        if p is not None:
            if p > 0.99:
                return 'p>0.99'
            elif p >= 0.01:
                return f'p={p:.2g}'
            elif p >= 0.001:
                return f'p={p:.2g}'
            elif p >= 0.0001:
                return f'p={p:.1g}'
            elif p < 0.0001:
                return 'p<0.0001'
            else:
                return 'N/A'
        return 'N/A'

    def make_stars(self, p) -> int:
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

    def make_stars_printed(self, n) -> str:
        return '*' * n if n else 'ns'

    def transpose(self, data):
        return list(map(list, zip(*data)))


class BaseStatPlot(Helpers):

    def __init__(self,
                 data_groups,
                 p_value_exact=None,
                 Test_Name='',
                 Paired_Test_Applied=False,
                 plot_title='',
                 x_label='',
                 y_label='',
                 print_x_labels=True,
                 x_manual_tick_labels=None,
                 Posthoc_Matrix=[],
                 Posthoc_Tests_Name='',
                 colormap=None,
                 print_p_label=True,
                 print_stars=True,
                 figure_scale_factor=1,
                 figure_h=4,
                 figure_w=0,  # 0 means auto
                 **kwargs):
        self.data_groups = [group if group else [0, 0, 0, 0]
                            for group in data_groups]
        self.n_groups = len(self.data_groups)
        self.p = p_value_exact
        self.testname = Test_Name
        self.posthoc_name = Posthoc_Tests_Name
        self.posthoc_matrix = Posthoc_Matrix
        self.n_significance_bars = 1
        self.dependent = Paired_Test_Applied
        self.plot_title = plot_title
        self.x_label = x_label
        self.y_label = y_label
        self.print_p_label = print_p_label
        self.print_stars = print_stars
        self.print_x_labels = print_x_labels
        self.figure_scale_factor = figure_scale_factor
        self.figure_h = figure_h
        self.figure_w = figure_w

        #  sd sem mean and median calculation if they are not provided
        self.mean = [
            np.mean(self.data_groups[i]).item() for i in range(self.n_groups)]
        self.median = [
            np.median(self.data_groups[i]).item() for i in range(self.n_groups)]
        self.sd = [
            np.std(self.data_groups[i]).item() for i in range(self.n_groups)]
        self.sem = [np.std(self.data_groups[i]).item() / np.sqrt(len(self.data_groups[i])).item()
                    for i in range(self.n_groups)]

        self.n = [len(i) for i in self.data_groups]
        self.p_printed = self.make_p_value_printed(self.p)
        self.stars_printed = self.make_stars_printed(self.make_stars(self.p))

        self.x_manual_tick_labels = x_manual_tick_labels if x_manual_tick_labels is not None else [
            '']

        if colormap is not None and colormap != ['']:
            colormap = colormap
            self.colormap_default = False
        else:
            colormap = []
            self.colormap_default = True
        self.colors_edge, self.colors_fill = self.get_colors(colormap)

        self.y_max = max([max(data) for data in self.data_groups])

    def setup_figure(self, ):
        fig, ax = plt.subplots(
            dpi=100,
            figsize=((0.5 + 0.9 * self.n_groups)
                     if not self.figure_w else self.figure_w, self.figure_h)
        )

        figure_size = plt.gcf().get_size_inches()
        plt.gcf().set_size_inches(self.figure_scale_factor * figure_size)

        return fig, ax

    def add_barplot(self, ax, x,
                    fill=True,
                    linewidth=2,
                    zorder=1):

        # Plot bar for mean
        ax.bar(x, self.mean[x],
               width=0.75,
               facecolor=self.colors_fill[x % len(self.colors_fill)],
               edgecolor=self.colors_edge[x % len(self.colors_edge)],
               fill=fill,
               linewidth=linewidth*self.figure_scale_factor,
               zorder=zorder)

    def add_violinplot(self, ax, x,
                       linewidth=2,
                       widths=0.85,
                       vert=True,
                       showmeans=True,
                       showmedians=True,
                       showextrema=True,
                       points=200,
                       bw_method=0.5):

        vp = ax.violinplot(self.data_groups[x], positions=[x], widths=widths, vert=vert,
                           showmeans=showmeans, showmedians=showmedians, showextrema=showextrema,
                           points=points, bw_method=bw_method)

        for pc in vp['bodies']:
            pc.set_facecolor(self.colors_fill[x % len(self.colors_fill)])
            pc.set_edgecolor(self.colors_edge[x % len(self.colors_edge)])
            pc.set_linewidth(linewidth*self.figure_scale_factor)

    def add_boxplot(self, ax,
                    # positions of boxes, defaults to range(1,n+1)
                    positions=None,
                    widths=0.6,
                    tickLabels=None,
                    notch=False,
                    confidences=None,
                    fliers=False,
                    fliersMarker='',
                    flierFillColor=None,
                    flierEdgeColor=None,
                    flierLineWidth=2,
                    flierLineStyle=None,
                    vertical=True,
                    # whiskers when one float is tukeys parameter, when a pair of percentages,
                    # defines the percentiles where the whiskers should be If a float,
                    # the lower whisker is at the lowest datum above Q1 - whis*(Q3-Q1),
                    # and the upper whisker at the highest datum below Q3 + whis*(Q3-Q1),
                    # where Q1 and Q3 are the first and third quartiles. The default value of whis = 1.5
                    # corresponds to Tukey's original definition of boxplots.
                    whiskers=1.5,
                    bootstrap=None,
                    whiskersColor=None,
                    whiskersLineWidth=2,
                    whiskersLineStyle=None,
                    showWhiskersCaps=True,
                    whiskersCapsWidths=None,
                    whiskersCapsColor=None,
                    whiskersCapsLineWidth=2,
                    whiskersCapsLineStyle=None,
                    boxFill=None,
                    boxBorderColor=None,
                    boxBorderWidth=2,
                    userMedians=None,
                    medianColor=None,
                    medianLineStyle=None,
                    medianLineWidth=2,
                    showMeans=False,
                    meanMarker=None,
                    meanFillColor=None,
                    meanEdgeColor=None,
                    meanLine=False,
                    meanLineColor=None,
                    meanLineStyle=None,
                    meanLineWidth=2,
                    autorange=False
                    ):

        positions = list(range(self.n_groups))
        # if (not hasattr(positions, "__len__") or
        #     len(positions) != self.length or
        #         any(not isinstance(x, (int, float)) for x in positions)):
        #     positions = None
        if fliers == False:
            fliersMarker = ""
        else:
            if fliersMarker == "":
                fliersMarker = 'b+'
        # write a function to make a dictionary
        whiskersCapsStyles = dict()
        if whiskersCapsColor != None:
            whiskersCapsStyles["color"] = whiskersCapsColor
        if whiskersCapsLineWidth != None:
            whiskersCapsStyles["linewidth"] = whiskersCapsLineWidth
        if whiskersCapsLineStyle != None:
            whiskersCapsStyles['linestyle'] = whiskersCapsLineStyle

        boxProps = {"facecolor": (0, 0, 0, 0),
                    "edgecolor": "black", "linewidth": 1}
        if boxFill != None:
            boxProps["facecolor"] = boxFill
        if boxBorderColor != None:
            boxProps["edgecolor"] = boxBorderColor
        if boxBorderWidth != None:
            boxProps['linewidth'] = boxBorderWidth
        # if boxBorderStyle != None:
        #     boxProps['linestyle'] = boxBorderStyle  !!!this feature is not working with patch_artist that is needed for facecolor to work

        whiskersProps = {"color": 'black',
                         "linestyle": "solid", "linewidth": 1}
        if whiskersColor != None:
            whiskersProps["color"] = whiskersColor
        if whiskersLineStyle != None:
            whiskersProps["linestyle"] = whiskersLineStyle
        if whiskersLineWidth != None:
            whiskersProps['linewidth'] = whiskersLineWidth

        flierProps = {"markerfacecolor": [
            0, 0, 0, 0], "markeredgecolor": "black", "linestyle": "solid", "markeredgewidth": 1}
        if flierFillColor != None:
            flierProps["markerfacecolor"] = flierFillColor
        if flierEdgeColor != None:
            flierProps["markeredgecolor"] = flierEdgeColor
        if flierLineWidth != None:
            flierProps['markeredgewidth'] = flierLineWidth
        if flierLineStyle != None:
            flierProps['linestyle'] = flierLineStyle
        medianProps = {"linestyle": 'solid', "linewidth": 1, "color": 'red'}
        if medianColor != None:
            medianProps["color"] = medianColor
        if medianLineStyle != None:
            medianProps["linestyle"] = medianLineStyle
        if medianLineWidth != None:
            medianProps['linewidth'] = medianLineWidth

        meanProps = {"color": "black", "marker": 'o', "markerfacecolor": "black",
                     "markeredgecolor": "black", "linestyle": "solid", "linewidth": 1}

        if meanMarker != None:
            meanProps['marker'] = meanMarker
        if meanFillColor != None:
            meanProps["markerfacecolor"] = meanFillColor
        if meanEdgeColor != None:
            meanProps['markeredgecolor'] = meanEdgeColor
        if meanLineColor != None:
            meanProps["color"] = meanLineColor
        if meanLineStyle != None:
            meanProps['linestyle'] = meanLineStyle
        if meanLineWidth != None:
            meanProps['linewidth'] = meanLineWidth

        bplot = ax.boxplot(self.data_groups,
                           positions=positions,
                           widths=widths,
                           # tick_labels=tickLabels,
                           notch=notch,
                           conf_intervals=confidences,
                           sym=fliersMarker,
                           flierprops=flierProps,
                           vert=vertical,
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
                           patch_artist=True)

        # apply use r colormap if provided
        # else left white face with black border
        if not self.colormap_default:
            for x, patch in enumerate(bplot['boxes']):
                patch.set_facecolor(
                    self.colors_fill[x % len(self.colors_fill)])

    def add_scatter(self, ax,
                    color='k',
                    alpha=0.5,
                    marker='o',
                    markersize=8,
                    linewidth=1.2,
                    zorder=2):
        # Generate x jitter pool.
        spread_pool = []  # storing x positions of data points
        for i, data in enumerate(self.data_groups):
            spread = tuple(random.uniform(-.10, .10) for _ in data)
            spread_pool.append(tuple(i + s for s in spread))

        for i, data in enumerate(self.transpose(self.data_groups)):
            # Plot individual data points with x jitter.
            ax.plot(self.transpose(spread_pool)[i], data,
                    color=color,
                    alpha=alpha,
                    marker=marker,
                    markersize=markersize*self.figure_scale_factor,
                    linewidth=linewidth*self.figure_scale_factor,
                    # Connect the data points if desired.
                    linestyle='-' if self.dependent else '',
                    zorder=zorder-1)

    def add_swarm(self, ax,
                  color='grey',
                  alpha=1,
                  marker='o',
                  markersize=8,
                  linewidth=1.4,
                  zorder=2):
        """
        Add a swarmplot (scatter-like plot with non-overlapping points)
        to the provided Axes. Automatically reduce point size if overcrowded.
        """

        # Prepare flattened data
        values = [v for i, group in enumerate(self.data_groups) for v in group]
        groups = [i for i, group in enumerate(self.data_groups) for _ in group]

        # Estimate overcrowding for adaptive sizing
        group_counts = [len(g) for g in self.data_groups]
        max_points = max(group_counts) if group_counts else 1

        # Determine horizontal space per category
        num_groups = len(self.data_groups)
        xlim = ax.get_xlim()
        width_per_group = (xlim[1] - xlim[0]) / max(num_groups, 1)

        # Empirical density threshold: if points are too dense, shrink
        density = max_points / (width_per_group + 1e-6)

        # Tunable constants to approximate best function of size adjustment
        size_scale = max(0.1, min(1, 3.5 / (density ** 0.5)))

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
        )

        # Connect points if data paired
        if self.dependent == True:
            for i, data in enumerate(self.transpose(self.data_groups)):
                ax.plot(range(len(data)), data,
                        color=color,
                        alpha=alpha * 0.7,
                        linewidth=linewidth * self.figure_scale_factor,
                        zorder=zorder - 1)

    def add_errorbar_sd(self, ax, x,
                        capsize=8,
                        ecolor='r',
                        linewidth=2,
                        zorder=3):
        # Add error bars
        ax.errorbar(x, self.mean[x],
                    yerr=self.sd[x],
                    fmt='none',
                    capsize=capsize*self.figure_scale_factor,
                    ecolor=ecolor,
                    linewidth=linewidth*self.figure_scale_factor,
                    zorder=zorder)

    def add_errorbar_sem(self, ax, x,
                         capsize=8,
                         ecolor='r',
                         linewidth=2,
                         zorder=3):
        # Add error bars
        ax.errorbar(x, self.mean[x],
                    yerr=self.sem[x],
                    fmt='none',
                    capsize=capsize*self.figure_scale_factor,
                    ecolor=ecolor,
                    linewidth=linewidth*self.figure_scale_factor,
                    elinewidth=linewidth*self.figure_scale_factor,
                    zorder=zorder)

    def add_mean_marker(self, ax, x,
                        marker='_',
                        markerfacecolor='#00000000',
                        markeredgecolor='r',
                        markersize=16,
                        markeredgewidth=1):
        # Overlay mean marker
        ax.plot(x, self.mean[x],
                marker=marker,
                markerfacecolor=markerfacecolor,
                markeredgecolor=markeredgecolor,
                markersize=markersize*self.figure_scale_factor,
                markeredgewidth=markeredgewidth*self.figure_scale_factor)

    def add_median_marker(self, ax, x,
                          marker='x',
                          markerfacecolor='#00000000',
                          markeredgecolor='r',
                          markersize=10,
                          markeredgewidth=1):
        # Overlay median marker
        ax.plot(x, self.median[x],
                marker=marker,
                markerfacecolor=markerfacecolor,
                markeredgecolor=markeredgecolor,
                markersize=markersize*self.figure_scale_factor,
                markeredgewidth=markeredgewidth*self.figure_scale_factor)

    def add_significance_bars(self, ax,
                              linewidth=2,
                              capsize=0.01,
                              col='k'):

        # # Estimate how many bars needed
        # self.n_significance_bars = comb(
        #     self.n_groups, 2) if self.n_groups > 2 else 1

        posthoc_matrix_printed = [[self.make_p_value_printed(element) for element in row]
                                  for row in self.posthoc_matrix] if self.posthoc_matrix else []
        posthoc_matrix_stars = [[self.make_stars_printed(self.make_stars(element)) for element in row]
                                for row in self.posthoc_matrix] if self.posthoc_matrix else []

        def draw_bar(p, stars, order=0, x1=0, x2=self.n_groups-1, capsize=capsize, linewidth=linewidth, col=col):

            match (self.print_p_label, self.print_stars):
                case (True, True):
                    vspace = (capsize+0.06)*self.figure_scale_factor
                    label = '{}\n{}'.format(p, stars)
                case (True, False):
                    vspace = (capsize+0.03)*self.figure_scale_factor
                    label = '{}'.format(p)
                case (False, True):
                    vspace = (capsize+0.03)*self.figure_scale_factor
                    label = '{}'.format(stars)

            if self.print_p_label or self.print_stars:
                # Draw significance bar connecting x1 and x2 coords
                y, h = ((1.05 + (order*vspace)) *
                        self.y_max), capsize * self.y_max
                ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
                        lw=linewidth*self.figure_scale_factor, c=col)

                ax.text((x1 + x2) * 0.5, y + h, label,
                        ha='center', va='bottom', color=col, fontweight='bold', fontsize=8*self.figure_scale_factor)

        def draw_bar_from_posthoc_matrix(x1, x2, o):
            draw_bar(
                posthoc_matrix_printed[x1][x2], posthoc_matrix_stars[x1][x2], order=o, x1=x1, x2=x2)

        # bars_args= []
        # vshift=[0 for _ in self.data_groups]

        # for i in range(len(self.posthoc_matrix)):
        #     for j in range(i+1, len(self.posthoc_matrix[i])):
        #         bars_args.append((i, j, j*3-i*3))
        # for i in bars_args:
        #     draw_bar(i[0], i[1], i[2])

        if (self.p is not None) or (self.posthoc_matrix != []):
            if not self.posthoc_matrix:
                draw_bar(
                    self.p_printed, self.stars_printed)
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
                draw_bar(
                    self.p_printed, self.stars_printed)

    def axes_formatting(self, ax,
                        linewidth=2):
        # Remove all spines except left
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.xaxis.set_visible(bool(self.x_label or self.print_x_labels))
        plt.tight_layout()

        # Set x ticks and labels
        if self.print_x_labels:
            plt.subplots_adjust(bottom=0.11)
            if self.x_manual_tick_labels != ['']:
                ax.set_xticks(range(self.n_groups))
                ax.set_xticklabels([self.x_manual_tick_labels[i % len(self.x_manual_tick_labels)]
                                    for i in range(self.n_groups)])
            else:
                ax.set_xticks(range(self.n_groups))
                ax.set_xticklabels(['Group {}'.format(i + 1)
                                   for i in range(self.n_groups)], fontweight='regular', fontsize=8*self.figure_scale_factor)
        else:
            plt.subplots_adjust(bottom=0.08)
            ax.tick_params(axis='x', which='both',
                           labeltop=False, labelbottom=False)

        # Additional formatting
        for ytick in ax.get_yticklabels():
            ytick.set_fontweight('bold')
        ax.tick_params(width=linewidth*self.figure_scale_factor)
        ax.xaxis.set_tick_params(labelsize=10*self.figure_scale_factor)
        ax.yaxis.set_tick_params(labelsize=12*self.figure_scale_factor)
        ax.spines['left'].set_linewidth(linewidth*self.figure_scale_factor)
        ax.tick_params(axis='y', which='both',
                       length=linewidth * 2*self.figure_scale_factor, width=linewidth*self.figure_scale_factor)
        ax.tick_params(axis='x', which='both', length=0)

    def add_titles_and_labels(self, fig, ax):
        if self.plot_title:
            ax.set_title(self.plot_title, fontsize=12 *
                         self.figure_scale_factor, fontweight='bold')
        if self.x_label:
            ax.set_xlabel(self.x_label, fontsize=10 *
                          self.figure_scale_factor, fontweight='bold')
        if self.y_label:
            ax.set_ylabel(self.y_label, fontsize=10 *
                          self.figure_scale_factor, fontweight='bold')
        fig.text(0.95, 0.0,
                 '{}{}\nn={}'.format(self.testname, (', ' + self.posthoc_name) if self.posthoc_name else '',
                                     str(self.n)[1:-1] if not self.dependent else str(self.n[0])),
                 ha='right', va='bottom', fontsize=8*self.figure_scale_factor, fontweight='regular')

    def show(self):
        plt.show()

    def save(self, path):
        plt.savefig(path,
                    pad_inches=0.1*self.figure_scale_factor,
                    transparent=True,
                    )

    def close(self):
        plt.close()

    def plot(self):
        # Abstract method—each subclass must implement its own plot method
        raise NotImplementedError(
            "Implement the plot() method in the subclass")


class BarStatPlot(BaseStatPlot):

    def plot(self):
        fig, ax = self.setup_figure()
        linewidth = 2

        for x in range(len(self.data_groups)):

            # Create a bar for given group.
            self.add_barplot(ax, x)

            # Overlay errbars, and markers.
            self.add_median_marker(ax, x)
            self.add_mean_marker(ax, x)
            self.add_errorbar_sd(ax, x)

        self.add_swarm(ax)
        self.add_significance_bars(ax, linewidth)
        self.add_titles_and_labels(fig, ax)
        self.axes_formatting(ax, linewidth)


class ViolinStatPlot(BaseStatPlot):
    '''
        Violin plot, for adjusting see
        https://matplotlib.org/stable/gallery/statistics/customized_violin.html#sphx-glr-gallery-statistics-customized-violin-py
        https://medium.com/@mohammadaryayi/anything-about-violin-plots-in-matplotlib-ffd58a62bbb5

        Kernel Density Estimation (violin shape prediction approach)
        https://scikit-learn.org/stable/modules/density.html

        SeaBorn violins:
        https://seaborn.pydata.org/archive/0.11/generated/seaborn.violinplot.html
    '''

    def plot(self):
        fig, ax = self.setup_figure()
        linewidth = 2

        for x in range(len(self.data_groups)):

            # Create a violin for given group.
            self.add_violinplot(ax, x)

            # Overlay errbars and markers.
            self.add_median_marker(ax, x)
            self.add_mean_marker(ax, x)
            # self.add_errorbar_sd(ax, x)

        self.add_swarm(ax)
        self.add_significance_bars(ax, linewidth)
        self.add_titles_and_labels(fig, ax)
        self.axes_formatting(ax, linewidth)

        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin - 0.3, xmax + 0.3)


class BoxStatPlot(BaseStatPlot):

    def plot(self):
        fig, ax = self.setup_figure()
        linewidth = 2

        self.add_boxplot(ax)
        self.add_swarm(ax)
        self.add_significance_bars(ax, linewidth)
        self.add_titles_and_labels(fig, ax)
        self.axes_formatting(ax, linewidth)


class ScatterStatPlot(BaseStatPlot):

    def plot(self):
        fig, ax = self.setup_figure()
        linewidth = 2

        for x in range(len(self.data_groups)):

            # Overlay errbars, and markers.
            self.add_median_marker(ax, x)
            self.add_mean_marker(ax, x)
            self.add_errorbar_sd(ax, x)

        self.add_scatter(ax)
        self.add_significance_bars(ax, linewidth)
        self.add_titles_and_labels(fig, ax)
        self.axes_formatting(ax, linewidth)

        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin - 0.3, xmax + 0.3)


class SwarmStatPlot(BaseStatPlot):

    def plot(self):
        fig, ax = self.setup_figure()
        linewidth = 2

        for x in range(len(self.data_groups)):

            # Overlay errbars, and markers.
            self.add_median_marker(ax, x)
            self.add_mean_marker(ax, x)
            self.add_errorbar_sd(ax, x)

        self.add_swarm(ax)
        self.add_significance_bars(ax, linewidth)
        self.add_titles_and_labels(fig, ax)
        self.axes_formatting(ax, linewidth)

        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin - 0.3, xmax + 0.3)
