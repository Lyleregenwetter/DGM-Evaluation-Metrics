from abc import ABCMeta, abstractmethod
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import itertools as it
import pandas as pd
import seaborn as sns

def trainingplots(all_status, savedir):
    statarray = np.zeros((len(all_status), len(all_status[0])))
    columns = []
    n = len(all_status[0])
    for j in range(n):
        for i in range(len(all_status)):
            statarray[i,j] = all_status[i][j][1].numpy()
        try: 
            columns.append(all_status[0][j][0].numpy().decode())
        except:
            columns.append(all_status[0][j][0])
    results = pd.DataFrame(statarray, columns = columns)
    if len(results.index)>400:
        results = results.rolling(len(results.index)//200, axis=0, win_type = "gaussian").sum(std=20)/50 #Smoothing
    colors = ["#DC267F", "#648FFF", "#42b27a", "#785EF0", "#FFB000", "#FE6100", "#555555", "#999999"]
    palette = colors*int(np.ceil(n/len(colors)))
    plt.figure(figsize=(10,6))
    sns.lineplot(data=results, palette=palette[:len(results.columns)])
    plt.savefig(f"{savedir}.png")
    plt.close()

    arrangement = SquareStrategy().get_grid_arrangement(n)
    grid = SquareStrategy().get_grid(n)
    fig = plt.figure(figsize=(max(arrangement)*10, 6*len(arrangement)))
    for i, sub in enumerate(grid):
        ax = fig.add_subplot(sub)
        sns.lineplot(ax = ax, data=results.iloc[:,i], color=palette[i])
        ax.set_ylabel("")
        ax.set_title(results.columns[i])
    fig.tight_layout()
    plt.savefig(f"{savedir}_indiv.png")
    plt.close()

#The following taken from https://github.com/matplotlib/grid-strategy
class GridStrategy(metaclass=ABCMeta):
    """
    Static class used to compute grid arrangements given the number of subplots
    you want to show. By default, it goes for a symmetrical arrangement that is
    nearly square (nearly equal in both dimensions).
    """

    def __init__(self, alignment="center"):
        self.alignment = alignment

    def get_grid(self, n):
        """
        Return a list of axes designed according to the strategy.
        Grid arrangements are tuples with the same length as the number of rows,
        and each element specifies the number of colums in the row.
        Ex (2, 3, 2) leads to the shape
             x x
            x x x
             x x
        where each x would be a subplot.
        """

        grid_arrangement = self.get_grid_arrangement(n)
        return self.get_gridspec(grid_arrangement)

    @classmethod
    @abstractmethod
    def get_grid_arrangement(cls, n):  # pragma: nocover
        pass

    def get_gridspec(self, grid_arrangement):
        nrows = len(grid_arrangement)
        ncols = max(grid_arrangement)

        # If it has justified alignment, will not be the same as the other alignments
        if self.alignment == "justified":
            return self._justified(nrows, grid_arrangement)
        else:
            return self._ragged(nrows, ncols, grid_arrangement)

    def _justified(self, nrows, grid_arrangement):
        ax_specs = []
        num_small_cols = np.lcm.reduce(grid_arrangement)
        gs = gridspec.GridSpec(
            nrows, num_small_cols, figure=plt.figure(constrained_layout=True)
        )
        for r, row_cols in enumerate(grid_arrangement):
            skip = num_small_cols // row_cols
            for col in range(row_cols):
                s = col * skip
                e = s + skip

                ax_specs.append(gs[r, s:e])
        return ax_specs

    def _ragged(self, nrows, ncols, grid_arrangement):
        if len(set(grid_arrangement)) > 1:
            col_width = 2
        else:
            col_width = 1

        gs = gridspec.GridSpec(
            nrows, ncols * col_width, figure=plt.figure(constrained_layout=True)
        )

        ax_specs = []
        for r, row_cols in enumerate(grid_arrangement):
            # This is the number of missing columns in this row. If some rows
            # are a different width than others, the column width is 2 so every
            # column skipped at the beginning is also a missing slot at the end.
            if self.alignment == "left":
                # This is left-justified (or possibly full justification)
                # so no need to skip anything
                skip = 0
            elif self.alignment == "right":
                # Skip two slots for every missing plot - right justified.
                skip = (ncols - row_cols) * 2
            else:
                # Defaults to centered, as that is the default value for the class.
                # Skip one for each missing column - centered
                skip = ncols - row_cols

            for col in range(row_cols):
                s = skip + col * col_width
                e = s + col_width

                ax_specs.append(gs[r, s:e])

        return ax_specs


class SquareStrategy(GridStrategy):
    SPECIAL_CASES = {3: (2, 1), 5: (2, 3)}

    @classmethod
    def get_grid_arrangement(cls, n):
        """
        Return an arrangement of rows containing ``n`` axes that is as close to
        square as looks good.
        :param n:
            The number of plots in the subplot
        :return:
            Returns a  :class:`tuple` of length ``nrows``, where each element
            represents the number of plots in that row, so for example a 3 x 2
            grid would be represented as ``(3, 3)``, because there are 2 rows
            of length 3.
        **Example:**
        .. code::
            >>> GridStrategy.get_grid(7)
            (2, 3, 2)
            >>> GridStrategy.get_grid(6)
            (3, 3)
        """
        if n in cls.SPECIAL_CASES:
            return cls.SPECIAL_CASES[n]

        # May not work for very large n
        n_sqrtf = np.sqrt(n)
        n_sqrt = int(np.ceil(n_sqrtf))

        if n_sqrtf == n_sqrt:
            # Perfect square, we're done
            x, y = n_sqrt, n_sqrt
        elif n <= n_sqrt * (n_sqrt - 1):
            # An n_sqrt x n_sqrt - 1 grid is close enough to look pretty
            # square, so if n is less than that value, will use that rather
            # than jumping all the way to a square grid.
            x, y = n_sqrt, n_sqrt - 1
        elif not (n_sqrt % 2) and n % 2:
            # If the square root is even and the number of axes is odd, in
            # order to keep the arrangement horizontally symmetrical, using a
            # grid of size (n_sqrt + 1 x n_sqrt - 1) looks best and guarantees
            # symmetry.
            x, y = (n_sqrt + 1, n_sqrt - 1)
        else:
            # It's not a perfect square, but a square grid is best
            x, y = n_sqrt, n_sqrt

        if n == x * y:
            # There are no deficient rows, so we can just return from here
            return tuple(x for i in range(y))

        # If exactly one of these is odd, make it the rows
        if (x % 2) != (y % 2) and (x % 2):
            x, y = y, x

        return cls.arrange_rows(n, x, y)

    @classmethod
    def arrange_rows(cls, n, x, y):
        """
        Given a grid of size (``x`` x ``y``) to be filled with ``n`` plots,
        this arranges them as desired.
        :param n:
            The number of plots in the subplot.
        :param x:
            The number of columns in the grid.
        :param y:
            The number of rows in the grid.
        :return:
            Returns a :class:`tuple` containing a grid arrangement, see
            :func:`get_grid` for details.
        """
        part_rows = (x * y) - n
        full_rows = y - part_rows

        f = (full_rows, x)
        p = (part_rows, x - 1)

        # Determine which is the more and less frequent value
        if full_rows >= part_rows:
            size_order = f, p
        else:
            size_order = p, f

        # ((n_more, more_val), (n_less, less_val)) = size_order
        args = it.chain.from_iterable(size_order)

        if y % 2:
            return cls.stripe_odd(*args)
        else:
            return cls.stripe_even(*args)

    @classmethod
    def stripe_odd(cls, n_more, more_val, n_less, less_val):
        """
        Prepare striping for an odd number of rows.
        :param n_more:
            The number of rows with the value that there's more of
        :param more_val:
            The value that there's more of
        :param n_less:
            The number of rows that there's less of
        :param less_val:
            The value that there's less of
        :return:
            Returns a :class:`tuple` of striped values with appropriate buffer.
        """
        (n_m, m_v) = n_more, more_val
        (n_l, l_v) = n_less, less_val

        # Calculate how much "buffer" we need.
        # Example (b = buffer number, o = outer stripe, i = inner stripe)
        #    4, 4, 5, 4, 4 -> b, o, i, o, b  (buffer = 1)
        #    4, 5, 4, 5, 4 -> o, i, o, i, o  (buffer = 0)
        n_inner_stripes = n_l
        n_buffer = (n_m + n_l) - (2 * n_inner_stripes + 1)
        assert n_buffer % 2 == 0, (n_more, n_less, n_buffer)
        n_buffer //= 2

        buff_tuple = (m_v,) * n_buffer
        stripe_tuple = (m_v, l_v) * n_inner_stripes + (m_v,)

        return buff_tuple + stripe_tuple + buff_tuple

    @classmethod
    def stripe_even(cls, n_more, more_val, n_less, less_val):
        """
        Prepare striping for an even number of rows.
        :param n_more:
            The number of rows with the value that there's more of
        :param more_val:
            The value that there's more of
        :param n_less:
            The number of rows that there's less of
        :param less_val:
            The value that there's less of
        :return:
            Returns a :class:`tuple` of striped values with appropriate buffer.
        """
        total = n_more + n_less
        if total % 2:
            msg = "Expected an even number of values, got {} + {}".format(
                n_more, n_less
            )
            raise ValueError(msg)

        assert n_more >= n_less, (n_more, n_less)

        # See what the minimum unit cell is
        n_l_c, n_m_c = n_less, n_more
        num_div = 0
        while True:
            n_l_c, lr = divmod(n_l_c, 2)
            n_m_c, mr = divmod(n_m_c, 2)
            if lr or mr:
                break

            num_div += 1

        # Maximum number of times we can half this to get a "unit cell"
        n_cells = 2 ** num_div

        # Make the largest possible odd unit cell
        cell_s = total // n_cells  # Size of a unit cell

        cell_buff = int(cell_s % 2 == 0)  # Buffer is either 1 or 0
        cell_s -= cell_buff
        cell_nl = n_less // n_cells
        cell_nm = cell_s - cell_nl

        if cell_nm == 0:
            stripe_cell = (less_val,)
        else:
            stripe_cell = cls.stripe_odd(cell_nm, more_val, cell_nl, less_val)

        unit_cell = (more_val,) * cell_buff + stripe_cell

        if num_div == 0:
            return unit_cell

        stripe_out = unit_cell * (n_cells // 2)
        return tuple(reversed(stripe_out)) + stripe_out
