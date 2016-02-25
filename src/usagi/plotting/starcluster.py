import numpy as np

from amuse.units import units

from parameters import *

try:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot
    from matplotlib.colors import LogNorm
    import matplotlib.cm as cm

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def gas_plot(
        gas,
        fig,
        ax,
        p,
        ):
    if not fig:
        fig = new_figure(p)
    if not ax:
        ax  = new_axes(fig, p)

    # Surface of a bin
    binsq   = (
            (p.plot_maxx - p.plot_minx) / p.plot_bins * 
            (p.plot_maxy - p.plot_miny) / p.plot_bins
            )

    counts, xedges, yedges  = np.histogram2d( 
            ( # Default: y
                gas.x.value_in(units.parsec) if (
                    p.plot_axes_y == "x"
                    ) else
                gas.z.value_in(units.parsec) if (
                    p.plot_axes_y == "z"
                    ) else
                gas.y.value_in(units.parsec)
                ),
            ( # Default: x
                gas.y.value_in(units.parsec) if (
                    p.plot_axes_x == "y"
                    ) else 
                gas.z.value_in(units.parsec) if ( 
                    p.plot_axes_x == "z"
                    ) else
                gas.x.value_in(units.parsec)
                ),
            bins    = p.plot_bins,
            range   = [
                [
                    p.plot_minx.value_in(units.parsec),
                    p.plot_maxx.value_in(units.parsec),
                    ],
                [
                    p.plot_miny.value_in(units.parsec),
                    p.plot_maxy.value_in(units.parsec),
                    ],
                ],
            weights = (
                gas.mass.value_in(units.MSun) /
                binsq.value_in(units.parsec**2)
                ),
            )
    
    logcounts   = np.log10(counts)
    
    gasplot     = ax.imshow(
            logcounts,
            origin          = "lower",
            #interpolation   = "none",
            interpolation   = "bicubic",
            extent          = [
                p.plot_minx.value_in(units.parsec),
                p.plot_maxx.value_in(units.parsec),
                p.plot_miny.value_in(units.parsec),
                p.plot_maxy.value_in(units.parsec),
                ],
            cmap            = p.plot_colormap,
            vmin            = p.plot_vmin,
            vmax            = p.plot_vmax,
            )
    
    pyplot.colorbar(
            gasplot,
            )
    return gasplot
    

def stars_plot(
        stars,
        fig,
        ax,
        p,
        ):
    if not fig:
        fig = new_figure(p)
    if not ax:
        ax  = new_axes(fig, p)

    starplot    = ax.scatter(
            (
                stars.z.value_in(units.parsec) if (
                    p.plot_axes_x == "z"
                    ) else 
                stars.y.value_in(units.parsec) if (
                    p.plot_axes_x == "y"
                    ) else 
                stars.x.value_in(units.parsec)
                ),
            (
                stars.x.value_in(units.parsec) if (
                    p.plot_axes_y == "x"
                    ) else 
                stars.z.value_in(units.parsec) if (
                    p.plot_axes_y == "z"
                    ) else 
                stars.y.value_in(units.parsec)
                ),
            marker  = 'o',
            s       = 0.25 * stars.mass.value_in(units.MSun)**(3./2.), # FIXME "realistic" size/colours?
            color   = 'w',
            lw      = 0,
            )
    return starplot


def new_figure(p):
    fig     = pyplot.figure(
            figsize = p.plot_figsize,
            dpi     = p.plot_dpi,
            )
    return fig

def new_axes(fig, p):
    ax      = fig.add_subplot(
            1,
            1,
            1,
            aspect  = 'equal',
            )
    ax.axis(
            [
                p.plot_minx.value_in(units.parsec), 
                p.plot_maxx.value_in(units.parsec), 
                p.plot_miny.value_in(units.parsec), 
                p.plot_maxy.value_in(units.parsec),
                ],
            )

    ax.set_axis_bgcolor(
            p.plot_bgcolor,
            )

    ax.set_autoscale_on( False )

    ax.set_title(
            p.plot_title,
            color   = "black",
            )

    if p.plot_axes_x == "y":
        ax.set_xlabel("Y [%s]"%units.parsec)
    elif p.plot_axes_x == "z":
        ax.set_xlabel("Z [%s]"%units.parsec)
    else:
        ax.set_xlabel("X [%s]"%units.parsec)

    if p.plot_axes_y == "x":
        ax.set_ylabel("X [%s]"%units.parsec)
    elif p.plot_axes_y == "z":
        ax.set_ylabel("Z [%s]"%units.parsec)
    else:
        ax.set_ylabel("Y [%s]"%units.parsec)

    return ax


def gas_stars_plot(
        i,
        time,
        gas,
        stars,
        p,
        plot_type   = "all",
        ):
    if not HAS_MATPLOTLIB:
        return -1

    if plot_type    == "all":
        plot_gas    = True
        plot_stars  = True
    elif plot_type  == "stars":
        plot_gas    = False
        plot_stars  = True
    elif plot_type  == "gas":
        plot_gas    = True
        plot_stars  = False
    else:
        plot_gas    = True
        plot_stars  = True
        
    # Total mass
    mtot    = gas.mass.sum()

    # Average mass per bin
    mavg    = mtot / (p.plot_bins**2)

    fig     = new_figure(p)

    if time:
        p.plot_title    = time.as_quantity_in(units.Myr)
    else:
        p.plot_title    = ""

    ax      = new_axes(fig, p)

    if plot_gas:
        gasplot = gas_plot(gas, fig, ax, p)

    if plot_stars:
        starplot    = stars_plot(stars, fig, ax, p)


    plot_name   = 'plot-%s-%s%s-%05i.png'%(
            plot_type,
            p.plot_axes_x,
            p.plot_axes_y,
            i,
            )

    pyplot.savefig(
            p.dir_plots + plot_name,
            dpi     = p.plot_dpi,
            )
    pyplot.close(fig)

    i   = i+1
    return i

if __name__ == "__main__":
    p   = Parameters() # FIXME: make argument_parser for plotting?

    stars   = read_set_from_file(
            p.stars_initial_file,
            'amuse',
            )

    gas     = read_set_from_file(
            p.gas_initial_file,
            'amuse',
            )

    i   = gas_stars_plot(
            0,
            False,
            gas,
            stars,
            p,
            plot_type="all"
            )

