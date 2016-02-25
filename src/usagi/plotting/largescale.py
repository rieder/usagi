import os,sys

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Agg')

from amuse.units import constants
from amuse.units import units

from usagi.plotting.colormaps import cm_plusmin

import healpy as hp

def add_spherical_coordinates(
        particles,
        ):
    particles.distance  = particles.position.lengths()
    particles.theta     = np.arccos(
            particles.z / 
            particles.distance
            )
    particles.phi       = np.arctan2(
            particles.y.value_in(unit_distance),
            particles.x.value_in(unit_distance),
            )
    return

def make_sky_hitmap(
        particles,
        NSIDE           = 512,
        calc_rthetaphi  = True,
        weighting       = "mass",
        hubble          = False,
        diff_with       = False,
        ):

    unit_distance   = particles.position.unit
    unit_mass       = particles.mass.unit
    unit_vel        = particles.vrad.unit

    distanceweight  = particles.position.lengths()
    massweight      = particles.mass
    if hubble:
        velweight   = particles.vrad + particles.distance * hubble
    else:
        velweight   = particles.vrad


    if diff_with:
        diff_particles      = diff_with
        diff_distanceweight = diff_particles.position.lengths()
        diff_massweight     = diff_particles.mass

    if calc_rthetaphi:
        ## make sure the particles have spherical coordinates
        add_spherical_coordinates(particles)
        if diff_with:
            add_spherical_coordinates(diff_particles)

    skydens = hp.ang2pix(
            NSIDE, 
            particles.theta, 
            particles.phi,
            )

    if diff_with:
        diff_skydens    = hp.ang2pix(
                NSIDE, 
                diff_particles.theta, 
                diff_particles.phi,
                )

    if weighting == "mass":
        mass_hitmap                 = np.zeros(hp.nside2npix(NSIDE))
        pixels_massbin              = np.bincount(
                skydens, 
                weights = massweight.value_in(unit_mass),
                )
        mass_hitmap[:len(pixels_massbin)]   = (pixels_massbin) 
        mass_hitmap = np.nan_to_num(mass_hitmap)
        mass_hitmap = np.array(mass_hitmap) | unit_mass
        return mass_hitmap
    elif weighting == "massdiff":
        mass_hitmap                 = np.zeros(hp.nside2npix(NSIDE))
        diff_mass_hitmap            = np.zeros(hp.nside2npix(NSIDE))
        pixels_massbin              = np.bincount(
                skydens, 
                weights = massweight.value_in(unit_mass),
                )
        diff_pixels_massbin         = np.bincount(
                diff_skydens, 
                weights = diff_massweight.value_in(unit_mass),
                )
        mass_hitmap[:len(pixels_massbin)]       = (pixels_massbin) 
        diff_mass_hitmap[:len(diff_pixels_massbin)] = (diff_pixels_massbin) 
        mass_hitmap         = np.nan_to_num(mass_hitmap)
        diff_mass_hitmap    = np.nan_to_num(diff_mass_hitmap)
        deltam_hitmap       = mass_hitmap - diff_mass_hitmap
        deltam_hitmap       = np.array(deltam_hitmap) | unit_mass
        return (np.array(diff_mass_hitmap)|unit_mass)#deltam_hitmap
    elif weighting == "vrad":
        vel_hitmap                  = np.zeros(hp.nside2npix(NSIDE))
        pixels_numbin               = np.bincount(
                skydens, 
                )
        pixels_velbin               = np.bincount(
                skydens, 
                weights = velweight.value_in(unit_vel),
                )
        pixels_velavg       = pixels_velbin/pixels_numbin
        vel_hitmap[:len(pixels_velavg)]     = (pixels_velavg) 
        vel_hitmap  = np.nan_to_num(vel_hitmap)
        vel_hitmap  = np.array(vel_hitmap) | unit_vel
        return vel_hitmap
    elif weighting == "massvel":
        massvel_hitmap              = np.zeros(hp.nside2npix(NSIDE))
        #massvel_hitmap[:len(pixels_velavg)]     = (pixels_massvelbin) 
        #massvel_hitmap  = np.nan_to_num(massvel_hitmap)
        #massvel_hitmap  = np.array(vel_hitmap) | unit_vel
        return massvel_hitmap
    elif weighting == "gravity":
        grav_hitmap                 = np.zeros(hp.nside2npix(NSIDE))
        pixels_gravbin              = np.bincount(
                skydens, 
                weights = massweight.value_in(unit_mass) / \
                        (distanceweight.value_in(unit_distance))**2,
                )
        grav_hitmap[:len(pixels_gravbin)]   = (pixels_gravbin) 
        grav_hitmap = np.nan_to_num(grav_hitmap)
        grav_hitmap = np.array(grav_hitmap) | unit_mass / unit_distance**2
        grav_hitmap *= constants.G
        return grav_hitmap
    elif weighting == "tidal":
        grav_hitmap                 = np.zeros(hp.nside2npix(NSIDE))
        pixels_gravbin              = np.bincount(
                skydens, 
                weights = massweight.value_in(unit_mass) / \
                        (distanceweight.value_in(unit_distance))**3,
                )
        grav_hitmap[:len(pixels_gravbin)]   = (pixels_gravbin) 
        grav_hitmap = np.nan_to_num(grav_hitmap)
        grav_hitmap = np.array(grav_hitmap) | unit_mass / unit_distance**3
        grav_hitmap *= constants.G
        return grav_hitmap

#def __make_sky_plot__(
def projected_density_plot(
        hitmap,
        figname     = "skyplot.png",
        title       = "", 
        mincount    = 35, 
        maxcount    = 40, 
        NSIDE       = 512, 
        dpi         = 150, 
        xsize       = 2000, 
        inertia     = [], 
        walldir     = [], 
        filamentdir = [], 
        cmap        = cm_plusmin(under="w"),
        cbar        = True, 
        grid        = False,
        scale       = "log",
        ):

    if scale == "log":
        displayhitmap   = np.log10(hitmap) 
    else:
        displayhitmap   = hitmap

    if mincount == "auto":
        mincount = min(displayhitmap)
    if maxcount == "auto":
        maxcount = max(displayhitmap)

    hp.mollview(
            displayhitmap, 
            xsize   = xsize, 
            norm    = 'lin', 
            min     = mincount, 
            max     = maxcount,
            title   = title, 
            cbar    = cbar, 
            cmap    = cmap,
            )

    if grid: hp.graticule()

    if len(filamentdir)!=0:
        filament = hp.vec2ang(filamentdir)
        anti_filament = hp.vec2ang(-filamentdir)
        hp.projscatter(filament, color='#ffff99')
        hp.projscatter(anti_filament, color='#ffff99')

    if len(walldir)!=0:
        wall = hp.vec2ang(walldir)
        anti_wall = hp.vec2ang(-walldir)
        hp.projscatter(wall, color='#ff99ff')
        hp.projscatter(anti_wall, color='#ff99ff')

    if len(inertia)!=0:
        inertia_axis_a = hp.vec2ang(inertia[0])
        anti_inertia_axis_a = hp.vec2ang(-inertia[0])
        inertia_axis_b = hp.vec2ang(inertia[1])
        anti_inertia_axis_b = hp.vec2ang(-inertia[1])
        inertia_axis_c = hp.vec2ang(inertia[2])
        anti_inertia_axis_c = hp.vec2ang(-inertia[2])
        hp.projscatter(inertia_axis_a, color='r')
        hp.projscatter(anti_inertia_axis_a, color='r')
        hp.projscatter(inertia_axis_b, color='y')
        hp.projscatter(anti_inertia_axis_b, color='y')
        hp.projscatter(inertia_axis_c, color='g')
        hp.projscatter(anti_inertia_axis_c, color='g')

    print "#saving fig as %s"%figname
    plt.savefig(
            figname, 
            dpi = dpi,
            #bbox_inches='tight',
            )
    plt.clf()
    return

def projected_plot(
        hitmap,
        figname             = "skyplot.png",
        fwhm                = False,
        sigma               = False,
        remove_monopole     = False,
        remove_dipole       = False,
        remove_quadrupole   = False,
        mincount            = -100,
        maxcount            = 100,
        xsize               = 4000,
        dpi                 = 70,
        scale               = "lin",
        semilog_sw          = 1e3,
        lmax                = False,
        NSIDE               = 128,
        smooth              = True,
        ):

    fwhm    = 3 * hp.nside2resol(NSIDE)  # sigma overrides fwhm
    sigma   = 4 * (np.pi/180.)        # in deg
    #sigma = 1/60.                   # in rad

    if smooth:
        hitmap_smoothed = hp.smoothing(
                hitmap,
                fwhm    = fwhm,
                )
    else:
        hitmap_smoothed = hitmap

    if lmax:
        alm = hp.map2alm(
                hitmap,
                lmax    = lmax,
                )
    else:
        alm = hp.map2alm(
                hitmap_smoothed,
                )
    
    monopole_amp    = np.abs(alm[0])
    dipole_amp      = np.abs(np.sum(alm[1:3]))/2
    quadpole_amp    = np.abs(np.sum(alm[3:6]))/3
    print "#Monopole amp: %s; Dipole amp: %s; Quadrupole amp: %s"%(
            monopole_amp, 
            dipole_amp, 
            quadpole_amp,
            )
    #almsmoothed[0]      -= almsmoothed[0]      # Removes monopole
    #almsmoothed[1:3]    -= almsmoothed[1:3]    # Removes dipole
    #almsmoothed[3:6]    -= almsmoothed[3:6]    # Removes quadrupole
    
    hm2 = hp.alm2map(alm, NSIDE)

    if scale == "log":
        hm2 = np.where(np.isfinite(hm2),hm2,np.zeros_like(hm2)+1e-99)
        hm2 = np.where((hm2 > 0),hm2,np.zeros_like(hm2)+1e-99)

    if scale == "semilog":
        hm2 = np.where(np.isfinite(hm2),hm2,np.zeros_like(hm2))
        hm2log = np.log10(hm2)
        hm2minlog = np.log10(-hm2)
        hm2temp = np.log10(semilog_sw)/semilog_sw*hm2
        hm2temp = np.where((hm2 > semilog_sw),hm2log,hm2temp)
        hm2temp = np.where((hm2 < -semilog_sw),-hm2minlog,hm2temp)
        hm2 = hm2temp
    #hm2_masked = hp.ma(hm2)

    projected_density_plot(
            hm2,
            figname     = figname,
            NSIDE       = NSIDE,
            mincount    = mincount,
            maxcount    = maxcount,
            xsize       = xsize,
            dpi         = dpi,
            scale       = scale,
            #inertia     = np.array([1,0,0,0,1,0,0,0,1]).reshape(3,3),
            )
    plt.close()
    return

