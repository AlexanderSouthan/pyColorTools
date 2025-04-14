# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 15:44:12 2025

@author: southan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from pyColorTools import colorset, sRGB
from visualization_RGB_gamut import fig1, ax1


# Optical parameters of layer and air
n_0 = 1 # refractive index of air
n_1 = 1.52 # refractive index of layer

# Get the wavelengths present in the color matching function imported in the
# sRGB colorspace.
wavelengths = sRGB.cmf.index.values

# Import and interpolate refractive index of substrate to wavelength scale in
# sRGB colorspace color matching function.
n_s = pd.read_csv('refractive index silicon.txt', skiprows=12, sep=' ',
                  header=0).dropna(axis=1)
n_s.iloc[:, 0] *= 1000 # convert to nm
n_s_interp = np.interp(wavelengths, n_s.iloc[:, 0], n_s.iloc[:, 1])
k_s_interp = np.interp(wavelengths, n_s.iloc[:, 0], n_s.iloc[:, 2])
# n_s_interp = np.full_like(wavelengths, 4)
# k_s_interp = np.zeros_like(wavelengths)

# Daylight spectrum of standard illuminant D65 according to doi
# 10.25039/CIE.DS.hjfjmt59 or as an alternative the black body radiation at
# 6504 K (similar to D65)
illuminants = {}  # dict to collect the emission spectra of different illuminants
int_d65 = pd.read_csv(
    'CIE_std_illum_D65.csv', delimiter=',', header=None, names=['intensity'],
    index_col=0)
illuminants['D65'] = int_d65.loc[wavelengths, 'intensity'].values
illuminants['black body with T=6504 K'] = 120*10**15/(wavelengths**5*np.exp(
    (6.62607015*10**-34 * 299792458)/(wavelengths*10**-9*6504*1.380649*10**-23)
    - 1))

# Define thickness range and number of data points for the calculations
min_thickness = 0  # in nm
max_thickness = 1500  # in nm
n_thickness = max_thickness - min_thickness + 1
thicknesses = np.linspace(min_thickness, max_thickness, n_thickness)

# Initialize the data containers for the data calculated below
r_i = np.ones((n_thickness, len(wavelengths))) # fraction of reflected light for a given number of layer thicknesses
reflected_spectra = {}  # dict to collect the reflected spectra for different illuminants
cmap_values = {}  # dict to collect the different color for the colormaps
color_objects = {} # dict to collect the film thickness colorsets for the different illuminants
cmaps = {}  # dict to collect the colormaps used for plotting
for curr_illuminant in illuminants:
    reflected_spectra[curr_illuminant] = np.ones(
        (n_thickness, len(wavelengths)))
    cmap_values[curr_illuminant] = np.ones((n_thickness, 4)) # colormap values used for plotting
    color_objects[curr_illuminant] = np.full_like(
        thicknesses, colorset, dtype=object)

# Fill r_i, reflected spectra and cmap_values
# according to "Experimental evaluation method of PSF functions used for 
# proximity effects", J. Vac. Sci. Technol. B, Vol. 29, No. 6, Nov/Dec 2011,
# https://doi.org/10.1116/1.3656343 or thinfilmv1.pdf (see in folder of this
# python file)
k_0_h = 2*np.pi*n_1*thicknesses[:, np.newaxis]/wavelengths
r_i = (((
        (1-n_s_interp)*n_1*np.cos(k_0_h) + k_s_interp*np.sin(k_0_h))**2 +
        (-n_1*k_s_interp*np.cos(k_0_h) + (n_1**2-n_s_interp)*np.sin(k_0_h))**2
        ) /
    (((1+n_s_interp)*n_1*np.cos(k_0_h) + k_s_interp*np.sin(k_0_h))**2 +
     (n_1*k_s_interp*np.cos(k_0_h) - (n_1**2+n_s_interp)*np.sin(k_0_h))**2))

for curr_illuminant in illuminants:
    reflected_spectra[curr_illuminant] = r_i * illuminants[curr_illuminant]
    color_objects[curr_illuminant] = colorset(
        reflected_spectra[curr_illuminant], init_format='spectrum')
    color_objects[curr_illuminant].spectrum_to_color(
        color_space=sRGB, norm=True)
    cmap_values[curr_illuminant][:, 0:3] =  color_objects[
        curr_illuminant].RGB.loc[:, ['R', 'G', 'B']]

figs = {}  # dict to collect the figure objects
axs = {}  # dict to collect the axis objects
for curr_illuminant in illuminants:
    # Correct RGB values for displaying according to soapfilmcalc.pdf or
    # http://www.color.org/sRGB.xalter
    cmap_values_small = cmap_values[curr_illuminant] <= 0.00304
    cmap_values_big = cmap_values[curr_illuminant] > 0.0034
    cmap_values[curr_illuminant][cmap_values_small] = cmap_values[
        curr_illuminant][cmap_values_small] * 12.92
    cmap_values[curr_illuminant][cmap_values_big] = (
        1.055*cmap_values[curr_illuminant][cmap_values_big]**(1/2.4) - 0.055)

    # Asign the colors to matplotlib colormaps
    cmaps[curr_illuminant] = ListedColormap(cmap_values[curr_illuminant])

    # Plot the colors of the thin films with a given illuminant
    figs[curr_illuminant], axs[curr_illuminant] = plt.subplots(dpi=600)
    axs[curr_illuminant].imshow(
        np.vstack((thicknesses, thicknesses)), cmap=cmaps[curr_illuminant],
        aspect='auto', extent=[min_thickness, max_thickness, 0, 1],
        interpolation='none')
    axs[curr_illuminant].set_xlabel('film thickness [nm]')
    axs[curr_illuminant].set_yticks([])
    axs[curr_illuminant].set_title(
        'Film color on Si at normal incidence\n({} = {}, illuminant: {}'
        ')'.format('$n_\mathrm{film}$', n_1, curr_illuminant))

    # Save the figures for a given illuminant.
    figs[curr_illuminant].savefig('film_color_on_Si_{}nm_{}.png'.format(
        max_thickness, curr_illuminant))

for curr_illuminant in illuminants:
    color_objects[curr_illuminant].XYZ_to_xy()
    ax1.plot(color_objects[curr_illuminant].xy['x'], color_objects[curr_illuminant].xy['y'], ls='-', c='k', lw=1)
fig1.savefig('thickness_path_through_RGB_gamut.png')