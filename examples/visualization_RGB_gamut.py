# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 13:58:19 2025

@author: southan
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from pyColorTools import colorset, sRGB

# Construct spectra that contain only one wavelength
spectra = np.zeros([len(sRGB.cmf.index.values)]*2)
np.fill_diagonal(spectra, 1)

# Calculate the perceived color in xy values for the monochrome spectra
monochrom = colorset(spectra, init_format='spectrum')
monochrom.spectrum_to_color(sRGB)
monochrom.XYZ_to_xy()

# number of data points in each dimension used for the RGB triangle
data_grid = 1000
rgb_triangle = np.tile(np.linspace(0, 1, data_grid), data_grid).reshape(-1, data_grid)

# Make a square of coordinates for the RGB gamut
x_coords, y_coords = np.meshgrid(
    np.linspace(sRGB.primaries.xy['x'].min(), sRGB.primaries.xy['x'].max(), data_grid),
    np.linspace(sRGB.primaries.xy['y'].min(), sRGB.primaries.xy['y'].max(), data_grid))
# Cut away the top left section from square
allowed_y = (0.6-0.06)/(0.3-0.15)*(x_coords-0.3)+0.6
top_left_mask = y_coords > allowed_y
rgb_triangle[top_left_mask] = np.nan
x_coords[top_left_mask] = np.nan
y_coords[top_left_mask] = np.nan
# Cut away the top right section from square
allowed_y = (0.6-0.33)/(0.3-0.64)*(x_coords-0.3)+0.6
top_right_mask = y_coords > allowed_y
rgb_triangle[top_right_mask] = np.nan
x_coords[top_right_mask] = np.nan
y_coords[top_right_mask] = np.nan
# Cut away the bottom right section from square
allowed_y = (0.33-0.06)/(0.64-0.15)*(x_coords-0.64)+0.33
bottom_right_mask = y_coords < allowed_y
rgb_triangle[bottom_right_mask] = np.nan
x_coords[bottom_right_mask] = np.nan
y_coords[bottom_right_mask] = np.nan


# Extract the allowed xy values from the coordinates and convert them to RGB
xy_values = np.vstack([x_coords[~np.isnan(x_coords)], y_coords[~np.isnan(y_coords)]]).T
rgb_colorset = colorset(xy_values, init_format='xy')
rgb_colorset.xy_to_RGB(sRGB, norm='individual')


# Make the colormap and assign ascending numbers to the rgb_triangle
color_values = rgb_colorset.RGB[['R', 'G', 'B']].values
cmap = ListedColormap(color_values)
rgb_triangle[~np.isnan(rgb_triangle)] = np.arange(len(xy_values))

# Make the plot
fig1, ax1 = plt.subplots(dpi=600)

ax1.imshow(
    rgb_triangle, origin='lower',
    extent=(sRGB.primaries.xy['x'].min(), sRGB.primaries.xy['x'].max(),
            sRGB.primaries.xy['y'].min(), sRGB.primaries.xy['y'].max()), cmap=cmap)

ax1.plot(monochrom.xy['x'], monochrom.xy['y'],
         np.roll(monochrom.xy['x'], 1), np.roll(monochrom.xy['y'], 1),
         ls='-', lw=1, marker='o', c='grey', ms=2)
ax1.plot(sRGB.primaries.xy.loc[['R', 'G', 'B'], 'x'],
         sRGB.primaries.xy.loc[['R', 'G', 'B'], 'y'],
         np.roll(sRGB.primaries.xy.loc[['R', 'G', 'B'], 'x'], 1),
         np.roll(sRGB.primaries.xy.loc[['R', 'G', 'B'], 'y'], 1), c='k', lw=1)

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_xlim([0, 0.8])
ax1.set_ylim([0, 0.9])

fig1.savefig('rgb_gamut.png')