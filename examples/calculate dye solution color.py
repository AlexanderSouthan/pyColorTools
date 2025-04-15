# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 10:09:24 2025

@author: southan
"""

import numpy as np
import pandas as pd
from pyColorTools import colorset, sRGB

import matplotlib.pyplot as plt

# Import the spectra
azobenzene_pristine = pd.read_csv(
    'azobenzene derivative in toluene pristine.csv', sep=', ', decimal=',',
    index_col=0, names=['Abs_pristine'], skiprows=1)
azobenzene_irradiated = pd.read_csv(
    'azobenzene derivative in toluene irradiated.csv', sep=', ', decimal=',',
    index_col=0, names=['Abs_irradiated'], skiprows=1)
spectra = pd.concat([azobenzene_pristine, azobenzene_irradiated], axis=1)

sample_names = ['pristine', 'irradiated']

# Get the wavelengths present in the color matching function imported in the
# sRGB colorspace.
wavelengths = sRGB.cmf.index.values

# Fill missing values in spectra so that wavelengths conform to cmf in sRGB.
added_wl = np.linspace(830, 605, 46)
new_index = np.concat([added_wl, spectra.index.values])
spectra = spectra.reindex(index=new_index)
spectra = spectra.loc[wavelengths]
for curr_name in sample_names:
    # Scale the absorption to a higher value
    spectra['Abs_{}'.format(curr_name)] *= 10
    # Add the extra wavelength points
    spectra.loc[added_wl, 'Abs_{}'.format(curr_name)] = spectra.at[600, 'Abs_{}'.format(curr_name)]
    # Calculate the fraction of transmitted light from absorbance
    spectra['T_{}'.format(curr_name)] = 10**(-spectra['Abs_{}'.format(curr_name)])

# Daylight spectrum of standard illuminant D65 according to doi
# 10.25039/CIE.DS.hjfjmt59
int_d65 = pd.read_csv(
    'CIE_std_illum_D65.csv', delimiter=',', header=None, names=['intensity'],
    index_col=0)
spectra['int_d65'] = int_d65.loc[wavelengths]

# Calculate transmitted spectrum of the dye solutions
trans_names = ['trans_spec_{}'.format(curr_name) for curr_name in sample_names]
for trans_name, curr_name in zip(trans_names, sample_names):
    spectra[trans_name] = spectra['int_d65'] * spectra['T_{}'.format(curr_name)]

# Calculate the RGB color corresponding to the sample spectra
trans_colorset = colorset(
    spectra[['int_d65'] + trans_names].values.T, init_format='spectrum',
    color_names=['D65'] + sample_names)
trans_colorset.spectrum_to_color(sRGB, norm='individual')

# Plot the colors as circles
fig1, ax1 = plt.subplots(dpi=600)
ax1.set_aspect('equal')
ax1.set_xlim([-0.5, 1.5])
ax1.set_ylim([-0.5, 0.5])
ax1.set_xticks([])
ax1.set_yticks([])
circles = {}
for idx, curr_name in enumerate(sample_names):
    circles[curr_name] = plt.Circle(
        (idx, 0), 0.5,
        color=trans_colorset.get_color_values('RGB').loc[curr_name].values)
    ax1.add_patch(circles[curr_name])
    ax1.annotate('Color of {} azobenzene\nderivative in toluene'.format(curr_name),
                 xy=(idx, 0), fontsize=8, ha='center', va='center')

fig1.savefig('Dye solution color from spectrum.png')