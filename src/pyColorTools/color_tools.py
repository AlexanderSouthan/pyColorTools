# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 10:34:48 2025

@author: southan
"""

import numpy as np
import pandas as pd

from .color_matching_functions import cmf_cie_1931

class colorset:
    
    def __init__(self, values, init_format='xy', color_names=None):
        """
        Initialize a colorset instance.
        
        A colorset is a collection of color information of various colors in
        the formats RGB, xy, XYZ and as a spectrum. The color information is
        found in the DataFrames self.RGB, self.xy, self.XYZ and self.spectrum.

        Parameters
        ----------
        values : ndarray
            A 2D array containing the information defining the colors. The kind
            of data and the shape of the array depend on the value of
            init_format.
        init_format : string, optional
            The information defining the kind of color information given. 
            Allowed values are 'xy', 'XYZ', 'RGB' or 'spectrum'. For n colors,
            'xy' requires an array with shape (n, 2) as values, 'XYZ' and 'RGB'
            require a shape of (n,3), and 'spectrum' requires a shape of (n, m)
            containing spectra of the colors with m data points. The number m
            must be equal to a color matching function given in a colorspace
            used for calulating a color for example in RGB coordinates, and
            should use the same wavelength axis. The wavelength values do not
            have to be given here because they are already contained in the
            colorspace. The default for init_format is 'xy'.
        color_names : None or a list, optional
            Contains the names of the colors used as the index of the
            DataFrames containing the color data. The default is None, rsulting
            in a numbered index.

        Returns
        -------
        None.

        """
        values = np.asarray(values)
        if values.ndim > 2:
            raise ValueError('values should be a 2D array, can be 1D for a '
                             'single color.')
        elif values.ndim == 1:
            values = np.expand_dims(values, 0)
        
        if color_names is None:
            color_index = np.arange(len(values))
        else:
            if len(values) == len(color_names):
                color_index = color_names
            else:
                raise ValueError(
                    'values must contain the same number of entries like '
                    'color_names, or color_names must be None.')
        
        self.xy = pd.DataFrame([], columns=['x', 'y'], index=color_index,
                               dtype=float)
        self.XYZ = pd.DataFrame([], columns=['X', 'Y', 'Z', 'scaled_Y'],
                                index=color_index, dtype=float)
        self.RGB = pd.DataFrame([], columns=['R', 'G', 'B', 'corrected'],
                                index=color_index, dtype=float)
        self.spectrum = pd.DataFrame(
            [], columns=np.arange(values.shape[1]), index=color_index,
            dtype=float)
        
        if init_format == 'xy':
            if values.shape[1] == 2:
                self.xy[['x', 'y']] = values
                self.xy_to_XYZ(scale_Y=True)
            else:
                raise ValueError(
                    'values should be a 2D array with a shape of (n, 2) for n '
                    'colors, but has a shape of {}.'.format(values.shape))
        elif init_format == 'spectrum':
            self.spectrum.loc[:] = values
        elif init_format == 'RGB':
            if values.shape[1] == 3:
                self.RGB[['R', 'G', 'B']] = values
            else:
                raise ValueError(
                    'values should be a 2D array with a shape of (n, 3) for n '
                    'colors, but has a shape of {}.'.format(values.shape))
        elif init_format == 'XYZ':
            if values.shape[1] == 3:
                self.XYZ[['X', 'Y', 'Z']] = values
                self.XYZ_to_xy()
            else:
                raise ValueError(
                    'values should be a 2D array with a shape of (n, 3) for n '
                    'colors, but has a shape of {}.'.format(values.shape))
        else:
            raise ValueError('No valid input format given.')

    def xy_to_XYZ(self, scale_Y=True):
        """
        Calculate the XYZ values from the xy values.
        
        Keep in mind that the xy values contain no information about the
        color brightness, so the brightness of the resulting XYZ array is
        arbitrary. 
    
        Parameters
        ----------
        scale_Y : boolean, optional
            Defines if in the output is scaled such that the output Y equals 1.
            Works only if y is not zero. The default is True.
    
        Returns
        -------
        DataFrame
            The XYZ values.
    
        """
        if scale_Y:
            scaling_factor = self.xy['y']
            scaling_factor[self.xy['y'] == 0] = 1
        else:
            scaling_factor = 1
        
        self.XYZ[['X', 'Y']] = self.xy
        self.XYZ['Z'] = 1 - self.xy.sum(axis=1)
        self.XYZ[['X', 'Y', 'Z']] = self.XYZ[['X', 'Y', 'Z']].div(
            scaling_factor, axis=0)
        return self.XYZ

    def XYZ_to_xy(self):
        """
        Convert the XYZ values to xy values.

        Returns
        -------
        DataFrame
            The xy values.

        """
        self.xy[['x', 'y']] = self.XYZ[['X', 'Y']].div(
            self.XYZ.sum(axis=1), axis=0)
        return self.xy

    def XYZ_to_RGB(self, color_space, norm='global'):
        """
        Calculate RGB color values from XYZ data.
        
        The calculation require the previous definition of the used colorspace
        and the corresponding conversion matrix. For this purpose, first a
        colorspace instance has to be created, containing the primaries and a
        white point.

        Parameters
        ----------
        color_space : colorspace
            A colorspace instance containing information about the conversion
            matrix.
        norm : string, optional
            Defines if the RGB output is scaled to the interval between 0 and
            1. This makes sense for example if there is no brightness
            information in the original data, such as xy color data. Allowed
            values are 'global', 'individual' or 'none'. The default is
            'global'.

        Returns
        -------
        DataFrame
            The RGB data.

        """
        self.RGB.loc[:, ['R', 'G', 'B']] = color_space.mat_XYZ_to_RGB.dot(
            self.XYZ[['X', 'Y', 'Z']].T).T
        
        if np.any(self.RGB < 0):
            # Some colors are outside the RGB gamut, as indicated by negative
            # RGB values. Approximation is done by setting negative values to
            # zero and scaling the other values so that the luminescence given
            # by Y ist maintained. Y is given by:
            #       Y = Sr*Yr*R + Sg*Yg*G + Sb*Yb*B
            # (Sr, Sg, Sb, Yr, Yg and Yb are defined by the color space.)
            # For example, by setting the R value to zero, i.e. by subtracting
            # Sr*Yr*R, one obtains:
            #       Y-Sr*Yr*R = Sg*Yg*G + Sb*Yb*B
            # The remaining G and B values therefore have to be multiplied by
            # the ratio of the original Y and Y-Sr*Yr*R to keep the Y constant,
            # and the corrected G and B are obtained:
            #       G_corr = G * Y/(Y-Sr*Yr*G)
            #       B_corr = B * Y/(Y-Sr*Yr*R)
            # For negative G and B, this is done accordingly. This procedure id
            # briefly mentioned in soapfilmcalc.pdf.
            neg_mask = self.RGB[['R', 'G', 'B']] < 0
            print('Warning: Correcting some colors because outside of RGB '
                  'gamut. See column \'corrected\' in self.RGB for details.')
            for curr_key in neg_mask:
                # Calculate the correction factor for either R, G or B, as
                # given by curr_key/the loop iterations
                corr_factor = (
                    self.XYZ.loc[neg_mask[curr_key], 'Y']/
                    (self.XYZ.loc[neg_mask[curr_key], 'Y'] -
                     color_space.mat_RGB_to_XYZ.at['Y', curr_key]*self.RGB.loc[
                         neg_mask[curr_key], curr_key]))
                # Identify the columns which have to be multiplied with the
                # correction factor
                other_keys = neg_mask.columns.tolist()
                other_keys.remove(curr_key)
                # Apply the correction factor to the other columns
                for other_key in other_keys:
                    self.RGB.loc[neg_mask[curr_key], other_key] *= corr_factor
                # Set the negative values in the current column to zero
                self.RGB.loc[neg_mask[curr_key], curr_key] = 0
            # Store the information about the color correction in respective
            # column.
            self.RGB.loc[neg_mask.sum(axis=1) > 0, 'corrected'] = 1

        if not np.all(self.RGB==0):
            if norm == 'global':
                # Normalize the RGB color values to the range between 0 and 1.
                self.RGB[['R', 'G', 'B']] /= self.RGB[['R', 'G', 'B']].values.max()
            elif norm == 'individual':
                self.RGB[['R', 'G', 'B']] = self.RGB[['R', 'G', 'B']].div(
                    self.RGB[['R', 'G', 'B']].max(axis=1), axis=0)
            elif norm == 'none':
                pass
            else:
                raise ValueError(
                    'No valid normalization keyword given. Allowed values are '
                    '\'global\', \'individual\' and \'none\', but \'{}\' was '
                    'given.'.format(norm))
    
        return self.RGB

    def RGB_to_XYZ(self, color_space):
        """
        Convert RGB to XYZ data.
        
        See docstring in self.XYZ_to_RGB for information on the color_space.

        Parameters
        ----------
        color_space : colorspace
            A colorspace instance containing information about the conversion
            matrix.

        Returns
        -------
        DataFrame
            The XYZ data.

        """
        self.XYZ.loc[:, ['X', 'Y', 'Z']] = color_space.mat_RGB_to_XYZ.dot(
            self.RGB[['R', 'G', 'B']].T).T
        
        return self.XYZ

    def xy_to_RGB(self, color_space, scale_Y=True, norm='global'):
        """
        Convert xy values to RGB.
        
        The calculation is performed via the XYZ values. Keep in mind that the
        xy values contain no brightness information, so the scaling is
        arbitrary. See docstring in self.XYZ_to_RGB for information on the
        color_space.

        Parameters
        ----------
        color_space : colorspace
            A colorspace instance containing information about the conversion
            matrix.
        scale_Y : boolean, optional
            Is passed to the call of self.xy_to_XYZ. The default is True.
        norm : string, optional
            Is passed to the call of self.XYZ_to_RGB. The default is 'global'.

        Returns
        -------
        DataFrame
            The RGB values.

        """
        self.xy_to_XYZ(scale_Y=scale_Y)
        self.XYZ_to_RGB(color_space, norm=norm)

        return self.RGB
        
    def RGB_to_xy(self, color_space):
        """
        Convert RGB to xy values.

        Keep in mind that the xy values contain no brightness information.
        See docstring in self.XYZ_to_RGB for information on the color_space.

        Parameters
        ----------
        color_space : colorspace
            A colorspace instance containing information about the conversion
            matrix.

        Returns
        -------
        DataFrame
            The xy values.

        """
        self.RGB_to_XYZ(color_space)
        self.XYZ_to_xy()

        return self.xy

    def spectrum_to_color(self, color_space, output='RGB', norm='global'):
        """
        Calculate the perceived color from an light spectrum.
        
        The calulations require a color matching function defined in the
        color_space. Therefore, first a colorspace instance has to be created,
        containing the corresponding color matching function. The calulations
        first yield the XYZ values matching the spectrum, which are
        subsequently converted to RGB values if requested.

        Parameters
        ----------
        color_space : colorspace
            A colorspace instance containing information about the color
            matching function.
        output : string, optional
            Defines which data is returned. Can be 'RGB' for RGB values or
            anything else for XYZ values. The default is 'RGB'.
        norm : string, optional
            Defines how the calulated RGB data are normalized. Is passed to the
            call of self.XYZ_to_RGB. The default is 'global'.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        wavelengths = color_space.cmf.index.values
        if self.spectrum.shape[1] != len(wavelengths):
            raise ValueError(
                'The number of data points in the provided spectra ({}) does '
                'not match the number of values of the color matching function'
                ' ({}).'.format(self.spectrum.shape[1], len(wavelengths)))
        self.spectrum.columns = wavelengths
        self.XYZ.iloc[:, 0:3] = np.sum(
            self.spectrum.values[:, :, np.newaxis] *
            color_space.cmf[['X','Y','Z']].values *
            color_space.cmf['lambda_spacing'].values[:, np.newaxis], axis=1)
        if output == 'RGB':
            self.RGB = self.XYZ_to_RGB(color_space, norm=norm)
            return self.RGB
        else:
            return self.XYZ

class colorspace:
    
    def __init__(self, primaries, cmf='CIE_1932'):
        """
        Initialize a colorspace instance.
        
        The colorspace contains the matrices for conversion of XYZ to RGB and
        vice verse. Additionally, it contains a color matching function. 

        Parameters
        ----------
        primaries : colorset
            A colorset containing information about the primaries and the white
            point. The color names must be ['R', 'G', 'B', 'white'].
        cmf : string, optional
            The color matching function (CMFs) used to calculate a perceived
            color from a light spectrum. The default is 'CIE_1932', meaning the
            CIE 1931 2-deg, XYZ CMFs, see http://cvrl.ioo.ucl.ac.uk/index.htm
            Other CMFs are not included currently, but could be included
            easily.

        Returns
        -------
        None.

        """
        # The color space primaries and the white point.
        self.primaries = primaries
        self.matrix_primaries = self.primaries.XYZ.loc[
            ['R', 'G', 'B'], ['X', 'Y', 'Z']].T
        
        # Import the color matching function for spectrum to color conversion
        import os
        print(os.getcwd())
        if cmf == 'CIE_1932':  # CIE 1931 2-deg, XYZ CMFs, 
            self.cmf = cmf_cie_1931
            self.cmf['lambda_spacing'] = np.diff(
                self.cmf.index, prepend=2*self.cmf.index[0]-self.cmf.index[1])
        else:
            self.cmf = None
        
        # Conversion of XYZ values to RGB values and vice versa via
        #       [X]   [Sr*Xr, Sg*Xg, Sb*Xb]   [R]
        #       [Y] = [Sr*Yr, Sg*Yg, Sb*Yb] * [G]
        #       [Z]   [Sr*Zr, Sg*Zg, Sb*Zb]   [B]
        # Sr, Sg and Sb are the relative strengths of the primaries.
        # [Xr, Yz, Zr], [Xg, Yg, Zg] and [Xb, Yb, Zb] are the XYZ values of the
        # primaries. So first [Sr, Sg, Sb] is calculated with the primaries,
        # knowing that the sum of the primaries must equal the white point:
        #       [Xw]   [Xr, Xg, Xb]   [Sr]
        #       [Yw] = [Yr, Yg, Yb] * [Sg]
        #       [Zw]   [Zr, Zg, Zb]   [Sb]
        self.s = np.linalg.solve(self.matrix_primaries, self.primaries.XYZ.loc[
            'white', ['X', 'Y', 'Z']])
        # Then the conversion matrix for RGB to XYZ conversion is constructed:
        self.mat_RGB_to_XYZ = self.s*self.matrix_primaries
        # Then the conversion matrix for XYZ to RGB conversion is calculated:
        self.mat_XYZ_to_RGB = np.linalg.inv(self.mat_RGB_to_XYZ)

# Define the sRGB color space with primaries and white point for D65 illuminant
primaries_sRGB = colorset(
    [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06], [0.3127, 0.3290]],
    color_names=['R', 'G', 'B', 'white'], init_format='xy')
sRGB = colorspace(primaries_sRGB)
