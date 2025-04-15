#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 19:59:07 2021

@author: Alexander Southan
"""

import numpy as np
import unittest

from src.pyColorTools import color_tools, sRGB


class TestColorTools(unittest.TestCase):

    def test_colortools(self):
        
        color_names = ['R', 'G', 'B', 'white', 'color1', 'color2', 'color3']
        color_values = sRGB.primaries.xy.values.tolist() + [[0.4, 0.3], [0.2, 0.1], [0.35, 0.5]]
        # test initialization with xy input with color names
        test_set = color_tools.colorset(
            color_values, init_format='xy', color_names=color_names)
        # Convert xy input to RGB and back to xy
        test_set.xy_to_RGB(sRGB, scale_Y=True, norm='individual')
        test_set.RGB_to_xy(sRGB)
        # Test initialization with RGB values without color names
        test_set_rgb = color_tools.colorset(
            test_set.get_color_values('RGB').values, init_format='RGB')
        # Convert RGB input back to xy
        test_set_rgb.RGB_to_xy(sRGB)

        # Test ValueErrors with incorrect inputs
        with self.assertRaises(ValueError):
            color_tools.colorset(color_values, init_format='xy',
                                 color_names=color_names[:-1])
        with self.assertRaises(ValueError):
            color_tools.colorset([color_values], init_format='xy',
                                 color_names=color_names)
        

        
        # Test if the color value query function works as expected
        self.assertTrue(test_set.RGB[['R', 'G', 'B']].equals(
            test_set.get_color_values('RGB')))
        self.assertTrue(test_set.XYZ[['X', 'Y', 'Z']].equals(
            test_set.get_color_values('XYZ')))
        self.assertTrue(test_set.xy[['x', 'y']].equals(
            test_set.get_color_values('xy')))
        self.assertTrue(test_set.spectrum.equals(
            test_set.get_color_values('spectrum')))
        with self.assertRaises(ValueError):
            test_set.get_color_values('sRGB')
        
        # Check if conversion of primaries to RGB worked
        self.assertAlmostEqual(test_set.RGB.at['R', 'R'], 1, 5)
        self.assertAlmostEqual(test_set.RGB.at['R', 'G'], 0, 5)
        self.assertAlmostEqual(test_set.RGB.at['R', 'B'], 0, 5)
        
        self.assertAlmostEqual(test_set.RGB.at['G', 'R'], 0, 5)
        self.assertAlmostEqual(test_set.RGB.at['G', 'G'], 1, 5)
        self.assertAlmostEqual(test_set.RGB.at['G', 'B'], 0, 5)
        
        self.assertAlmostEqual(test_set.RGB.at['B', 'R'], 0, 5)
        self.assertAlmostEqual(test_set.RGB.at['B', 'G'], 0, 5)
        self.assertAlmostEqual(test_set.RGB.at['B', 'B'], 1, 5)
        
        # Test if individual normalization works as expected
        self.assertTrue((test_set.get_color_values('RGB').max(axis=1)==1).all())

        # Test if global normalization works as expected
        test_set.xy_to_RGB(sRGB, scale_Y=True, norm='global')
        self.assertTrue((test_set.get_color_values('RGB').max(axis=1)==1).any())
        
        
        # Check if conversion from RGB back to xy gives correct result
        for curr_name, curr_vals in zip(color_names, color_values):
            self.assertAlmostEqual(
                test_set.xy.at[curr_name, 'x'], curr_vals[0], 5)
            self.assertAlmostEqual(
                test_set.xy.at[curr_name, 'y'], curr_vals[1], 5)
        

        # test if black body with 6504 K is perceived as white
        wavelengths = sRGB.cmf.index.values
        black_body_6504 = 120*10**15/(wavelengths**5*np.exp(
            (6.62607015*10**-34 * 299792458)/(wavelengths*10**-9*6504*1.380649*10**-23)
            - 1))
        black_body_testset = color_tools.colorset(black_body_6504, init_format='spectrum')
        black_body_testset.spectrum_to_color(sRGB)
        self.assertTrue((black_body_testset.RGB[['R', 'G', 'B']] > 0.9).all().all())
        
