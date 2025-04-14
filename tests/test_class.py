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
        
        
        test_set = color_tools.colorset(
            sRGB.primaries.xy.values.tolist(), init_format='xy',
            color_names=['R', 'G', 'B', 'white'])
        
        test_set.xy_to_RGB(sRGB, scale_Y=True, norm='individual')
        
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
        
        test_set.RGB_to_xy(sRGB)
        
        # Check if conversion from RGB back to xy gives correct result
        self.assertAlmostEqual(
            test_set.xy.at['R', 'x'], sRGB.primaries.xy.at['R', 'x'], 5)
        self.assertAlmostEqual(
            test_set.xy.at['R', 'y'], sRGB.primaries.xy.at['R', 'y'], 5)
        
        self.assertAlmostEqual(
            test_set.xy.at['G', 'x'], sRGB.primaries.xy.at['G', 'x'], 5)
        self.assertAlmostEqual(
            test_set.xy.at['G', 'y'], sRGB.primaries.xy.at['G', 'y'], 5)
        
        self.assertAlmostEqual(
            test_set.xy.at['B', 'x'], sRGB.primaries.xy.at['B', 'x'], 5)
        self.assertAlmostEqual(
            test_set.xy.at['B', 'y'], sRGB.primaries.xy.at['B', 'y'], 5)

        # test if black body with 6504 K is perceived as white
        wavelengths = sRGB.cmf.index.values
        black_body_6504 = 120*10**15/(wavelengths**5*np.exp(
            (6.62607015*10**-34 * 299792458)/(wavelengths*10**-9*6504*1.380649*10**-23)
            - 1))
        black_body_testset = color_tools.colorset(black_body_6504, init_format='spectrum')
        black_body_testset.spectrum_to_color(sRGB)
        self.assertTrue((black_body_testset.RGB[['R', 'G', 'B']] > 0.9).all().all())
        
