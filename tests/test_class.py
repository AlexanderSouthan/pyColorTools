#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 19:59:07 2021

@author: Alexander Southan
"""

import numpy as np
import unittest

from src.pyColorTools import color_tools


class TestColorTools(unittest.TestCase):

    def test_colortools(self):

        test_set = color_tools.colorset([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]], init_format='xy')
