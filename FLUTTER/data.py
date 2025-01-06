# -*- coding: utf-8 -*-
"""

Pierre DOERFLER, January 2025

"""
import numpy as np

class Data():
    def __init__(self):
        """
        Initial parameters for Flutter()
        """
        self.params = dict(rho = 1,  # kg/m³
                           rho_s = 600,  # kg/m³
                           CL_alpha = 2 * np.pi,
                           CmF = 0.01,
                           alpha0 = np.radians(5),
                           alphaL0 = np.radians(-3),
                           k_alpha = 9e4, # N.m/rad
                           k_z = 3e4, # N/m
                           U = 120, # m/s
                           J0 = 205,  # kg.m²
                           a = 0.5,  # m
                           m = 600,  # kg
                           d = -0.2,  # m
                           c = 2,  # m
                           delta_b = 1)  # m
