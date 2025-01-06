# -*- coding: utf-8 -*-
"""

Pierre DOERFLER, January 2025

"""
from FLUTTER.flutter import Flutter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



if __name__ == "__main__":
    D = Data()
    D.params["rho"] = 3
    F = Flutter(D.params)
    F.solve()
