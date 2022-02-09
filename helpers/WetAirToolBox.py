"""
-------------------------------------------------------------------------------
Name:        WetAirToolBox
Purpose:     Different calculations in conjunction with humid air

Author:      Christian Buchholz, Marcus Vogt

Created:     01.12.2021
Copyright:   Chair of Sustainable Manufacturing and Life Cycle Engineering, Institute of Machine Tools and Production Technology, Technische Universität Braunschweig, Langer Kamp 19b, 38106 Braunschweig, Germany
Licence:     MIT (see License)
-------------------------------------------------------------------------------
"""


import numpy as np

# coefficients of the Magnus-formula
a = 6.112
b = 17.62
c = 243.12

R_L = 287   # gas constant air
R_WD = 461.4 # gas constant water vapor
ratio_R = R_L/R_WD
pG = 1013 # air pressure [hPa]

def humidity_dewpoint2abs (T, T_TP):
    """
    This function calculates the absolute humidity from the dew point and the temperature
    :param T: float: temperature [°C]
    :param T_TP: float: dew point [°C]
    :return X: float: absolute humidity [kg water / kg air]
    """
    ps = a * np.exp((b * T) / (c + T))   # Magnus-formula
    phi = np.exp((((T_TP*b*c)/(c+T)) - ((c*b*T)/(c+T))) / (T_TP + c))
    pD = phi * ps
    X = ratio_R * (pD / (pG - pD))
    return X


def humidity_abs2dewpoint(X):
    """
    This function calculates the dew point from the absolute humidity.
    :param X: float: absolute humidity [kg water / kg air]
    :return T_TP: float: dew point [°C]
    """
    pD = X*pG/(ratio_R+X)
    T_TP = (c*np.log(pD/a)) / (b - np.log(pD/a))
    return T_TP

