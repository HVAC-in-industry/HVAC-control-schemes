"""
-------------------------------------------------------------------------------
Name:        V2
Purpose:     Simple P-control

Author:      Christian Buchholz, Marcus Vogt

Created:     01.12.2021
Copyright:   Chair of Sustainable Manufacturing and Life Cycle Engineering, Institute of Machine Tools and Production Technology, Technische Universität Braunschweig, Langer Kamp 19b, 38106 Braunschweig, Germany
Licence:     MIT (see License)
-------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import helpers.WetAirToolBox as WetAirToolBox

def low_pass_filter(X_room_act: float, X_filt_last: float):
    """
    This function is a low pass filter for data pretreatment to filter out the high frequency data
    :param X_room_act: float: current absolute humidity (mixing ratio kg_moisture/kg_dry-air)
    :param X_filt_last: float: last filtered value of absolute humidity (mixing ratio kg_moisture/kg_dry-air)
    :return float: filtered values of current absolute humidity (mixing ratio kg_moisture/kg_dry-air)
    """
    # time step in seconds
    t_step = 60
    # filtering time constant
    Tf = 4000
    X_filt_act = ((t_step/Tf)*X_room_act + X_filt_last)/(1+(t_step/Tf))
    return X_filt_act

def P_controller(actual_value: float, set_point: float, u_max: float=3.55, u_min: float=1.63, kp:float=-190000): #190000
    """
    This function implements a generic proportional controller with a manipulated variable limitation in Python
    :param actual_value: float: generic actual input value of control circuit
    :param set_point: float: set point for the manipulated value u
    :param u_max: float: max. value of manipulated value u
    :param u_min: float: min. value of manipulated value u
    :param kp: float: Slope of the proportional controller
    :return float: manipulated value u
    """
    e = (abs(set_point) - abs(actual_value))
    u = np.ones((1, 1)) * kp * e
    if u > u_max:
        u = np.ones((1, 1)) * u_max
    elif u < u_min:
        u = np.ones((1, 1)) * u_min
    return u

if __name__ == '__main__':

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        "data", "t_step300_kp-30000_Tf4000_u_massflow.xlsx")
    data = pd.read_excel(path, engine='openpyxl')
    # Input temperature of supply air into room
    T = 20
    # Dew point temperature (°C) inside room
    T_TP_set = -52

    u = np.zeros(len(data.X_room_filt))
    X = np.zeros(len(data.X_room_filt))
    # iterate through filtered absolute moisture (mixing ratio)
    for i in range(len(data.X_room_filt)):
        # first case needs to treated differently due to filtering iteration loop based on last value
        if i == 0:
            X[i] = low_pass_filter(data.X_room[i], data.X_room[i])
            u[i] = P_controller(data.X_room[i], WetAirToolBox.humidity_dewpoint2abs(T, T_TP_set))
        else:
            X[i] = low_pass_filter(data.X_room[i], X[i-1])
            u[i] = P_controller(X[i], WetAirToolBox.humidity_dewpoint2abs(T, T_TP_set))

    fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    ax[0].plot(u)
    ax[1].plot(data.X_room_filt, 'r')
    ax[1].plot(data.X_room, 'b')
    ax[1].plot(X, color='goldenrod', linestyle='--')
    ax[0].set_title("V2 (P-control): Control of volume flow depending on dew point")
    ax[1].set_xlabel("Time in [s]")
    ax[0].set_ylabel(r"Volume flow in [$m^3/s$]")
    ax[1].set_ylabel(r"Dew point temperature in [$°C$]")
    plt.tight_layout()
    plt.show()





