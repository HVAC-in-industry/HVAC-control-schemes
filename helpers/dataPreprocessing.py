"""
-------------------------------------------------------------------------------
Name:        dataPreprocessing
Purpose:     Pre-processing of the local test data

Author:      Christian Buchholz, Marcus Vogt

Created:     01.12.2021
Copyright:   Chair of Sustainable Manufacturing and Life Cycle Engineering, Institute of Machine Tools and Production Technology, Technische Universität Braunschweig, Langer Kamp 19b, 38106 Braunschweig, Germany
Licence:     MIT (see License)
-------------------------------------------------------------------------------
"""

import numpy as np

def humanheat (activity):
    # regarding to VDI 2078 p.27
    # in [W]
    if activity == 1:
        heatFlowHuman = 100
    elif activity == 2:
        heatFlowHuman = 125
    elif activity == 3:
        heatFlowHuman = 170
    elif activity == 4:
        heatFlowHuman = 210
    else:
        raise Exception('wrong input for activity in modul "helpers.dataPreprocessing", function "humanheat". Allowed input arguments are: 1, 2, 3, 4')
    return heatFlowHuman # [W]



def humanhumidity (T_Ges, activity):
    # regarding to VDI 2078 p. 26
    # in [kg/s]

    if activity == 1 :
        X_activity = (-86 + 5.4 * T_Ges) / (3600 * 1000)  # Activity I ; at 20°C 22 g/h
    elif activity == 2 :
        X_activity = (-58 + 5.4 * T_Ges) / (3600 * 1000)  # Activity II ; at 20°C 50 g/h
    elif activity == 3 :
        X_activity = (-18 + 5.8 * T_Ges) / (3600 * 1000)  # Activity III at 20°C 98g/h // e.g -> 0.0000272 kg/s
    elif activity == 4 :
        X_activity = (-75 + 9.4 * T_Ges) / (3600 * 1000)  # Activity IV ; at 20°C 113 g/h
    else:
        raise Exception('wrong input for activity in modul "helpers.dataPreprocessing", function "humanhumidity". Allowed input arguments are: 1, 2, 3, 4')
    return X_activity

def short_new(data, defined_length):

    """
    This function shortens the data series "data" to the specified length "defined_length". This is done by averaging over several data points of the original data series. Important: len(data) must be an integer multiple of defined_length.
    """

    long_data = len(data)
    short_data = defined_length
    q = int(long_data / short_data)
    data_new = np.zeros(short_data)

    for j in range(short_data):
        k = 0
        for i in range(q):
            if i + q * j >= long_data:
                break
            k = k + data[q * j + i]
        data_new[j] = k / q

    return data_new


def long_new(data, defined_length):
    """
    This function plugs the data series "data" into the specified length "defined_length". This is done by copying existing data points to create the missing data points. Important: defined_length must be an integer multiple of len(data).
    """

    long_data = defined_length
    short_data = len(data)
    q = int(long_data / short_data)
    data_new = np.zeros(long_data)

    for j in range(short_data):
        for i in range(q):
            if i + q * j >= long_data:
                break
            data_new[i + q * j] = data[j]

    return data_new


def rescale_data(data, defined_length):

    """
    This function calls the functions short_new() or long_new() depending on the length and desired length of the data series.
    """

    if len(data) < defined_length:

        data_new = long_new(data, defined_length)
        return data_new

    elif len(data) > defined_length:

        data_new = short_new(data, defined_length)
        return data_new

    elif len(data) == defined_length:

        data_new = data
        return data_new

def room_model(x, t, m_in, T_in, X_in, m_X_del, beta_CO2_in, beta_CO2_prod, m_prod, Q_gain, T_amb_room, k, A, C_sub, specificHeatCapacityDryAir, specificHeatCapacityWaterVapor, delta_h_water, m_air_room):

    """
    This function implements a differential equation system that describes the dynamic spatial behaviour.
    :param x: float: initial states
    :param t: float: time series for integration
    :param m_in: float: value of the air mass flow entering the room [kg/s]
    :param T_in: float: temperature of the air mass flow entering the room [°C]
    :param X_in: float: humidity of the air mass flow entering the room [kg water / kg air]
    :param m_X_del: float: moisture load [kg/s]
    :param beta_CO2_in: float: CO2-concentration of the air mass flow entering the room [ppm]
    :param beta_CO2_prod: float: CO2 concentration of the air emitted by humans [ppm]
    :param m_prod: float: air mass flow emitted by humans [kg/s]
    :param Q_gain: float: heat input into the room [W]
    :param T_amb_room: float: temperature in the building outside the room [°C]
    :param k: float: heat transfer coefficient [J/(m^2*K)]
    :param A: float: wall surface [m^2]
    :param C_sub: float: heat capacity [J]
    :param specificHeatCapacityDryAir: float: specific heat capacity of dry air [J/kg]
    :param specificHeatCapacityWaterVapor: float: specific heat capacity of water vapor [J/kg]
    :param delta_h_water: float: Specific evaporation enthalpy of water [J/kg]
    :param m_air_room: float: Mass of the air inside the room [kg]
    :return dxdt: float: solution of ODE
    """

    # summarising some expressions into coefficients in order to subsequently shorten the model equations

    c1 = (k * A) / C_sub
    c2 = specificHeatCapacityDryAir / C_sub
    c3 = specificHeatCapacityWaterVapor / C_sub
    c4 = delta_h_water / C_sub
    c5 = 1 / C_sub
    c6 = 1 / m_air_room

    T_room = x[0]
    X_room = x[1]
    beta_CO2_room = x[2]
    dT_roomdt = -c1 * T_room - c2 * m_in * T_room - c3 * m_in * X_room * T_room + c2 * m_in * T_in + c4 * m_in * X_in + c3 * m_in * X_in * T_in - c4 * m_in * X_room + c5 * Q_gain + c1 * T_amb_room
    dX_roomdt = -c6 * m_in * X_room + c6 * m_in * X_in + c6 * m_X_del
    dbeta_CO2_roomdt = c6 * (beta_CO2_in * m_in - beta_CO2_room * m_in + beta_CO2_prod * m_prod)
    dxdt = [dT_roomdt, dX_roomdt, dbeta_CO2_roomdt]

    return dxdt







