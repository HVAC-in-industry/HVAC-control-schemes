"""
-------------------------------------------------------------------------------
Name:        HVAC
Purpose:     Ventilation system map

Author:      Christian Buchholz, Marcus Vogt

Created:     01.12.2021
Copyright:   Chair of Sustainable Manufacturing and Life Cycle Engineering, Institute of Machine Tools and Production Technology, Technische Universität Braunschweig, Langer Kamp 19b, 38106 Braunschweig, Germany
Licence:     MIT (see License)
-------------------------------------------------------------------------------
"""

import helpers.WetAirToolBox as WetAirToolBox
from casadi import *


def Ventilator(P1, m1, Eta, V, airDensity, specificHeatCapacityDryAir, calc):
    """
    This function calculates the power of a ventilator as well as the increase in air temperature caused by the ventilator.
    :param P1: float: nominal power of the ventilator [W]
    :param m1: float: nominal air mass flow of the ventilator [kg/s]
    :param Eta: float: efficiency coefficient of the ventilator
    :param V: float: current volume flow of the ventilator [m^2/s]
    :param airDensity: float: air density [kg/m^3]
    :param specificHeatCapacityDryAir: float: specific heat capacity of dry air [J/kg]
    :param calc: float: defines the return value(s). (Power, TemperatureIncrease, Power&TemperatureIncrease)
    :return P float: power demand of the ventilator [W]
    :return delta_T_out: float: temperature increase due to ventilation [°C]
    """

    m = airDensity * V
    P = ((m / m1) ** 3) * P1
    delta_T_out = (P * Eta) / (m * specificHeatCapacityDryAir)
    if calc == 'Power':
        return P
    elif calc == 'TemperatureIncrease':
        return delta_T_out
    elif calc == 'Power&TemperatureIncrease':
        return P, delta_T_out
    else:
        raise Exception('Wrong input for "calc". Allowed are: Power, TemperatureIncrease, Power&TemperatureIncrease')


def Heatflow_fromto_Fluidflow(m_VK, T_VK_in, T_VK_out, X_VK_in, specificHeatCapacityDryAir,
                              specificHeatCapacityWaterVapor, delta_h_water, calc):
    """
    This function calculates the heat flow that is added to or removed from an air flow when it undergoes a change of state. It also calculates the humidity at the end of the change of state.
    :param m_VK: float: value of the air mass flow [kg/s]
    :param T_VK_in: float: temperature at the beginning [°C]
    :param T_VK_out: float: temperature at the end [°C]
    :param X_VK_in: float: humidity at the beginning [kg water / kg air]
    :param specificHeatCapacityDryAir: float: specific heat capacity of dry air [J/kg]
    :param specificHeatCapacityWaterVapor: float: specific heat capacity of water vapor [J/kg]
    :param delta_h_water: float: specific evaporation enthalpy [J/kg]
    :param calc: float: defines the return value(s). (Q -> heat flow, X -> humidity at the end, Q&X -> heat flow and humidity at the end)
    :return Q: float: heat flow [W]
    :return X: float: humidity at the end of the change of state [kg water / kg air]
    """

    T_TP_in = WetAirToolBox.humidity_abs2dewpoint(X_VK_in)

    X_VK_out = if_else(T_TP_in <= T_VK_out, X_VK_in, WetAirToolBox.humidity_dewpoint2abs(T_VK_out, T_VK_out))

    h_VK_in = specificHeatCapacityDryAir * T_VK_in + X_VK_in * (
                delta_h_water + specificHeatCapacityWaterVapor * T_VK_in)
    h_VK_out = specificHeatCapacityDryAir * T_VK_out + X_VK_out * (
                delta_h_water + specificHeatCapacityWaterVapor * T_VK_out)

    m_K = m_VK * (X_VK_in - X_VK_out)
    h_K = specificHeatCapacityWaterVapor * T_VK_out

    Q_VK = if_else(T_TP_in <= T_VK_out, m_VK * (h_VK_in - h_VK_out), m_VK * (h_VK_in - h_VK_out) - m_K * h_K)

    Q_VK = if_else(Q_VK < 0, Q_VK * (-1), Q_VK)

    if calc == 'Q':
        return Q_VK
    elif calc == 'X':
        return X_VK_out
    elif calc == 'Q&X':
        return Q_VK, X_VK_out


def MassFlow_Mixing(m1, T1, X1, m2, T2, X2, specificHeatCapacityDryAir, specificHeatCapacityWaterVapor, delta_h_water,
                    calc):
    """
    This function the resulting state variables when mixing two air flows.
    :param m1: float: value of the first air mass flow [kg/s]
    :param T1: float: temperature of the first air mass flow [°C]
    :param X1: float: humidity of the first air mass flow [kg water / kg air]
    :param m2: float: value of the second air mass flow [kg/s]
    :param T2: float: temperature of the second air mass flow [°C]
    :param X2: float: humidity of the second air mass flow [kg water / kg air]
    :param specificHeatCapacityDryAir: float: specific heat capacity of dry air [J/kg]
    :param specificHeatCapacityWaterVapor: float: specific heat capacity of water vapor [J/kg]
    :param delta_h_water: float: specific evaporation enthalpy [J/kg]
    :param calc: float: defines the return value(s). (Humidity(abs), Temperature)
    :return Xa float: resulting humidity [kg water / kg air]
    :return Ta float: resulting temperature [°C]
    """

    Xa = (m1 / (m1 + m2)) * X1 + (m2 / (m1 + m2)) * X2

    Ta = (m1 * (specificHeatCapacityDryAir * T1 + X1 * (delta_h_water + specificHeatCapacityWaterVapor * T1))
          + m2 * (specificHeatCapacityDryAir * T2 + X2 * (delta_h_water + specificHeatCapacityWaterVapor * T2))
          - (m1 + m2) * Xa * delta_h_water) / (
                     (m1 + m2) * (specificHeatCapacityDryAir + Xa * specificHeatCapacityWaterVapor))

    if calc == 'Humidity(abs)':
        return Xa
    elif calc == 'Temperature':
        return Ta


def P_RegHeater(MRC):
    """
    This function describes a simple relationship between the amount of water to be dehumidified by the desiccant wheel and the output of the regeneration heater
    :param MRC: float: moisture removal capacity [l/h]
    :return P: float: power demand of the regeneration heater [W]
    """

    MRC0 = 20
    P = (105 / 20) * MRC + (20 - (105 / 20) * MRC0)
    P = if_else(P > 125, 125, P)
    P = if_else(P < 20, 20, P)
    P = P * 1000

    return P






