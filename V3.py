"""
-------------------------------------------------------------------------------
Name:        V3
Purpose:     P-Control with prediction

Author:      Christian Buchholz, Marcus Vogt

Created:     01.12.2021
Copyright:   Chair of Sustainable Manufacturing and Life Cycle Engineering, Institute of Machine Tools and Production Technology, Technische Universität Braunschweig, Langer Kamp 19b, 38106 Braunschweig, Germany
Licence:     MIT (see License)
-------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import os
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import helpers.WetAirToolBox as WetAirToolBox
import helpers.dataPreprocessing as dataPreprocessing
from V2 import P_controller

def predictive_p_controller(i, u_min, u_max, x0, T_TP_room_set, N, t_step_controller, P, TemperatureIn_prediction, DewPointIn_set_prediction, m_X_delta_prediction, CO2_in_prediction):
    """
    This function implements a room model in order to be able to predict the behaviour of the dry room under consideration. The function also calculates the manipulated variable of the current time step on the basis of the prediction.
    :param i: int: number of the current time step
    :param u_min: float: min. value of control value u
    :param u_max: float: max. value of control value u
    :param x0: float: current system states
    :param T_TP_room_set: float: set point for the dew point [°C]
    :param N: int: length of the prediction horizon as number of time steps
    :param t_step_controller: int: length of a time steps between two calculations of the manipulated variable [sec]
    :param P: float: slope of the proportional controller
    :param TemperatureIn_prediction: float: predicted temperature curve of the supply air over the prediction horizon [°C]
    :param DewPointIn_set_prediction: float: predicted dew point curve of the supply air over the prediction horizon [°C]
    :param m_X_delta_prediction: float: predicted moisture load curve over the prediction horizon [kg water / s]
    :param CO2_in_prediction: float: predicted CO2-concentration of the supply air over the prediction horizon
    :return u0: float: value of the manipulated variable in the current time step
    """

    beta_CO2_prod = 0   # CO2 concentration of air emitted by humans [ppm] (is neglected in this example)
    m_prod = 0 # Air mass flow of air emitted by humans [kg/s] (is neglected in this example)
    Q_gain = 6000 # additional heat flow into the room (during the day) [W]
    T_amb_room = 22 # Air temperature in the building outside the dry room [°C]

    # air properties
    specificHeatCapacityDryAir = 1006 # [kJ/kg]
    specificHeatCapacityWaterVapor = 1869 # [kJ/kg]
    delta_h_water = 2256600 # specific enthalpy of vaporization of water [kJ/kg]
    
    m_air_room = 593.9433287999999  # mass of air in the room [kg]
    k = 5.4 # heat transfer coefficient [W/(m^2*K)]
    A = 495.24 # wall area [m^2]
    C_sub = 671346.88 # heat capacity of the room [J]

    # actual values of the system states temperatur, humidity and CO2-concentration
    T_room_act = x0[0] # [°C]
    X_room_act = x0[1] # [kg water / kg air]
    beta_CO2_room_act = x0[2] # [ppm]

    X_room_set = WetAirToolBox.humidity_dewpoint2abs(T_room_act, T_TP_room_set)

    z0 = np.zeros(3)
    z0[0] = T_room_act
    z0[1] = X_room_act
    z0[2] = beta_CO2_room_act

    # time points
    t = np.linspace(0, N * t_step_controller, N + 1)

    m_in = u_min

    # Save solution in array
    T_room = np.zeros(len(t))
    X_room = np.zeros(len(t))
    beta_CO2_room = np.zeros(len(t))
    # record initial conditions
    T_room[0] = z0[0]
    X_room[0] = z0[1]
    beta_CO2_room[0] = z0[2]

    TemperatureIn = TemperatureIn_prediction[i:i + N + 1:1]
    DewPointIn_set = DewPointIn_set_prediction[i:i + N + 1:1]
    m_X_delta = m_X_delta_prediction[i:i + N + 1:1]
    CO2_in = CO2_in_prediction[i: i + N +1:1]

    X_in = WetAirToolBox.humidity_dewpoint2abs(T_room_act, DewPointIn_set)

    # solve ODE
    for j in range(1, N+ 1):
        # span for next time step
        tspan = [t[j - 1], t[j]]
        # solve for next step
        inputs = (m_in, TemperatureIn[j], X_in[j], m_X_delta[j],
                  CO2_in[j], beta_CO2_prod, m_prod, Q_gain, T_amb_room, k, A, C_sub, specificHeatCapacityDryAir, specificHeatCapacityWaterVapor,    delta_h_water, m_air_room)
        z = odeint(dataPreprocessing.room_model, z0, tspan, args=inputs)
        # save solution for plotting
        T_room[j] = z[1][0]
        X_room[j] = z[1][1]
        beta_CO2_room[j] = z[1][2]
        # next initial condition
        z0 = z[1]

    r = 0
    for n in range(N + 1):
        r = r + X_room[n]
    r = r / (N + 1)

    u0 = P_controller(r, X_room_set, u_max, u_min, P)

    return u0

if __name__ == '__main__':

    dataPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
    data = pd.read_excel(os.path.join(dataPath, "t_step300_kp-30000_Tf4000_u_massflow.xlsx"), engine='openpyxl')
    moisture_load = pd.read_csv(os.path.join(dataPath, "moisture_load.csv"))

    T_TP_room_set = -60 # Dew point temperature of the supply air entering the room [°C]
    P = -100000 # Slope of the proportional controller
    t_step_controller = 300 # length of a time steps between two calculations of the manipulated variable [sec]
    t_step_Measurements_BLB = 60 # length of a time steps between two measurements of the state variables [sec]
    N = 7 # ength of the prediction horizon as number of time steps
    u_min = 1.63 # minimum value of manipulated variable / mass flow [kg/s]
    u_max = 3.55 # maximum value of manipulated variable / mass flow [kg/s]
    T_room_0 = 20 # initial value of temperatur in the room [°C]
    T_TP_room_0 = -50 # initial value of dew point temperatur in the room [°C]
    X_room_0 = WetAirToolBox.humidity_dewpoint2abs(T_room_0, T_TP_room_0) # initial value of humidity in the room [kg water / kg air]
    beta_CO2_room_0 = 350 # initial value of CO2-concentration in the room [ppm]

    CO2_in = data.beta_CO2_in
    CO2_in_prediction = dataPreprocessing.rescale_data(CO2_in, int((t_step_Measurements_BLB / t_step_controller) * len(CO2_in)))

    m_X_delta = moisture_load.moisture_load
    m_X_delta_prediction = dataPreprocessing.rescale_data(m_X_delta, int((t_step_Measurements_BLB / t_step_controller) * len(m_X_delta)))

    TemperatureIn_prediction = np.ones(1432)*20
    DewPointIn_set_prediction = np.ones(1432)*(-60)

    # This part is for testing the code. The control variable curve shown is based on sample curves of the state variables and there is no closed control loop.
    Range = 200
    t = np.arange(0, Range*t_step_controller, t_step_controller)
    fig, ax = plt.subplots()
    for i in range(Range):
        if i == 0:
            x0 = np.array([T_room_0, X_room_0, beta_CO2_room_0]).reshape(-1, 1)
            u0 = predictive_p_controller(i, u_min, u_max, x0, T_TP_room_set, N, t_step_controller, P, TemperatureIn_prediction, DewPointIn_set_prediction, m_X_delta_prediction, CO2_in_prediction)
            ax.scatter(t[i], u0, color='blue')
        else:
            x0 = np.array([data.T_room[i], data.X_room[i], data.beta_CO2_room[i]]).reshape(-1, 1)
            u0 = predictive_p_controller(i, u_min, u_max, x0, T_TP_room_set, N, t_step_controller, P, TemperatureIn_prediction, DewPointIn_set_prediction, m_X_delta_prediction, CO2_in_prediction)
            ax.scatter(t[i], u0, color='blue')
            
    ax.set_xlabel('time [s]')
    ax.set_ylabel('air mass flow [kg/s]')
    plt.show()

