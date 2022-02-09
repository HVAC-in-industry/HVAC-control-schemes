"""
-------------------------------------------------------------------------------
Name:        V4
Purpose:     Model-predictive controller with optimization designated as V4a and V4b in the paper

Author:      Christian Buchholz, Marcus Vogt

Created:     01.12.2021
Copyright:   Chair of Sustainable Manufacturing and Life Cycle Engineering, Institute of Machine Tools and Production Technology, Technische Universität Braunschweig, Langer Kamp 19b, 38106 Braunschweig, Germany
Licence:     MIT (see License)
-------------------------------------------------------------------------------
"""

import do_mpc
from casadi import *
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import helpers.WetAirToolBox as WetAirToolBox
import helpers.HVAC as HVAC
import helpers.dataPreprocessing as dataPreprocessing

# air properties
volumeFlowRespirationHuman = 1 / 3000
airDensity = 1.17343
specificHeatCapacityDryAir = 1006
specificHeatCapacityWaterVapor = 1869
delta_h_water = 2256600

def room_model(q, mpc_type):
    """
    This function implements the spatial model with symbolic variables.
    :param q: int: weighting factor of the quadratic control difference in the objective function of the MPC
    :param mpc_type: int: there are two options for the mpc_type: mpc_type = 'economic' -> V4a, mpc_type = 'standard' -> V4b
    :return (float): returns the model object
    """

    # dry room
    m_air_room = 593.9433287999999  # mass of air in the room [kg]
    k = 5.4 # heat transfer coefficient [W/(m^2*K)]
    A = 495.24 # wall area [m^2]
    C_sub = 671346.88 # heat capacity of the room [J]

    # summarising some expressions into coefficients in order to subsequently write the model equations more clearly

    c1 = (k * A) / C_sub
    c2 = specificHeatCapacityDryAir / C_sub
    c3 = specificHeatCapacityWaterVapor / C_sub
    c4 = delta_h_water / C_sub
    c5 = 1 / C_sub
    c6 = 1 / m_air_room

    model_type = 'continuous'
    model = do_mpc.model.Model(model_type)

    """
    definition of the variables of the model as symbolic variables
    """
    # definition of system states
    T_room = model.set_variable(var_type='_x', var_name='T_room', shape=(1, 1))
    X_room = model.set_variable(var_type='_x', var_name='X_room', shape=(1, 1))
    beta_CO2_room = model.set_variable(var_type='_x', var_name='beta_CO2_room', shape=(1, 1))

    # set point
    T_TP_room_set = model.set_variable(var_type='_tvp', var_name='T_TP_room_set')

    # manipulated variable
    m_in = model.set_variable('_u', 'm_in', shape=(1, 1))

    # uncontrolled input variables (defined as time variing )
    T_in = model.set_variable('_tvp', 'T_in', shape=(1, 1))
    X_in_set = model.set_variable('_tvp', 'X_in_set', shape=(1, 1))

    T_amb_room = model.set_variable(var_type='_tvp', var_name='T_amb_room') # temperature in the building outside the room [°C]
    beta_CO2_in = model.set_variable(var_type='_tvp', var_name='beta_CO2_in') # CO2-concentration of the air mass flow m_in [ppm]
    n_hum = model.set_variable(var_type='_tvp', var_name='n_hum') # number of humans in the room
    beta_CO2_prod = 50000 # CO2-concentration of the air emitted by humans in the room [ppm]
    m_prod = volumeFlowRespirationHuman * airDensity # value of air mass flow emitted by humans [kg/s]
    Q_gain = model.set_variable(var_type='_tvp', var_name='Q_gain') # heat input into the room [W]
    Q_rad = 0 # Heat input through thermal radiation [W]
    m_X_del = model.set_variable(var_type='_tvp', var_name='m_X_del') # moisture load [kg water/s]

    # Definition of first order ODEs
    model.set_rhs('T_room', -c1 * T_room - c2 * m_in * T_room - c3 * m_in * X_room * T_room + c2 * m_in * T_in + c4 * m_in * X_in_set
                  + c3 * m_in * X_in_set * T_in - c4 * m_in * X_room + c5 * Q_gain + c5 * Q_rad + c1 * T_amb_room)
    model.set_rhs('X_room', -c6 * m_in * X_room + c6 * m_in * X_in_set + c6 * m_X_del)
    model.set_rhs('beta_CO2_room', c6 * (beta_CO2_in * m_in - beta_CO2_room * m_in + n_hum * beta_CO2_prod * m_prod))


    T_TP_room = WetAirToolBox.humidity_abs2dewpoint(X_room)
    model.set_expression('T_TP_room', T_TP_room)


    T_TP_in = WetAirToolBox.humidity_abs2dewpoint(X_in_set)
    model.set_expression('T_TP_in', T_TP_in)

    model, P_el, P_fw, P_gas = cost_model(model, m_in, X_in_set, T_in, T_room, X_room, T_TP_room)
    model = objective_function(model, mpc_type, T_TP_room, T_TP_room_set, P_el, P_fw, q)

    model.setup()

    return model


def cost_model(model, m_in, X_in_set, T_in, T_room, X_room, T_TP_room):

    """
    This function implements the HVAC model used for the V4a controller variant.
    All inputs are symbolic variables.
    :param model: model object
    :param m_in: air mass flow going into the room / manipulated variable [kg/s]
    :param X_in_set: humidity of the air mass flow m_in [kg water / kw air]
    :param T_in: temperature of the air mass flow m_in [°C]
    :param T_room: temperature of the air in the room [°C]
    :param X_room: humidity of the air in the room [kg water / kg air]
    :param T_TP_room: dew point of the air in the room [°C]
    :return: model object
    :return: electrical power demand of the HVAC-system P_el
    :return: power demand of the supply air heater of the HVAC system by district heating P_fw
    :return: power demand of the regeneration heater of the HVAC system by burning gas P_gas

    """

    T_amb_buil = model.set_variable(var_type='_tvp', var_name='T_amb_buil') # ambient temperature [°C]
    X_amb_buil = model.set_variable(var_type='_tvp', var_name='X_amb_buil') # ambient humidity [kg water / kg air]

    # paramters of ventiators

    P1_Proz_Vent = 15000 # nominal power of process ventilator [W]
    V1_Proz_Vent = 11000 # nominal volume flow of process ventilator [m^3/h]
    m1_Proz_Vent = (airDensity * V1_Proz_Vent) / 3600 # nominal air mass flow of process ventilator [kg/s]
    Eta_Proz_Vent = 0.693 # efficiency coefficient of process ventilator

    P1_Reg_Vent = 3000 # nominal power of regeneration ventilator [W]
    V1_Reg_Vent = 3580 # nominal volume flow of regeneration ventilator [m^3/h]
    m1_Reg_Vent = (airDensity * V1_Reg_Vent) / 3600 # nominal air mass flow of regeneration ventilator [kg/s]
    Eta_Reg_Vent = 0.8 # efficiency coefficient of regeneration ventilator

    FlowLoss = 0.05 # 5 % of process volume flow are used as purge volume flow
    AirRatio = 0.7045 # V_ReturnAir / V_ProcessVent

    # parameters of supply air heater
    Eta_ZL = 0.22 # efficiency coefficient of supply air heater

    # parameters of pre-cooler 1
    Eta_VK_1 = 1.45 # performance number of pre-cooler 1
    T_VK1_out = model.set_variable(var_type='_tvp', var_name='T_VK1_out')  # temperature set point for pre-cooler 1 [°C]

    # parameters of pre-cooler 2
    Eta_VK_2 = Eta_VK_1 #performance number of pre-cooler 2
    T_VK2_out = model.set_variable(var_type='_tvp', var_name='T_VK2_out') # temperature set point for pre-cooler 2 [°C]

    # perameters of desiccant wheel
    T_Sorption_out = 14  # temperature at process air outlet [°C]

    # return air dehumidification (parameters of equation 17)
    humidification_ReturnAir_slope = 0.5032990300444488
    humidification_ReturnAir_intercept = -15.834114072180622


    V_Proz_Vent = m_in / (airDensity * (1 - FlowLoss))  # equation 27
    model.set_expression('V_Proz_Vent', V_Proz_Vent)

    V_P_Zuluft = V_Proz_Vent * (1 - AirRatio)   # equation 21
    model.set_expression('V_P_Zuluft', V_P_Zuluft)

    V_Umluft = V_Proz_Vent * AirRatio   # return air flow
    model.set_expression('V_Umluft', V_Umluft)

    V_Reg_Vent = V_Proz_Vent / (V1_Proz_Vent/V1_Reg_Vent)
    model.set_expression('V_Reg_Vent', V_Reg_Vent)

    V_VK1 = (1 - (7750 / 11000)) * V_Proz_Vent # volume flow through pre-cooler 1
    model.set_expression('V_VK1', V_VK1)

    # equations 19
    P_Proz_Vent = HVAC.Ventilator(P1_Proz_Vent, m1_Proz_Vent, Eta_Proz_Vent, V_Proz_Vent, airDensity, specificHeatCapacityDryAir, calc='Power')
    model.set_expression('P_Proz_Vent', P_Proz_Vent)
    P_Reg_Vent = HVAC.Ventilator(P1_Reg_Vent, m1_Reg_Vent, Eta_Reg_Vent, V_Reg_Vent, airDensity, specificHeatCapacityDryAir, calc='Power')
    model.set_expression('P_Reg_Vent', P_Reg_Vent)

    # equation 23
    (Q_VK1, X_VK1_out) = HVAC.Heatflow_fromto_Fluidflow(airDensity * V_VK1, T_amb_buil, T_VK1_out, X_amb_buil,
                                                       specificHeatCapacityDryAir, specificHeatCapacityWaterVapor,
                                                       delta_h_water, calc='Q&X')
    model.set_expression('Q_VK1', Q_VK1)
    model.set_expression('X_VK1_out', X_VK1_out)

    # equation 15
    T_VK2_in = HVAC.MassFlow_Mixing(airDensity * V_P_Zuluft, T_VK1_out, X_VK1_out, airDensity * V_Umluft, T_room, X_room,
                                   specificHeatCapacityDryAir, specificHeatCapacityWaterVapor, delta_h_water,
                                   calc='Temperature')

    # equation 18
    T_VK2_in = T_VK2_in + HVAC.Ventilator(P1_Proz_Vent, m1_Proz_Vent, Eta_Proz_Vent, airDensity, V_Proz_Vent,
                                         specificHeatCapacityDryAir, calc='TemperatureIncrease')

    model.set_expression('T_VK2_in', T_VK2_in)

    # equation 17
    T_TP_ReturnAir = humidification_ReturnAir_slope * T_TP_room + humidification_ReturnAir_intercept
    model.set_expression('T_TP_ReturnAir', T_TP_ReturnAir)

    # equation 14
    X_VK2_in = HVAC.MassFlow_Mixing(airDensity * V_P_Zuluft, T_VK1_out, X_VK1_out, airDensity * V_Umluft, T_room, WetAirToolBox.humidity_dewpoint2abs(T_room, T_TP_ReturnAir),
                                   specificHeatCapacityDryAir, specificHeatCapacityWaterVapor, delta_h_water,
                                   calc='Humidity(abs)')
    model.set_expression('X_VK2_in', X_VK2_in)

    # equation 24
    (Q_VK2) = HVAC.Heatflow_fromto_Fluidflow(airDensity * V_Proz_Vent, T_VK2_in, T_VK2_out, X_VK2_in,
                                            specificHeatCapacityDryAir, specificHeatCapacityWaterVapor, delta_h_water,
                                            calc='Q')
    model.set_expression('Q_VK2', Q_VK2)

    # equation 29
    Q_ZL = HVAC.Heatflow_fromto_Fluidflow(m_in, T_Sorption_out, T_in, X_in_set, specificHeatCapacityDryAir,
                                         specificHeatCapacityWaterVapor, delta_h_water, calc='Q')
    model.set_expression('Q_ZL', Q_ZL)

    # equation 22
    P_VK1 = Q_VK1 / Eta_VK_1
    model.set_expression('P_VK1', P_VK1)
    P_VK2 = Q_VK2 / Eta_VK_2
    model.set_expression('P_VK2', P_VK2)
    P_VK12 = P_VK1 + P_VK2
    model.set_expression('P_VK12', P_VK12)

    # equation 28
    P_ZL = Q_ZL / Eta_ZL
    model.set_expression('P_ZL', P_ZL)

    P_el = P_VK1 + P_VK2 + P_Proz_Vent + P_Reg_Vent
    model.set_expression('P_el', P_el)

    P_fw = P_ZL
    model.set_expression('P_fw', P_fw)

    # moisture removal capacity of the desiccant wheel
    MRC = 3600 * airDensity * V_Proz_Vent * (X_VK2_in - (1 - FlowLoss) * X_in_set)  # X_VK2_in = X_VK2_out
    model.set_expression('MRC', MRC)


    P_RegHeater = HVAC.P_RegHeater(MRC)
    model.set_expression('P_RegHeater', P_RegHeater)

    P_gas = P_RegHeater
    model.set_expression('P_gas', P_gas)

    return model, P_el, P_fw, P_gas

def objective_function(model, mpc_type, T_TP_room, T_TP_room_set, P_el, P_fw, q):
    """
    This function defindes the lagrange_term (which is also used as mayer-term) and the function f(m_in) from equation 10
    All inputs are symbolic variables.
    :param model: model object
    :param mpc_type: there are two options for the mpc_type: mpc_type = 'economic' -> V4a, mpc_type = 'standard' -> V4b
    :param T_TP_room: dew point of the air in the room [°C]
    :param T_TP_room_set: set point of the dew point of the air in the room [°C]
    :param P_el: electrical power demand of the HVAC-system P_el
    :param P_fw: power demand of the supply air heater of the HVAC system by district heating P_fw
    :param q: weighting factor of the quadratic control difference in the objective function of the MPC
    :return: model object
    """


    lagrange_term = (q * (T_TP_room - T_TP_room_set)**2) # part of equation 10

    # defining f(m_in)
    if mpc_type == 'default' or mpc_type == 'standard':

        u = list()
        u.append('m_in')
        cost = 0
        for i in range(len(u)):
            cost += model.u[u[i]]

    elif mpc_type == 'economic':

        cost = P_el + P_fw

    else:
        raise Exception('wrong input argument for "mpc_type" in modul "mpc_model", function "objective_function". Allowed arguments are: standard, economic.')


    model.set_expression('cost', cost)
    model.set_expression('lagrange_term', lagrange_term)

    return model




def mpc_controller(r, model, t_step_controller, n_horizon, T_TP_room_set, TemperatureIn, DewPointIn_set, T_amb_room, AmbientTemperature, NumOfPersons_prediction, AmbientHumidityAbs, m_X_delta, activity):

    """
    :param r: weighting of the rate of change of the manipulated variable
    :param model: model object
    :param t_step_controller: length of a time steps between two calculations of the manipulated variable [sec]
    :param n_horizon: number of steps within the prediction horizon
    :param T_TP_room_set: set point of the dew point of the air in the room [°C]
    :param TemperatureIn: predicted temperature curve of the supply air over the prediction horizon [°C]
    :param DewPointIn_set: predicted dew point curve of the supply air over the prediction horizon [°C]
    :param T_amb_room: temperature in the building outside the room [°C]
    :param AmbientTemperature: predicted ambient temperature [°C]
    :param NumOfPersons_prediction: predicted number of humans in the room
    :param AmbientHumidityAbs: predicted ambient humidity
    :param m_X_delta: predicted moisture load curve over the prediction horizon [kg water / s]
    :param activity: degree of physical work of the humans in the room (see helpers.dataPreprocessing.humanheat)
    :return: mpc object
    """


    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': n_horizon,
        't_step': t_step_controller,
        'n_robust': 0,
        'state_discretization': 'collocation',
        'collocation_type': 'radau',
        'collocation_deg': 1,
        'collocation_ni': 1,
        'store_full_solution': True,
        'nlpsol_opts': {'ipopt.linear_solver': 'mumps'}
    }
    mpc.set_param(**setup_mpc)

    # defining the objective function (equation 10)
    terminal_cost = model.aux['lagrange_term']
    stage_cost = model.aux['lagrange_term'] + model.aux['cost']
    mpc.set_objective(mterm=terminal_cost, lterm=stage_cost)
    mpc.set_rterm(m_in=r)

    tvp_template_mpc = mpc.get_tvp_template()

    # Here, the curves of the time-varying parameters for the current optimal control problem are read in at each time step. The variable t_now identifies the current time step.
    def tvp_fun_mpc(t_now):
        now = int(t_now // t_step_controller)

        for k in range(n_horizon + 1):
            tvp_template_mpc['_tvp', k, 'X_in_set'] = WetAirToolBox.humidity_dewpoint2abs(TemperatureIn[k + now], DewPointIn_set[k + now])
            tvp_template_mpc['_tvp', k, 'T_in'] = TemperatureIn[k + now]
            tvp_template_mpc['_tvp', k, 'T_TP_room_set'] = T_TP_room_set[k + now]
            tvp_template_mpc['_tvp', k, 'T_amb_buil'] = AmbientTemperature[k + now]
            tvp_template_mpc['_tvp', k, 'n_hum'] = NumOfPersons_prediction[k + now]
            tvp_template_mpc['_tvp', k, 'X_amb_buil'] = AmbientHumidityAbs[k + now]
            tvp_template_mpc['_tvp', k, 'Q_gain'] = if_else(NumOfPersons_prediction[k + now] > 0, NumOfPersons_prediction[k + now] * dataPreprocessing.humanheat(activity) + 6000, NumOfPersons_prediction[k + now] * dataPreprocessing.humanheat(activity))
            tvp_template_mpc['_tvp', k, 'm_X_del'] = m_X_delta[k + now]
            tvp_template_mpc['_tvp', k, 'T_VK1_out'] = 4
            tvp_template_mpc['_tvp', k, 'T_VK2_out'] = 8
            tvp_template_mpc['_tvp', k, 'T_amb_room'] = T_amb_room[k + now]
        return tvp_template_mpc

    mpc.set_tvp_fun(tvp_fun_mpc)
    mpc.bounds['lower', '_x', 'beta_CO2_room'] = 0
    mpc.bounds['upper', '_x', 'beta_CO2_room'] = 1000

    # This is an optional soft constraint on the humidity of the air in the room
    #mpc.set_nl_cons('X_room', model.x['X_room'], ub=WetAirToolBox.humidity_dewpoint2abs(20, -48.5), soft_constraint=True, penalty_term_cons=1e12)


    mpc.bounds['lower', '_u', 'm_in'] = 1.63
    mpc.bounds['upper', '_u', 'm_in'] = 3.55  # m_in_max = airDensity*(V_Proz_Vent_max*0.95/3600) 1.17343[kg/m^3]*(11500[m^3/h]*0.95*3600[s/h])


    mpc.scaling['_x', 'T_room'] = 1
    mpc.scaling['_x', 'X_room'] = 1
    mpc.scaling['_x', 'beta_CO2_room'] = 1

    mpc.setup()

    return mpc

if __name__ == '__main__':

    activity = 2 # degree of physical work of the humans in the room (see helpers.dataPreprocessing.humanheat)

    dataPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

    # reading weather data
    WeatherDataPath = os.path.join(dataPath, "Weatherdata.xlsx")
    Weather_LWI = pd.read_excel(WeatherDataPath, engine='openpyxl')
    AmbientTemperature = Weather_LWI.Temperatur
    AmbientDewPoint = Weather_LWI.Taupunkt
    AmbientHumidityAbs = WetAirToolBox.humidity_dewpoint2abs(AmbientTemperature, AmbientDewPoint)
    # reading moisture load data
    moisture_load = pd.read_csv(os.path.join(dataPath, "moisture_load.csv"))
    m_X_delta = moisture_load.moisture_load
    # reading predicted number of humans
    NumOfHum = pd.read_excel(os.path.join(dataPath, "NumOfHum.xlsx"), engine='openpyxl')
    NumOfPersons = NumOfHum.NumOfHum
    NumOfPersons_prediction = NumOfPersons.to_numpy()
    # reading data which includes: the CO2-concentration of air flow m_in, and examples of
    data = pd.read_excel(os.path.join(dataPath, "t_step300_kp-30000_Tf4000_u_massflow.xlsx"), engine='openpyxl')

    mpc_type = 'economic' # options: mpc_type = 'economic' -> V4a, mpc_type = 'standard' -> V4b
    q = 2000 # weighting factor of the quadratic control difference in the objective function of the MPC
    r = 50000 # weighting of the rate of change of the manipulated variable
    t_step_controller = 300 # length of a time steps between two calculations of the manipulated variable [sec]
    HorizonLength = 1 # length of prediction horizon in hours [h]
    T_TP_room_set = -60 # set point of the dew point of the air in the room [°C]
    TemperatureIn = 20 # set point of the temperature of the air mass flow m_in [°C]
    DewPointIn_set = -60 # set point of the dew point of the air mass flow m_in [°C]
    AmbientTemperature_room = 22 # temperature in the building outside the room [°C]
    # initial values in the room
    T_room_0 = 20  # °C
    T_TP_room_0 = -50 # °C
    X_room_0 = WetAirToolBox.humidity_dewpoint2abs(T_room_0, T_TP_room_0)  # kg_water/kg_dryair
    beta_CO2_room_0 = 350  # ppm CO2

    t_step_Measurements_BLB = 60 # length of a time steps between two measurements of the state variables [sec]
    t_step_Weather_LWI = 600 # length of a time steps between two weather predictions [sec]

    SimulationTime = t_step_Measurements_BLB * (len(moisture_load.index))
    hours = 3600 / t_step_controller
    n_horizon = HorizonLength * hours
    n_horizon = int(n_horizon)
    n_steps = int(SimulationTime / t_step_controller) - n_horizon

    # define arrays for constant input variables
    T_TP_room_set_prediction = T_TP_room_set * np.ones(int(SimulationTime/t_step_controller))
    DewPointIn_set_prediction = DewPointIn_set * np.ones(int(SimulationTime / t_step_controller))
    TemperatureIn_prediction = TemperatureIn * np.ones(int(SimulationTime / t_step_controller))
    AmbientTemperature_room_prediction = AmbientTemperature_room * np.ones(int(SimulationTime / t_step_controller))

    # bringing data sets with prediction data to length
    m_X_delta_prediction = dataPreprocessing.rescale_data(m_X_delta, int((t_step_Measurements_BLB / t_step_controller) * len(m_X_delta)))
    AmbientTemperature_prediction = dataPreprocessing.rescale_data(AmbientTemperature, int((t_step_Weather_LWI / t_step_controller) * len(AmbientTemperature)))
    AmbientHumidityAbs_prediction = dataPreprocessing.rescale_data(AmbientHumidityAbs, int((t_step_Weather_LWI / t_step_controller) * len(AmbientHumidityAbs)))



    prediction_model = room_model(q, mpc_type)

    mpc = mpc_controller(r, prediction_model, t_step_controller, n_horizon, T_TP_room_set_prediction,
                                 TemperatureIn_prediction, DewPointIn_set_prediction,
                                 AmbientTemperature_room_prediction, AmbientTemperature_prediction, NumOfPersons_prediction,
                                 AmbientHumidityAbs_prediction, m_X_delta_prediction, activity)


    # This part is for testing the code. The control variable curve shown is based on sample curves of the state variables and there is no closed control loop.
    Range = 200
    t = np.arange(0, Range*200*t_step_controller, t_step_controller)
    fig, ax = plt.subplots()
    for i in range(Range):
        if i == 0:
            x0 = np.array([T_room_0, X_room_0, beta_CO2_room_0]).reshape(-1, 1)
            mpc.x0 = x0
            mpc.set_initial_guess()
            u0 = mpc.make_step(x0)
            ax.scatter(t[i], u0, color='blue')
            ax.set_xlabel('time [s]')
            ax.set_ylabel('air mass flow [kg/s]')
        else:
            x0 = np.array([data.T_room[i], data.X_room[i], data.beta_CO2_room[i]]).reshape(-1, 1)
            u0 = mpc.make_step(x0)
            ax.scatter(t[i], u0, color='blue')
            ax.set_xlabel('time [s]')
            ax.set_ylabel('air mass flow [kg/s]')

    plt.show()
