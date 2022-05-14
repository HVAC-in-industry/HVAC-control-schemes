# Model-predictive control schemes for HVAC systems applied to the case of battery production

[![License](http://img.shields.io/:license-mit-blue.svg)](http://doge.mit-license.org)

In order to continuously leverage the energy efficiency potential of HVAC systems in the operation phase, automated control is essential.
To implement automated control, accurate indoor air condition monitoring in the building is necessary as part of the control approach.
For automated control, four main control schemes with increasing complexity can be distinguished with regard to the TBS:

- **V1**: Constant or scheduled operation
- **V2**: Open or closed loop based control
- **V3**: Model-predictive control without optimisation
- **V4**: Model-predictive control with optimisation

This folder contains the used control schemes **V2 - V4**.
In addition, scripts are available for communication with a programmable logic controller (PLC) for sending the set-points.
Currently, Siemens S7 protocol or OPC UA are available as protocols.
This folder is organised as follows:

```
HVAC-control-schemes
│   V2.py
│   V3.py
│   V4.py
│   README.md
│   environment.yml
│
└───data
│   │   moisture_load.csv
│   │   NumOfHum.xlsx
│   │   t_step300_kp-30000_Tf4000_u_massflow.xlsx
│   │   Weatherdata.xlsx
│
└───helpers
│   │   dataPreprocessing.py
│   │   HVAC.py
│   │   WetAirToolBox.py
│
└───plc
│   │   opcUaServer.py
│   │   plcClientCommunication.py
│   │   S7Server.py
```

The most important contents of the folder are briefly described below.

`V2.py`: Simple P-control. This is a proportional controller with a control variable limitation and a first-order low-pass filter for smoothing the control variable signal. Controlled variable: humidity, manipulated variable: air mass flow entering the room

`V3.py`: P-Control with prediction. This is a proportional controller with control value limitation extended by a room model for prediction. Via the room model, the states of the air in the room are predicted over N time steps. The resulting N control differences are averaged and fed to the proportional controller as an input variable. For prediction, the room model uses an air mass flow that corresponds to the value of the lower control variable limitation.

`V4.py`: Model-predictive controller with optimisation designated as **V4a** and **V4b** in the paper. This is a model predictive control based on the solution of an optimisation problem. The aim of the optimisation is to minimise an objective function that contains the control objectives. The objective function contains a penalty for the control error as well as for the use of the manipulated variable. On the one hand, the quadratic change in the manipulated variable is weighted, on the other hand, the value of the manipulated variable itself is weighted by a function f. This function f is a model of the HVAC system and returns its power demand. In the **V4a** variant, the model of the HVAC system consists of various equations that describe the static system behaviour on the basis of physical laws. Variant **V4b** uses a surrogate HVAC model that corresponds to a linear relationship between the system performance and the generated air mass flow.

## Test this repository online
The scripts `V2`, `V3` and `V3` can be tested via the following link via Binder online without installation
without installation:
<>

## Disclaimer

> The scripts made available demonstrate the implementation of the control schemes `V2.py`, `V3.py` & `V4.py`.
> They are intended to simplify the implementation of the control approaches presented in the associated publication from the user's point of view.
> No claim is made for completeness.
> In order to realise this continuously in practice, all data that is read in locally from the "data" folder, and partly
> also data that is calculated from past measurement data, must be transmitted to the `V2.py`, `V3.py` & `V4.py` scripts in real time.  
> The script `dataPreprocessing.py` is only needed in the demo case shown to preprocess the local data.
> The scripts `WetAirToolBox.py` and `HVAC.py` are also needed in the actual implementation.

`helpers` Folder: Helper functions for `V2.py`, `V3.py` and `V4.py`

`plc` Folder: `plcClientCommunication.py` is the most important script if you want to send set-points from Python to a PLC.
Currently, either Siemens S7 protocol or OPC UA are supported via this script. The files `S7Server.py` and `opcUaServer.py` emulate the behaviour of a PLC and are there
to test the communication from the client script `plcClientCommunication.py` locally.

`environment.yml`: Conda environment to run the python scripts.

## How to cite

If you use this software, please cite the following paper:

> Marcus Vogt, Christian Buchholz, Sebastian Thiede, Christoph Herrmann,
> Energy efficiency of Heating, Ventilation and Air Conditioning systems in production environments through model-predictive control schemes: The case of battery production,
> Journal of Cleaner Production,
> Volume 350,
> 2022,
> https://doi.org/10.1016/j.jclepro.2022.131354

## Installation
Either conda or pip virtual environment can be used for installation. 
On Windows use the following commands for installation of the environment in the command line.

Via conda:
```
1.) conda env create -f environment.yml
2.) conda activate envMPCy3.8
```

Via pip virtual environment:
```
1.) python -m venv envMPCy3.8
2.) .\envMPCy3.8\Scripts\activate
3.) pip install -r requirements.txt
```
