"""
-------------------------------------------------------------------------------
Name:        plcCommunication
Purpose:     This file connects, reads and writes to a PLC for real-time control.

Author:      Marcus Vogt

Created:     28.01.2022
Copyright:   Chair of Sustainable Manufacturing and Life Cycle Engineering, Institute of Machine Tools and Production Technology, Technische Universität Braunschweig, Langer Kamp 19b, 38106 Braunschweig, Germany
Licence:     MIT (see License)
-------------------------------------------------------------------------------
"""

import snap7
from snap7 import util
import re
import opcua

def connect2S7(IPString: str) -> (bool, snap7.client.Client()):
    """
    This function connects to the PLC via Siemens S7 protocol
    :param IPString: str: IP address of PLC
    :return bool and s7 client: returns true and s7 client if connected successfully to PLC
    """
    RACK = 0
    SLOT = 2
    client = snap7.client.Client()
    client.connect(IPString, RACK, SLOT)
    isConnected = client.get_connected()
    return isConnected, client

def readAddressS7(client: snap7.client.Client(), logicalAddress: str) -> float:
    """
    This function reads a value from the PLC for a given logical address.
    :param client: snap7.client.Client() client
    :param logicalAddress: str: Logical addresses of datapoint e.g. %DB500.DBD138.
    :return float: returns real from logical address
    """
    # extract db and dbd from logicalAddress
    # db: int: db number to read from
    # dbd: int: start address to read from. The end address is then calculated by added 4 (because a real has 4 bytes)
    db, dbd = [int(s) for s in re.findall("[0-9]+", logicalAddress)]
    nodeId = client.db_read(db, dbd, 4) #read 4 bytes (real) at address
    t = util.get_real(nodeId, 0)
    return t

def clamp(val: float, minVal: float, maxVal: float) -> float:
    """Small helper function to clamp value between min and max.
    :param val: float: input value
    :param minVal: float: minimum value
    :param maxVal: float: maximum value
    :return float: clamped input value
    """
    if val < minVal:
        return minVal
    elif val > maxVal:
        return maxVal
    else:
        return val

def writeValueCheck(value: float, min: float, max: float) -> (bool, float):
    """
    This function checks if the value to be written is in the correct ranges.
    Currently only volume flow (m^3/h) and the dew point temperature can be written.
    :param value: float: Value to write a the logical address
    :param min: float: minimum value
    :param max: float: maximum value
    :return bool and float: returns true if everything is correct and new value after the check
    """
    isNumber = isinstance(value, float) or isinstance(value, int)
    if not isNumber:
        return isNumber, value
    else:
        newValue = clamp(val=value, minVal=min, maxVal=max)
        return isNumber, newValue

def writeValue2AddressS7(client: snap7.client.Client(), logicalAddress: str,
                         value: float, min: float, max: float) -> (bool, float):
    """
    This function writes a given value to the given logical address of the PLC.
    :param client: snap7.client.Client() client
    :param logicalAddress: str: Logical addresses of datapoint e.g. %DB500.DBD138.
    :param value: float: Value to write a the logical address
    :param min: float: minimum value
    :param max: float: maximum value
    :return bool, float: returns true if value is float and the written value to the PLC.
    """
    db, dbd = [int(s) for s in re.findall("[0-9]+", logicalAddress)]
    nodeId = client.db_read(db, dbd, 4) #read 4 bytes (real) at address
    #nodeId = client.read_area(snap7.types.Areas.DB, db, dbd, 4) # equivalent to above
    isNumber, newValue = writeValueCheck(value=value, min=min, max=max)
    if isNumber:
        setBuffer = util.set_real(nodeId, 0, newValue)
        client.db_write(db, dbd, setBuffer)
        #client.write_area(snap7.types.Areas.DB, db, dbd, nodeId) # equivalent to above
        t = util.get_real(nodeId, 0)
        return isNumber, t
    else:
        print(f"The value {value} is not a float or int and has not been written to the PLC!")
        return isNumber, value

def connect2OPCUA(IPString: str) -> opcua.Client:
    """
    This function connects to the PLC via OPC UA
    :param IPString: str: IP address of PLC
    :return opc ua client: returns opc ua client if connected successfully to PLC
    """
    client = opcua.Client(f"opc.tcp://{IPString}:4840") #(f"opc.tcp://{IPString}:4840/SIMATIC.S7-1500.OPC-UA.Application:PLC_1")
    client.connect()
    return client

def readAddressOPCUA(client: opcua.Client, logicalAddress: str) -> float:
    """
    This function reads a value from the PLC for a given logical address.
    :param client: opcua.Client client
    :param logicalAddress: str: Logical addresses of datapoint e.g. "ns=3;s=\"RLT01\".\"Raumtemperatur1_IST\"".
    :return float: returns real from logical address
    """
    nodeId = client.get_node(logicalAddress)
    t = nodeId.get_value()
    return t

def writeValue2OPCUA(client: opcua.Client, logicalAddress: str, value: float) -> float:
    """
    This function reads a value from the PLC for a given logical address.
    :param client: opcua.Client client
    :param logicalAddress: str: Logical addresses of datapoint e.g. "ns=3;s=\"RLT01\".\"Raumtemperatur1_IST\"".
    :param value: float: Value to write a the logical address
    :return float: returns float of the written value to the PLC.
    """
    nodeId = client.get_node(logicalAddress)
    nodeId.set_value(value)
    t = nodeId.get_value()
    return t

if __name__ == '__main__':
    # Todo: To test the server locally first start the S7 server (S7Server.py) or OPC UA (opcUaServer.py)
    #  to test the communication with this client script
    testS7Connection = True
    testOPCUAConnection = True
    ### S7 connect read/write ###
    if testS7Connection:
        # Todo: Replace IPStrings and logicalAddresses with addresses for your case
        isConnected, clientS7 = connect2S7(IPString="127.0.0.1")
        print(f"Established connection to PLC: {isConnected}")
        logicalAddressFan = "%DB1.DBD10"
        maxFan = 30000
        minFan = 3000
        volumeFlowValue = 15000
        logicalAddressDewPoint = "%DB1.DBD6"
        maxDewPoint = -30
        minDewPoint = -70
        dewPointValue = -45
        t1 = readAddressS7(client=clientS7, logicalAddress=logicalAddressFan)
        print(f"Process air set point is: {t1}")
        t2 = readAddressS7(client=clientS7, logicalAddress=logicalAddressDewPoint)
        print(f"Dew point temperature set point is: {t2}")
        isNmb, t3 = writeValue2AddressS7(client=clientS7, logicalAddress=logicalAddressFan,
                                  value=volumeFlowValue, min=minFan, max=maxFan)
        if isNmb:
            print(f"Wrote {volumeFlowValue} to process air set point "
                  f"and now is: {t3}")
        isNumber1, t4 = writeValue2AddressS7(client=clientS7, logicalAddress=logicalAddressDewPoint,
                                            value=dewPointValue, min=minDewPoint, max=maxDewPoint)
        if isNumber1:
            print(f"Wrote {dewPointValue} to dew point air set point "
                  f"and now is: {t4}")
        clientS7.disconnect()

    ### OPC UA connect read/write ###
    if testOPCUAConnection:
        clientOpcUa = connect2OPCUA(IPString="127.0.0.1")
        logicalAddressOPCUA = "ns=2;i=2" #"ns=3;s=\"RLT01\".\"Raumtemperatur1_IST\""
        t5 = readAddressOPCUA(client=clientOpcUa, logicalAddress=logicalAddressOPCUA)
        print(f"Room temperature of RLT01 is: {t5}")
        temperatureValue = 25
        tempVal = writeValue2OPCUA(client=clientOpcUa, logicalAddress=logicalAddressOPCUA, value=temperatureValue)
        print(f"Wrote {temperatureValue} to server and got {tempVal} back")
        clientOpcUa.disconnect()





