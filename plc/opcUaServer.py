import sys
sys.path.insert(0, "..")
from random import randint
import datetime
import time
from opcua import ua, Server

if __name__ == "__main__":

    # setup our server
    server = Server()
    url = "opc.tcp://0.0.0.0:4840"
    server.set_endpoint(url)

    # setup our own namespace, not really necessary but should as spec
    uri = "OPCUA_SIMULATION_SERVER"
    addspace = server.register_namespace(uri)

    # get Objects node, this is where we should put our nodes
    node = server.get_objects_node()

    Param = node.add_object(addspace, "Parameters")

    Temp = Param.add_variable(addspace, "Temperature", 15)
    Press = Param.add_variable(addspace, "Pressure", 0)
    Time = Param.add_variable(addspace, "Time", 0)

    Temp.set_writable()
    Press.set_writable()
    Time.set_writable()

    # starting!
    server.start()
    print("Server started at {}".format(url))

    try:
        while True:
            #Temperature = randint(10, 50)
            T = Temp.get_value()
            Pressure = randint(200, 999)
            TIME = datetime.datetime.now()
            print(T, Pressure, TIME)
            #Temp.set_value(Temperature)
            Press.set_value(Pressure)
            Time.set_value(TIME)
            time.sleep(10)
    finally:
        # close connection, remove subcsriptions, etc
        server.stop()