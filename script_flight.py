import logging
import time
from threading import Timer

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

import ast
import sys
from NatNetClient import NatNetClient

import numpy as np
import math

from tools.kine_UAV import KineUAV
from tools.rotation_matrix import RotationMatrix

from control import lqr


uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

pos_ot = np.zeros((1,3))
phi, theta, psi, vex, vey, vez = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

kine_UAV = KineUAV()
rm = RotationMatrix()

Q = np.diag([0,0,0,0,0,0,0,0,1,1,1])
R = np.diag([0.01,0.01,0.01])
T_trim = 9.81
A_aug, B_aug = kine_UAV.augsys_linear()
K, _, _ = lqr(A_aug,B_aug,Q,R)

class LoggingDrone:
    """
    Logging the data acquired from the drone when it is connected
    Modified from the LoggingExample of the Crazyflie Demo
    """

    def __init__(self, link_uri):
        """ 
        Initialize and run the logging with the specified link_uri 
        """
        self._cf = Crazyflie(rw_cache='./cache')

        # Connect some callbacks from the Crazyflie API
        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)

        print('Connecting to %s' % link_uri)

        # Try to connect to the Crazyflie
        self._cf.open_link(link_uri)

        # Variable used to keep main loop occupied until disconnect
        self.is_connected = True

    def _connected(self, link_uri):
        """ 
        This callback is called from the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded.
        """
        print('Connected to %s' % link_uri)

        self._cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        self._cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(2)

        # The definition of the logconfig can be made before connecting
        self._lg_stab = LogConfig(name='Stabilizer', period_in_ms=10)
        self._lg_stab.add_variable('stabilizer.roll', 'float')
        self._lg_stab.add_variable('stabilizer.pitch', 'float')
        self._lg_stab.add_variable('stabilizer.yaw', 'float')
        self._lg_stab.add_variable('stateEstimate.vx', 'float')
        self._lg_stab.add_variable('stateEstimate.vy', 'float')
        self._lg_stab.add_variable('stateEstimate.vz', 'float')
        self._lg_stab.add_variable('acc.z', 'float')
        # The fetch-as argument can be set to FP16 to save space in the log packet
        self._lg_stab.add_variable('pm.vbat', 'FP16')

        # Adding the configuration cannot be done until a Crazyflie is
        # connected, since we need to check that the variables we
        # would like to log are in the TOC.
        try:
            self._cf.log.add_config(self._lg_stab)
            # This callback will receive the data
            self._lg_stab.data_received_cb.add_callback(self._stab_log_data)
            # This callback will be called on errors
            self._lg_stab.error_cb.add_callback(self._stab_log_error)
            # Start the logging
            self._lg_stab.start()
        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Stabilizer log config, bad configuration.')

        # Start a timer to disconnect in connect range s
        con_ran = 300 # adjust it
        t = Timer(con_ran, self._cf.close_link) 
        t.start()

    def _stab_log_error(self, logconf, msg):
        """
        Callback from the log API when an error occurs
        """
        print('Error when logging %s: %s' % (logconf.name, msg))

    def _stab_log_data(self, timestamp, data, logconf):
        """
        Callback from a the log API when data arrives
        """
        global phi, theta, psi, vex, vey, vez
        phi = data['stabilizer.roll']
        theta = -data['stabilizer.pitch'] # make it right-hand
        psi = data['stabilizer.yaw']
        vbx = data['stateEstimate.vx']
        vby = data['stateEstimate.vy']
        vbz = data['stateEstimate.vz']

        v_body = np.array([vbx,vby,vbz])
        v_earth = np.matmul(rm.b2e_0psi(phi, theta), v_body)
        vex = v_earth[0]
        vey = v_earth[1]
        vez = v_earth[2]

    def _connection_failed(self, link_uri, msg):
        """
        Callback when connection initial connection fails 
        (i.e no Crazyflie at the specified address)
        """
        print('Connection to %s failed: %s' % (link_uri, msg))
        self.is_connected = False

    def _connection_lost(self, link_uri, msg):
        """
        Callback when disconnected after a connection has been made 
        (i.e Crazyflie moves out of range)
        """
        print('Connection to %s lost: %s' % (link_uri, msg))

    def _disconnected(self, link_uri):
        """
        Callback when the Crazyflie is disconnected (called in all cases)
        """
        print('Disconnected from %s' % link_uri)
        self.is_connected = False


def receiveNewFrame(frameNumber, markerSetCount, unlabeledMarkersCount, rigidBodyCount, skeletonCount,
                    labeledMarkerCount, latency, timecode, timecodeSub, timestamp, isRecording, trackedModelsChanged):
    """
    A callback function that gets connected to the NatNet client and called once per mocap frame.
    """
    pass


def receiveRigidBodyFrame(id, position, rotation):
    """
    A callback function that gets connected to the NatNet client. 
    It is called once per rigid body per frame
    """
    global pos_ot
    if id==1:
        # print(position)
        # print(rotation)

        # The coordinate of OptiTrack and CrazyFlie are different 
        # opti_track z x y
        # crazyflie x y z
        pos_ot[0,0] = position[2]
        pos_ot[0,1] = position[0]
        pos_ot[0,2] = position[1]
    

if __name__ == '__main__':

    streamingClient = NatNetClient() # Create a new NatNet client
    streamingClient.newFrameListener = receiveNewFrame
    streamingClient.rigidBodyListener = receiveRigidBodyFrame
    streamingClient.run() # Run perpetually on a separate thread.

 # Initialize the low-level drivers
    cflib.crtp.init_drivers()
    ld = LoggingDrone(uri)

    file = open('./dat00.csv', 'w')
    file.write('timeStamp, OTx, OTy, OTz, vex, vey, vez, roll, pitch, yaw\n')

    # The Crazyflie lib doesn't contain anything to keep the application alive,
    # so this is where your application should do something. In our case we
    # are just waiting until we are disconnected.
    time.sleep(5)
    time0 = round(time.time()*1000)%1000000
    while ld.is_connected:
        try:
            time.sleep(0.01)
            time_now = round(time.time()*1000)%1000000-time0 # timestamp (ms)
            file.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(time_now, pos_ot[0,0], pos_ot[0,1], pos_ot[0,2], vex, vey, vez, phi, theta, psi))
            # ld._cf.commander.send_setpoint(roll, pitch, yaw, thrust) # thrust 0-FFFF 
            # print('vex',vex, 'theta', theta)
            
            for y in range(10):
                ld._cf.commander.send_hover_setpoint(0, 0, 0, y / 25)
                time.sleep(0.1)

            for _ in range(20):
                ld._cf.commander.send_hover_setpoint(0, 0, 0, 0.4)
                time.sleep(0.1)

            for _ in range(50):
                ld._cf.commander.send_hover_setpoint(0.5, 0, 36 * 2, 0.4)
                time.sleep(0.1)

            for _ in range(50):
                ld._cf.commander.send_hover_setpoint(0.5, 0, -36 * 2, 0.4)
                time.sleep(0.1)

            for _ in range(20):
                ld._cf.commander.send_hover_setpoint(0, 0, 0, 0.4)
                time.sleep(0.1)

            for y in range(10):
                ld._cf.commander.send_hover_setpoint(0, 0, 0, (10 - y) / 25)
                time.sleep(0.1)

            ld._cf.commander.send_stop_setpoint()

        except KeyboardInterrupt:
            print("stop")
            ## landing procedure
            for i in range(50):
                ld[i]._cf.commander.send_hover_setpoint(0, 0, 0, 0.6-i*0.01)
                time.sleep(0.1)
            raise
