import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
os.environ['PROJ_LIB'] = '/Users/micheltossaint/Documents/anaconda3/lib/python3.6/site-packages/pyproj/data'
from mpl_toolkits.basemap import Basemap
from numpy.linalg import norm
from math import sin, cos, asin, degrees, radians, log10
# Project modules
from constants import K_BOLTZMANN
from analysis import AnalysisBase
import misc_fn
import logging_svs as ls


class AnalysisComGr2SpBudget(AnalysisBase):

    def __init__(self):
        super().__init__()
        self.station_id = None
        self.transmitter_object = None
        self.carrier_frequency = None
        self.transmit_power = None
        self.transmit_losses = None
        self.transmit_gain = None
        self.rain_p_exceed = None
        self.rain_height = None
        self.rain_rate = None
        self.receive_gain = None
        self.receive_losses = None
        self.receive_temp = None

        self.idx_found_station = None
        self.eirp = None

    def read_config(self, node):
        if node.find('GroundStationID') is not None:
            self.station_id = int(node.find('GroundStationID').text)
        if node.find('TransmitterObject') is not None:
            self.transmitter_object = node.find('TransmitterObject').text.lower()
        if node.find('CarrierFrequency') is not None:
            self.carrier_frequency = float(node.find('CarrierFrequency').text)

        if node.find('TransmitPowerW') is not None:
            self.transmit_power = float(node.find('TransmitPowerW').text)
        if node.find('TransmitLossesdB') is not None:
            self.transmit_losses = float(node.find('TransmitLossesdB').text)
        if node.find('TransmitGaindB') is not None:
            self.transmit_gain = float(node.find('TransmitGaindB').text)

        if node.find('RainPExceedPerc') is not None:
            self.rain_p_exceed = float(node.find('RainPExceedPerc').text)
        if node.find('RainHeight') is not None:
            self.rain_height = float(node.find('RainHeight').text)
        if node.find('RainfallRate') is not None:
            self.rain_rate = float(node.find('RainfallRate').text)

        if node.find('ReceiveGaindB') is not None:
            self.receive_gain = float(node.find('ReceiveGaindB').text)
        if node.find('ReceiveLossesdB') is not None:
            self.receive_losses = float(node.find('ReceiveLossesdB').text)
        if node.find('ReceiveTempK') is not None:
            self.receive_temp = float(node.find('ReceiveTempK').text)

    def before_loop(self, sm):
        for idx_station, station in enumerate(sm.stations):
            if station.constellation_id == self.station_id and \
                    station.constellation_id == self.station_id:
                self.idx_found_station = idx_station
                break
        self.eirp = 10*log10(self.transmit_power) + self.transmit_gain - self.transmit_losses


    def in_loop(self, sm):
        idx_station = self.idx_found_station

        # TODO Loop over satellites in view instead
        for idx_sat, satellite in enumerate(sm.satellites):
            elevation = sm.gr2sp[idx_station][idx_sat].elevation
            distance = sm.gr2sp[idx_station][idx_sat].distance
            fsl = 20*log10(distance/1000) + 20*log10(self.carrier_frequency/1e9)
            att_gas = misc_fn.comp_gas_attenuation(self.carrier_frequency, elevation)
            att_rain = misc_fn.comp_rain_attenuation(self.carrier_frequency, elevation, self.rain_p_exceed,
                                                     sm.stations[idx_station].lla[0], sm.stations[idx_station].lla[2],
                                                     self.rain_rate)
            if self.transmitter_object == 'satellite':
                ant_temp = 10 + misc_fn.temp_brightness(self.carrier_frequency, elevation)
            else:  # station is the transmitter
                ant_temp = 290
            temp_sys = self.receive_temp + ant_temp
            cn0 = self.eirp - fsl - att_gas - att_rain - K_BOLTZMANN - 10*np.log10(temp_sys)

    def after_loop(self, sm):
        pass

