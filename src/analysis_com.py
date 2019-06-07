import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
os.environ['PROJ_LIB'] = '/Users/micheltossaint/Documents/anaconda3/lib/python3.6/site-packages/pyproj/data'
from mpl_toolkits.basemap import Basemap
from numpy.linalg import norm
from math import sin, cos, asin, degrees, radians, log10
import itur
import astropy.units as u
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
        self.p_exceed = None
        self.rain_height = None
        self.rain_rate = None
        self.receive_gain = None
        self.receive_losses = None
        self.receive_temp = None
        self.modulation_type = None
        self.ber = None
        self.data_rate = None

        self.idx_found_station = None
        self.eirp = None

        self.metric = None
        self.cn0_required = 0

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

        if node.find('PExceedPerc') is not None:
            self.p_exceed = float(node.find('PExceedPerc').text)
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

        if node.find('ModulationType') is not None:
            self.modulation_type = node.find('ModulationType').text
        if node.find('BitErrorRate') is not None:
            self.ber = float(node.find('BitErrorRate').text)
        if node.find('DataRateBitPerSec') is not None:
            self.data_rate = float(node.find('DataRateBitPerSec').text)

    def before_loop(self, sm):
        for idx_station, station in enumerate(sm.stations):
            if station.constellation_id == self.station_id and \
                    station.constellation_id == self.station_id:
                self.idx_found_station = idx_station
                break
        self.eirp = 10*log10(self.transmit_power) + self.transmit_gain - self.transmit_losses
        self.metric = np.zeros((sm.num_epoch, 7))
        if self.modulation_type is not None:
            self.cn0_required = misc_fn.comp_cn0_required(self.modulation_type, self.ber, self.data_rate)

    def in_loop(self, sm):
        idx_station = self.idx_found_station
        for idx_sat in sm.stations[idx_station].idx_sat_in_view:
            elevation = sm.gr2sp[idx_station][idx_sat].elevation
            distance = sm.gr2sp[idx_station][idx_sat].distance
            fsl = 20*log10(distance/1000) + 20*log10(self.carrier_frequency/1e9) + 92.45
            # a_g = misc_fn.comp_gas_attenuation(self.carrier_frequency, elevation)  # Fast method but unaccurate <5 deg
            if self.rain_rate is not None:
                # fast method of the rain model
                # a_r = misc_fn.comp_rain_attenuation(self.carrier_frequency, elevation,
                #                                          sm.stations[idx_station].lla[0], sm.stations[idx_station].lla[2],
                #                                          self.rain_p_exceed, self.rain_rate, self.rain_height)
                a_g, a_c, a_r, a_s, a_t = itur.atmospheric_attenuation_slant_path(
                    degrees(sm.stations[idx_station].lla[0]),
                    degrees(sm.stations[idx_station].lla[1]),
                    self.carrier_frequency / 1e9 * itur.u.GHz,
                    degrees(elevation), self.p_exceed,
                    D=1.0 * itur.u.m,
                    include_scintillation=False,
                    include_clouds=False,
                    return_contributions=True)
            else:
                a_g, a_c, a_r, a_s, a_t = itur.atmospheric_attenuation_slant_path(
                    degrees(sm.stations[idx_station].lla[0]),
                    degrees(sm.stations[idx_station].lla[1]),
                    self.carrier_frequency / 1e9 * itur.u.GHz,
                    degrees(elevation), self.p_exceed,
                    D=1.0 * itur.u.m,
                    include_rain=False,
                    include_scintillation=False,
                    include_clouds=False,
                    return_contributions=True)
            if self.transmitter_object == 'satellite':
                ant_temp = 10 + misc_fn.temp_brightness(self.carrier_frequency, elevation)
            else:  # station is the transmitter
                ant_temp = 290
            temp_sys = self.receive_temp + ant_temp
            cn0 = self.eirp - fsl - a_g.value - a_r.value +self.receive_gain - self.receive_losses - \
                  K_BOLTZMANN - 10*np.log10(temp_sys)
            self.metric[sm.cnt_epoch,:] = [self.times_f_doy[sm.cnt_epoch], degrees(elevation),
                                           cn0, a_g.value, a_r.value, fsl, self.cn0_required]

    def after_loop(self, sm):
        self.metric = self.metric[~np.all(self.metric == 0, axis=1)]  # Clean up empty rows
        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=.1, right=.95, top=0.95, bottom=0.07)
        plt.plot(self.metric[:, 0], self.metric[:, 1], 'k.', label='Elevation')
        plt.plot(self.metric[:, 0], self.metric[:, 2], 'b.', label='CN0 Computed')
        plt.plot(self.metric[:, 0], self.metric[:, 3], 'g.', label='Gas Attenuation')
        if self.rain_rate is not None:
            plt.plot(self.metric[:, 0], self.metric[:, 4], 'm.', label='Rain Attenuation')
        plt.plot(self.metric[:, 0], self.metric[:, 5]-100, 'y.', label='Free space loss - 100dB')
        if self.modulation_type is not None:
            plt.plot(self.metric[:, 0], self.metric[:, 6], 'r.', label='CN0 Required')
        plt.xlabel('DOY[-]'); plt.ylabel('Elevation [deg], Power values [dB]')
        plt.legend(); plt.grid()
        plt.savefig('../output/'+self.type+'.png')
        plt.show()


