import os
import numpy as np
import matplotlib.pyplot as plt
# os.environ['PROJ_LIB'] = '/Users/micheltossaint/Documents/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap
from numpy.linalg import norm
from math import sin, cos, asin, degrees, radians, log10
import itur
import astropy.units as u
import ast
# Project modules
from constants import K_BOLTZMANN, C_LIGHT
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
        self.include_rain = None
        self.include_gas = None
        self.include_scintillation = None
        self.include_clouds = None
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
        if node.find('IncludeRain') is not None:
            self.include_rain = misc_fn.str2bool(node.find('IncludeRain').text)
        if node.find('IncludeGas') is not None:
            self.include_gas = misc_fn.str2bool(node.find('IncludeGas').text)
        if node.find('IncludeScintillation') is not None:
            self.include_scintillation = misc_fn.str2bool(node.find('IncludeScintillation').text)
        if node.find('IncludeClouds') is not None:
            self.include_clouds = misc_fn.str2bool(node.find('IncludeClouds').text)

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
            if station.station_id == self.station_id:
                self.idx_found_station = idx_station
                break
        self.eirp = 10*log10(self.transmit_power) + self.transmit_gain - self.transmit_losses
        self.metric = np.zeros((sm.num_epoch, 10))
        if self.modulation_type is not None:
            self.cn0_required = misc_fn.comp_cn0_required(self.modulation_type, self.ber, self.data_rate)

    def in_loop(self, sm):
        idx_station = self.idx_found_station
        for idx_sat in sm.stations[idx_station].idx_sat_in_view:
            elevation = sm.gr2sp[idx_station][idx_sat].elevation
            distance = sm.gr2sp[idx_station][idx_sat].distance
            fsl = 20*log10(distance/1000) + 20*log10(self.carrier_frequency/1e9) + 92.45
            # a_g = misc_fn.comp_gas_attenuation(self.carrier_frequency, elevation)  # Fast method but unaccurate <5 deg
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
                include_gas=self.include_gas,
                include_rain=self.include_rain,
                include_scintillation=self.include_scintillation,
                include_clouds=self.include_clouds,
                return_contributions=True)
            if self.transmitter_object == 'satellite':
                ant_temp = 10 + misc_fn.temp_brightness(self.carrier_frequency, elevation)
            else:  # station is the transmitter
                ant_temp = 290
            temp_sys = self.receive_temp + ant_temp
            cn0 = self.eirp - fsl \
                  - a_g.value - a_r.value - a_c.value - a_s.value \
                  +self.receive_gain - self.receive_losses - \
                  K_BOLTZMANN - 10*np.log10(temp_sys)
            self.metric[sm.cnt_epoch,:] = [self.times_f_doy[sm.cnt_epoch], degrees(elevation),
                                           cn0, a_g.value, a_r.value, a_c.value, a_s.value, a_t.value,
                                           fsl, self.cn0_required]

    def after_loop(self, sm):
        self.metric = self.metric[~np.all(self.metric == 0, axis=1)]  # Clean up empty rows
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(left=.1, right=.95, top=0.95, bottom=0.07)
        time_list = self.metric[:, 0]
        plt.plot(time_list, self.metric[:, 1], 'k.', label='Elevation')
        plt.plot(time_list, self.metric[:, 2], 'b.', label='CN0 Computed')
        if self.include_gas:
            plt.plot(time_list, self.metric[:, 3], 'g+', label='Gas Attenuation')
        if self.include_rain:
            plt.plot(time_list, self.metric[:, 4], 'm+', label='Rain Attenuation')
        if self.include_clouds:
            plt.plot(time_list, self.metric[:, 5], 'r+', label='Cloud Attenuation')
        if self.include_scintillation:
            plt.plot(time_list, self.metric[:, 6], 'y+', label='Scintillation Attenuation')
        plt.plot(time_list, self.metric[:, 7], 'r.', label='Total Atmospheric Attenuation')
        plt.plot(time_list, self.metric[:, 8]-100, 'c+', label='Free space loss - 100dB')
        if self.modulation_type is not None:
            plt.plot(time_list, self.metric[:, 9], 'b+', label='CN0 Required')
        plt.xlabel('Day Of Year DOY [-]'); plt.ylabel('Elevation [deg], Power values [dB]')
        plt.legend(); plt.grid()
        plt.savefig('../output/'+self.type+'.png')
        ax.ticklabel_format(useOffset=False, style='plain')
        plt.show()


class AnalysisComSp2SpBudget(AnalysisBase):

    def __init__(self):
        super().__init__()
        self.sat_id1 = None
        self.sat_id1 = None

        self.carrier_frequency = None
        self.transmit_power = None
        self.transmit_losses = None
        self.transmit_gain = None
        self.receive_losses = None
        self.receive_temp = None
        self.modulation_type = None
        self.ber = None
        self.data_rate = None

        self.idx_found_sat1 = None
        self.idx_found_sat2 = None

        self.eirp = None
        self.metric = None
        self.cn0_required = 0

    def read_config(self, node):
        if node.find('SatelliteID1') is not None:
            self.sat_id1 = int(node.find('SatelliteID1').text)
        if node.find('SatelliteID2') is not None:
            self.sat_id2 = int(node.find('SatelliteID2').text)

        if node.find('CarrierFrequency') is not None:
            self.carrier_frequency = float(node.find('CarrierFrequency').text)
        if node.find('TransmitPowerW') is not None:
            self.transmit_power = float(node.find('TransmitPowerW').text)
        if node.find('TransmitLossesdB') is not None:
            self.transmit_losses = float(node.find('TransmitLossesdB').text)
        if node.find('TransmitGaindB') is not None:
            self.transmit_gain = float(node.find('TransmitGaindB').text)

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
        for idx_sat, satellite in enumerate(sm.satellites):
            if satellite.sat_id == self.sat_id1:
                self.idx_found_sat1 = idx_sat
            if satellite.sat_id == self.sat_id2:
                self.idx_found_sat2 = idx_sat
        self.eirp = 10*log10(self.transmit_power) + self.transmit_gain - self.transmit_losses
        self.metric = np.zeros((sm.num_epoch, 5))
        if self.modulation_type is not None:
            self.cn0_required = misc_fn.comp_cn0_required(self.modulation_type, self.ber, self.data_rate)

    def in_loop(self, sm):
        if self.idx_found_sat2 in sm.satellites[self.idx_found_sat1].idx_sat_in_view:
            elevation = sm.sp2sp[self.idx_found_sat1][self.idx_found_sat2].elevation
            distance = sm.sp2sp[self.idx_found_sat1][self.idx_found_sat2].distance
            fsl = 20*log10(distance/1000) + 20*log10(self.carrier_frequency/1e9) + 92.45
            temp_sys = self.receive_temp + 20
            cn0 = self.eirp - fsl \
                  +self.receive_gain - self.receive_losses - \
                  K_BOLTZMANN - 10*np.log10(temp_sys)
            self.metric[sm.cnt_epoch,:] = [self.times_f_doy[sm.cnt_epoch], degrees(elevation),
                                           cn0, fsl, self.cn0_required]

    def after_loop(self, sm):
        self.metric = self.metric[~np.all(self.metric == 0, axis=1)]  # Clean up empty rows
        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=.1, right=.95, top=0.95, bottom=0.07)
        plt.plot(self.metric[:, 0], self.metric[:, 1], 'k.', label='Elevation')
        plt.plot(self.metric[:, 0], self.metric[:, 2], 'b.', label='CN0 Computed')
        plt.plot(self.metric[:, 0], self.metric[:, 3]-100, 'y.', label='Free space loss - 100dB')
        if self.modulation_type is not None:
            plt.plot(self.metric[:, 0], self.metric[:, 4], 'r.', label='CN0 Required')
        plt.xlabel('DOY[-]'); plt.ylabel('Elevation [deg], Power values [dB]')
        plt.legend(); plt.grid()
        plt.savefig('../output/'+self.type+'.png')
        plt.show()


class AnalysisComDoppler(AnalysisBase):

    def __init__(self):
        super().__init__()
        self.station_id = None
        self.carrier_frequency = None
        self.metric = None

    def read_config(self, node):
        if node.find('StationID') is not None:
            self.station_id = int(node.find('StationID').text)
        if node.find('CarrierFrequency') is not None:
            self.carrier_frequency = float(node.find('CarrierFrequency').text)

    def before_loop(self, sm):
        for idx_station, station in enumerate(sm.stations):
            if station.station_id == self.station_id:
                self.idx_found_station = idx_station
                break
        self.metric = np.zeros((sm.num_epoch, 3))

    def in_loop(self, sm):
        idx_station = self.idx_found_station
        for idx_sat in sm.stations[idx_station].idx_sat_in_view:
            velocity = sm.satellites[idx_sat].vel_ecf
            range = sm.gr2sp[idx_station][idx_sat].gr2sp_ecf
            range_rate = np.dot(velocity,range)/norm(range)
            doppler = self.carrier_frequency*range_rate/C_LIGHT
            elevation = sm.gr2sp[idx_station][idx_sat].elevation
            self.metric[sm.cnt_epoch,:] = [self.times_f_doy[sm.cnt_epoch], degrees(elevation), doppler]

    def after_loop(self, sm):
        self.metric = self.metric[~np.all(self.metric == 0, axis=1)]  # Clean up empty rows
        fig, ax1 = plt.subplots(figsize=(10, 6))
        plt.grid()
        ax1.set_ylabel('Doppler [kHz]')
        ax1.yaxis.label.set_color('red')
        ax1.tick_params(axis='y', colors='red')
        ax1.plot(self.metric[:, 0], self.metric[:, 2]/1000, 'r.', label='Doppler [kHz]')
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Elevation [deg]')
        ax2.yaxis.label.set_color('blue')
        ax2.tick_params(axis='y', colors='blue')
        ax2.plot(self.metric[:, 0], self.metric[:, 1], 'b.', label='Elevation [deg]')
        plt.xlabel('DOY[-]');
        plt.legend();
        plt.savefig('../output/'+self.type+'.png')
        plt.show()


class AnalysisComGr2SpBudgetInterference(AnalysisBase):

    def __init__(self):
        super().__init__()
        self.station_id = None
        self.transmitter_object = None
        self.carrier_frequency = None
        self.bandwith = None
        self.transmit_power = None
        self.transmit_losses = None
        self.transmit_gain = None
        self.transmit_gain_manual = None
        self.p_exceed = None
        self.include_rain = None
        self.include_gas = None
        self.include_scintillation = None
        self.include_clouds = None
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
        if node.find('BandWidth') is not None:
            self.bandwidth = float(node.find('BandWidth').text)

        if node.find('TransmitPowerW') is not None:
            self.transmit_power = float(node.find('TransmitPowerW').text)
        if node.find('TransmitLossesdB') is not None:
            self.transmit_losses = float(node.find('TransmitLossesdB').text)
        if node.find('TransmitGaindB') is not None:
            self.transmit_gain = float(node.find('TransmitGaindB').text)
        if node.find('TransmitGainManualdB') is not None:
            self.transmit_gain_manual = np.array(list(ast.literal_eval(node.find('TransmitGainManualdB').text)))
        if node.find('TransmitAntennaDiameter') is not None:
            self.transmit_ant_dia = float(node.find('TransmitAntennaDiameter').text)

        if node.find('PExceedPerc') is not None:
            self.p_exceed = float(node.find('PExceedPerc').text)
        if node.find('IncludeRain') is not None:
            self.include_rain = misc_fn.str2bool(node.find('IncludeRain').text)
        if node.find('IncludeGas') is not None:
            self.include_gas = misc_fn.str2bool(node.find('IncludeGas').text)
        if node.find('IncludeScintillation') is not None:
            self.include_scintillation = misc_fn.str2bool(node.find('IncludeScintillation').text)
        if node.find('IncludeClouds') is not None:
            self.include_clouds = misc_fn.str2bool(node.find('IncludeClouds').text)

        if node.find('ReceiveGaindB') is not None:
            self.receive_gain = float(node.find('ReceiveGaindB').text)
        if node.find('ReceiveAntennaDiameter') is not None:
            self.receive_ant_dia = float(node.find('ReceiveAntennaDiameter').text)
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
            if station.station_id == self.station_id:
                self.idx_found_station = idx_station
                break
        self.metric = np.zeros((sm.num_epoch, 12))
        if self.modulation_type is not None:
            self.cn0_required = misc_fn.comp_cn0_required(self.modulation_type, self.ber, self.data_rate)

    def in_loop(self, sm):
        idx_station = self.idx_found_station
        if len(sm.stations[idx_station].idx_sat_in_view) > 1:

            elevation = sm.gr2sp[idx_station][0].elevation
            distance = sm.gr2sp[idx_station][0].distance

            if self.transmit_gain_manual is not None:
                self.transmit_gain = misc_fn.dish_pattern_manual(self.transmit_gain_manual,0)

            # Nominal satellite C/N0
            self.eirp = 10 * log10(self.transmit_power) + self.transmit_gain - self.transmit_losses
            fsl = 20 * log10(distance / 1000) + 20 * log10(self.carrier_frequency / 1e9) + 92.45
            a_g, a_c, a_r, a_s, a_t = itur.atmospheric_attenuation_slant_path(
                degrees(sm.stations[idx_station].lla[0]),
                degrees(sm.stations[idx_station].lla[1]),
                self.carrier_frequency / 1e9 * itur.u.GHz,
                degrees(elevation), self.p_exceed,
                D=1.0 * itur.u.m,
                include_gas=self.include_gas,
                include_rain=self.include_rain,
                include_scintillation=self.include_scintillation,
                include_clouds=self.include_clouds,
                return_contributions=True)
            if self.transmitter_object == 'satellite':
                ant_temp = 10 + misc_fn.temp_brightness(self.carrier_frequency, elevation)
            else:  # station is the transmitter
                ant_temp = 290
            temp_sys = self.receive_temp + ant_temp
            cn0 = self.eirp - fsl \
                  - a_g.value - a_r.value - a_c.value - a_s.value \
                  + self.receive_gain - self.receive_losses - \
                  K_BOLTZMANN - 10*np.log10(temp_sys)

            # Interference computation from second satellite
            u = sm.satellites[0].pos_ecf - sm.stations[0].pos_ecf
            v = sm.satellites[1].pos_ecf - sm.stations[0].pos_ecf
            off_boresight_angle = misc_fn.angle_two_vectors(u,v,np.linalg.norm(u),np.linalg.norm(v))
            C = self.eirp - fsl \
                  - a_g.value - a_r.value - a_c.value - a_s.value \
                  + self.receive_gain - self.receive_losses
            C_fact = np.power(10, C/10) # all in factors since C0/(N0+I0)
            N0 = K_BOLTZMANN + 10*np.log10(temp_sys)
            N0_fact = np.power(10, N0/10)
            if self.transmit_gain_manual is not None:
                tx_gain = misc_fn.dish_pattern_manual(self.transmit_gain_manual,off_boresight_angle)
            else:
                tx_gain = misc_fn.dish_pattern(self.carrier_frequency,
                                           self.transmit_ant_dia,self.transmit_gain,off_boresight_angle)
            rx_gain = misc_fn.dish_pattern(self.carrier_frequency,
                                           self.receive_ant_dia,self.receive_gain,off_boresight_angle)
            I = 10 * log10(self.transmit_power) + tx_gain - self.transmit_losses + \
                - fsl - a_g.value - a_r.value - a_c.value - a_s.value + \
                + rx_gain - self.receive_losses
            I0 = I - 10*np.log10(self.bandwidth)
            I0_fact = np.power(10, I0/10)
            cni0 = 10*np.log10(C_fact / (N0_fact+I0_fact))

            self.metric[sm.cnt_epoch,:] = [self.times_f_doy[sm.cnt_epoch], degrees(elevation),
                                           cn0, cni0, a_g.value, a_r.value, a_c.value, a_s.value, a_t.value,
                                           degrees(off_boresight_angle), tx_gain, rx_gain]

    def after_loop(self, sm):
        self.metric = self.metric[~np.all(self.metric == 0, axis=1)]  # Clean up empty rows
        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=.1, right=.95, top=0.95, bottom=0.07)
        time_list = self.metric[:, 0]
        plt.plot(time_list, self.metric[:, 1], label='Elevation leading satellite')
        plt.plot(time_list, self.metric[:, 2]-100, label='CN0 Nominal - 100')
        plt.plot(time_list, self.metric[:, 3]-100, label='CN0 with Interference - 100')
        plt.plot(time_list, self.metric[:, 8], label='Total Atmospheric Attenuation')
        plt.plot(time_list, self.metric[:, 9], label='Off boresight angle interferer')
        plt.plot(time_list, self.metric[:, 10], label='Tx Gain Interferer')
        plt.plot(time_list, self.metric[:, 11], label='Rx Gain Interferer')
        plt.plot(time_list, np.ones(len(time_list))*self.transmit_gain, label='Tx Gain Nom')
        plt.plot(time_list, np.ones(len(time_list)) *self.receive_gain, label='Rx Gain Nom')
        plt.gca().ticklabel_format(axis='both', style='plain', useOffset=False)
        plt.xlabel('Day Of Year DOY [-]'); plt.ylabel('Angles [deg], Power values [dB]')
        plt.legend(); plt.grid()
        plt.savefig('../output/'+self.type+'.png')
        plt.show()

        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=.1, right=.95, top=0.95, bottom=0.07)
        plt.plot(time_list, self.metric[:, 2]-self.metric[:, 3], label='CN0 drop [dB]')
        plt.gca().ticklabel_format(axis='x', style='plain', useOffset=False)
        plt.gca().ticklabel_format(axis='y', style='sci', useOffset=False)
        plt.xlabel('Day Of Year DOY [-]'); plt.ylabel('Power values [dB]')
        plt.legend(); plt.grid()
        plt.show()
