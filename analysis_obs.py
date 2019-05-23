import os
import numpy as np
import pandas as pd
from math import degrees, radians
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
os.environ['PROJ_LIB'] = '/Users/micheltossaint/Documents/anaconda3/lib/python3.6/site-packages/pyproj/data'
from mpl_toolkits.basemap import Basemap
# Project modules
import misc_fn
from constants import PI
from analysis import AnalysisBase


class AnalysisCovDepthOfCoverage(AnalysisBase):

    def __init__(self):
        super().__init__()

    def read_config(self, node):
        pass

    def before_loop(self, sm):
        for satellite in sm.satellites:
            satellite.metric = np.zeros((sm.num_epoch, 3))

    def in_loop(self, sm):
        for satellite in sm.satellites:
            satellite.det_lla()
            satellite.metric[sm.cnt_epoch, 0] = degrees(satellite.lla[0])
            satellite.metric[sm.cnt_epoch, 1] = degrees(satellite.lla[1])
            satellite.metric[sm.cnt_epoch, 2] = len(satellite.idx_stat_in_view)

    def after_loop(self, sm):
        fig = plt.figure(figsize=(10, 4))
        cm = plt.cm.get_cmap('RdYlBu')
        for satellite in sm.satellites:
            sc = plt.scatter(satellite.metric[:, 1], satellite.metric[:, 0], cmap=cm,
                        c=satellite.metric[:, 2], vmin=0, vmax=len(sm.stations))
        plt.colorbar(sc, shrink=0.85)
        m = Basemap(projection='cyl', lon_0=0)
        m.drawparallels(np.arange(-90., 99., 30.), labels=[True, False, False, True])
        m.drawmeridians(np.arange(-180., 180., 60.), labels=[True, False, False, True])
        for station in sm.stations:
            plt.plot(degrees(station.lla[1]), degrees(station.lla[0]), 'r^')
        m.drawcoastlines()
        plt.subplots_adjust(left=.12, right=.999, top=0.999, bottom=0.01)
        plt.text(50, 80, 'Red triangles: station locations')
        plt.savefig('output/'+self.type+'.png')
        plt.show()
