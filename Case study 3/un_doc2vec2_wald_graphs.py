# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
#%matplotlib qt
plt.style.use("ggplot")

myPath = "C:\\Users\\S\\Documents\\Uni\\UNspeeches\\"

ts = pd.read_csv(myPath + "doc2vec2\\"  + "wald_ts_ch6.csv")

fig_den = plt.figure()
ax_den = fig_den.add_subplot(111)
ax_den.plot(ts.year,ts.waldDEN1, label = "AR(1)")
ax_den.plot(ts.year,ts.waldDEN2, label = "AR(2)")
yy = np.repeat(1985, int(ax_den.get_ylim()[1]))
ax_den.plot(yy, range(0,int(ax_den.get_ylim()[1])), linestyle = "--", 
                      linewidth=2, label = "Lowest p-value", color = "#B03A2E")
ax_den.set_ylabel("Wald test statistic", fontsize = 16)
ax_den.set_xlabel("Year", fontsize = 16)
ax_den.legend()

fig_den = plt.figure()
ax_den = fig_den.add_subplot(111)
ax_den.plot(ts.year,ts.waldUSA1, label = "AR(1)")
ax_den.plot(ts.year,ts.waldUSA2, label = "AR(2)")
yy = np.repeat(1989, int(ax_den.get_ylim()[1]))
ax_den.plot(yy, range(0,int(ax_den.get_ylim()[1])), linestyle = "--", 
                      linewidth=2, label = "Lowest p-value", color = "#B03A2E")
ax_den.set_ylabel("Wald test statistic", fontsize = 16)
ax_den.set_xlabel("Year", fontsize = 16)
ax_den.legend()

fig_den = plt.figure()
ax_den = fig_den.add_subplot(111)
ax_den.plot(ts.year,ts.waldRUS1, label = "AR(1)")
ax_den.plot(ts.year,ts.waldRUS2, label = "AR(2)")
yy = np.repeat(1992, int(ax_den.get_ylim()[1]))
ax_den.plot(yy, range(0,int(ax_den.get_ylim()[1])), linestyle = "--", 
                      linewidth=2, label = "Lowest p-value", color = "#B03A2E")
ax_den.set_ylabel("Wald test statistic", fontsize = 16)
ax_den.set_xlabel("Year", fontsize = 16)
ax_den.legend()