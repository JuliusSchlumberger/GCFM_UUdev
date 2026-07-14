# -*- coding: utf-8 -*-
"""
Authors: S.L. Verhoeve & J.R.F.W. Leuven
Contact: j.r.f.w.leuven@uu.nl
Last edit: November, 2018

This tool is described in the following corresponding article. If you use the tool, please cite the article as:  
    Leuven, J.R.F.W., Verhoeve, S.L., van Dijk, W.M., Selakovic, S. and Kleinhans, M.G. (2018 or 2019). Empirical assessment tool for bathymetry, flow velocity and salinity in estuaries based on tidal
    amplitude and remotely-sensed imagery. Remote Sensing, Special Issue "Remote Sensing of Flow Velocity, Channel Bathymetry, and River Discharge". 
    
Please note that the most recent version of the tool and instructions can be found at:
    https://github.com/JasperLeuven/EstuarineMorphologyEstimator/     
 
"""
#%%
from scipy.signal import savgol_filter
import csv
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec
import time
from bisect import bisect_left
import pandas as pd
import warnings
warnings.simplefilter("ignore")                             # possible warnings are suppressed
start_time = time.time()

def takeClosest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return after
    else:
       return before

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

"""
Two smoothing methods:
    -average value of the surrounding x values
    -convolution of the surrounding x values with an Savitzky-Golay filter
"""
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

#%%
"""reading input variables"""
file = pd.read_excel(open("input_variables.xls",'rb'), sheetname='Input_variables')
file_name = file.ix[0,1]

# constants
dist = file.ix[1,1]                                       # Spacing between points in the width profile (m)
amp_m = file.ix[2,1]                                      # Tidal amplitude at the mouth of the estuary (m)
amp_r = file.ix[3,1]                                      # Tidal amplitude at upstream boundary of width profile (m)
T = file.ix[4,1]                                          # Time for one tidal cycle (from maximum low water to maximum low water) (hours)

# s is the shape factor of the cross section, where 1 is perfect rectangular and 2 is a v-shaped cross-section.
s_r = file.ix[5,1]                                        # Shape factor of the channel at the river side of the estuary (-)
s_m = file.ix[6,1]                                        # Shape factor of the channel at the mouth side of the estuary (-)
w_r2 = file.ix[7,1]                                       # Width of the river with at the tidal limit (m)

add = file.ix[8,1]                                        # The name which is added to all created files
crea = file.ix[9,1]                                       # Do you want to create hypsometries yes or no
discharge = file.ix[10,1]                                 # river fresh water discharge (m3/s) manual or model input
q_r = file.ix[11,1]                                       # river fresh water discharge (m3/s) manual input

# input salinity
salin = file.ix[12,1]                                     # do salinity calculations need to take place, yes or no
S_m = file.ix[13,1]                                       # salinity at estuary mouth (ppt)
S_r = file.ix[14,1]                                       # salinity at river, assumed to be 0 (ppt)
rho_0 = file.ix[15,1]                                     # density of seawater (kg/m3)
rho_1 = file.ix[16,1]                                     # density of riverwater (kg/m3)

# input r and z
r_and_z = file.ix[17,1]                                   # are the r and z values manually or automatically used as input
r1 = file.ix[18,1]                                        # manually input for r value
z1 = file.ix[19,1]                                        # manual input for z value
excel = file.ix[20,1]                                     # Create an excel file with output data? yes or no

# Optional input with measured depth
Measured_depth = file.ix[21,1]                            # input of measured depth values? yes or no
h_m_measured = file.ix[22,1]                              # Maximum depth at estuary mouth (m)
h_r_measured = file.ix[23,1]                              # Maximum depth at upstream river (m)

print("--- %s seconds for importing data---" % (time.time() - start_time))

# width of the estuary
w = []                                                    # Along-channel width profile of the estuary (m)
raw_data = csv.reader(open(file_name, "r"))               # het openen van de data (outline vd estuary) en toeschrijven aan een variable in python zodat het uitgelezen kan worden (r)
next(raw_data)                                            # skips the first row with headers
for row in raw_data:                                      # de for loop die runt nu elke rij uit het bestand data en doet per rij de code uitvoeren
    w.append(float(row[1]))
w = np.array(w)

#%%
""" Basic calculations"""
points = len(w)                                     # number of transacts (-)
length_est = dist * (points)                        # length of the estuary (m)
amp = np.linspace(amp_m,amp_r,points)               # tidal amplitude along the estuary (m)
ten_perc = int(round(0.1*points))                   # ten percent of the number of transects (-)
two_perc = int(round(0.02*points))                  # two percent of the number of transects (-)
t = (T*60*60*0.5)                                   # duration from mean high tide to mean low tide (seconds)
tr_m = 2 * amp_m                                    # tidal range mouth
tr_r = 2 * amp_r                                    # tidal range upstream
tr = np.linspace(tr_m,tr_r,points)                  # tidal range in estuary, a linear profile is assumed (m)

#%%
"""tidal prism"""
x = []                                              # location at the estuary from the mouth (m)
tp_temp = []                                        # Local tidal prism that occurs at one cross section of the estuary (m^3)
tp = []                                             # Local tidal prism that occurs landward of location x in the estuary (m^3)
tp2 = []                                            # Local tidal prism that occurs landward of location x in the estuary (m^3)
i = 0

# discharge
if discharge == 0:
    q_b = (w_r2/3.67)**(1/0.45)                     # bankfull discharge (m^3/s), hydraulic geometry based on river width (e.g. Hey and Thorne, 1986)
    q_r = 0.5 * q_b                                 # river discharge (m^3/s), estimation
else:
    q_b = q_r * 2

#tidal prism
while i<points:
    x.append(i*dist)                                # multiply each iterating i with the distance between two transacts to compute the location of each x on the centerline
    tp_temp.append(w[i]*amp[i]*dist)                # compute the tidal prism of the upstream area. Upstream area*amplitude*2.0 (formula) (m^3)
    i=i+1

typical_excursion = (t)*1                           # typical tidal excursion length [in meters] based on average flow of 1 m/s               
nr_of_trans_exc = round(typical_excursion/dist)     # typical tidal excursion converted to nr of transects

for i in range(points):   
    if i+nr_of_trans_exc<points:
        tp.append(sum(tp_temp[i:(i+nr_of_trans_exc)])*2+q_r*t)  # Tidal prism over a tidal excursion length - multiply by 2 because tidal amplitude is used, subsequently add river discharge
        tp2.append(sum(tp_temp[i:])*2+q_r*t)                    # Tidal prism over entire basin - multiply by 2 because tidal amplitude is used, subsequently add river discharge
    else:
        tp.append(sum(tp_temp[i:])*2+q_r*t)                     # Tidal prism over a tidal excursion length - multiply by 2 because tidal amplitude is used, subsequently add river discharge
        tp2.append(sum(tp_temp[i:])*2+q_r*t)                    # Tidal prism over entire basin -  multiply by 2 because tidal amplitude is used, subsequently add river discharge

csa_x = 0.13 * 0.001 * np.array(tp)                             # Estimation of cross sectional area at locatoin x (m^2) = 0.13*10^-3 * TP (Leuven et al., 2018, ESPL) (m^2)

if  amp_m<=0.1:    
    csa_x[0] =  w[0] * 0.33 * (2*tp[0]/(6*60*60)) ** 0.35       # Estimation of cross sectional area at locatoin x (m^2) = 0.13*10^-3 * TP (Leuven et al., 2018, ESPL) (m^2)

h_x = csa_x/w                                                   # average depth of the cross section at location x (m) (Eq. 5 in Leuven et al., 2018, Esurf)
x_km = np.array(x)/1000                                         # the location at the estuary from the mouth (km)
del i, row, tp_temp

#%% Not used

cont_fac = []                                             # Correction factor for continuity?
"""ratio tidal prism and width increae"""
for i in range(points-1):       
    cont_fac.append((tp[i]/tp[i+1])/(w[i]/w[i+1]))

cont_fac.append(1)
#%%
"""Along channel profile with average and maximum depth"""
w_m = w[0]                                          # width at the estuary mouth (m)
w_r = min(w[points-two_perc:])                      # minimum width at the most landward 10% of 10% percent (riverside) (m)

if Measured_depth == 0:       
    h_avg_river = 0.33 * q_b ** 0.35                # average depth of the river (m) (e.g. Hey and Thorne, 1986)
    h_max_mouth = s_m * (h_x[0])                    # maximum depth at the mouth determined by the maximum depth of the first 2% transects (m)
    h_avg_mouth = (h_x[0])
    h_max_river = s_r * h_avg_river                         # maximum depth of the river dependent on w_r and a constant (m)
    h_max = np.linspace(h_max_mouth, h_max_river, points)   # Calculate linear profile with estimated maximum depth (m)

if Measured_depth == 1:  
    h_max_mouth = h_m_measured
    h_max_river = h_r_measured
    h_avg_river = h_r_measured/s_r
    h_avg_mouth = h_m_measured/s_r
    h_max = np.linspace(h_max_mouth, h_max_river, points)   # Calculate linear profile with estimated maximum depth (m)

#%%
"""Estuary shape"""
#Convergence length is the distance over which the channel width of the estuary mouth reduces by a factor e
lw = (-length_est) * (1/(math.log(w_r/w_m)))        # convergence length (m)
w_ideal = []                                        # ideal estuary width (m)

for i in range(points):
    w_ideal.append(w_m * math.exp(-(x[i]) / lw))
w_excess = w - np.array(w_ideal)                    # excess estuary width (m)

#%%
"""Bar patterns"""
w_bar = 0.39 * w **0.92                                             # Predicted partitioned bar width (m), (eq. 8 in Leuven et al., 2018, ESPL)
bi = savgol_filter(np.array(w_excess)/w_bar,3,2,mode='nearest')     # Braiding index (eq. 5 in Leuven et al., 2018, ESPL)
bi[bi < 1] = 1                                                      # Braiding index minimum is 1                            

#%%
"""Hypsometry"""
if r_and_z == 1:
    r = r1
else:
    r = [0.01, 0.05, 0.10, 0.25, 0.50]                          # Array of possible r-values according to Strahler (1952)
    r = r[4]                                                    # Based on Leuven et al., 2018 (Esurf) an r-value of 0.50 is used
w2 = []                                                         # Rounded and integer list of w in order to be able to retrieve the index

for i in range(points):
    w2.append(int(round(w[i])))

w_max = max(w2[:])                                              # maximum width of the estuary
w_arr = np.ones((points, w_max)) * np.nan                       # 2d array of points lenght and the width of the estuary for width, filled with nan values
for i, xx in enumerate(w2):
    w_arr[i,:xx] = np.arange(0,xx,1)/xx                         # changing the nan-values in w_arr into values linear from 0 to 1
del xx
ww=np.ones((points, w_max)) * np.nan                            # 2d array of points lenght and the width of the estuary for width, filled with nan values
for i, xx in enumerate(w2):
    ww[i,:xx] = np.ones((xx))                                   # changing the nan-values in ww into ones, with respect to the width of the estuary
del xx

if r_and_z == 1:
    z = np.ones((1,points)) * z1
else:
    z = 1.4 * (np.array(w_ideal) / w2)**1.2                     # Calculate z at every transect, based on the ideal width, (Eq. 5 in Leuven et al., 2018, Esurf)

hyps_ra = h_max + amp                                           # Range over which the hypsometry is spread out (m)
hyps_norm = (((r/(1-r))*abs(1/((1-r)*w_arr.T+r)-1))**z).T       # Normalized hypsometry, calculated from r and z, Strahler (1952)
hyps_amp = ((hyps_norm.T - 0.5) * 2 * 0.5 * hyps_ra).T          # Stretch normalized hypsometry over the range set by the depth and tidal amplitude
hyps = (hyps_amp.T- (0.5 * hyps_ra) + amp).T                    # Hypsometry for each cross section, such that top is at +amplitute and bottom is at maximum depth

if r_and_z == 1:
    z = z.T

#%%
"""
## Sub-area calculations ##
This is done based on the distance below mean high water. 
Firstly the level of -3 times the tidal amplitude (the lower limit of the shallow subtidal area) is calculated.
Secondly the location of approximately this value on each cross section is determined (with the find_nearest definition).
Thirdly a linear line is created between 0 and the value closest to -3 times amplitude.
"""

hyps_below_HWL = (hyps.T-amp).T                    # depth (hypsometry) below mean high water level
hyps_for_save = np.sort(hyps_below_HWL)            # Sort such that deepest part is on left side for save in xls

#hy=hyps                                           # depth below mean water level
#hyp                                               # rename of hyps to the depth below mean high water level
start_time3 = time.time()

# Predefine variables
intertidal_high = []
h_intertidal_high = []
h_intertidal_high2 = []
intertidal_low = []
h_intertidal_low = []
h_intertidal_low2 = []
h_subtidal = []
h_sub_deep = []
h_sub_shal = []
a = []
b = []
c = []
ii = []

# Set amplitude tresholds
amp2 = (amp * -2).tolist()
amp3 = (amp * -3).tolist()
hyps2 = hyps_below_HWL[:].tolist()
length_linearline = 1000


ii = np.sort(hyps_below_HWL)
for i in range(points):
    a.append(find_nearest(ii[i],amp3[i]))                   # Determine the value of the hypsometry that is closest to -3 amp
    b.append(hyps2[i][:].index(a[i]))                       # Determine the indexed location of the value a in the list from hypsometry (b)
    c.append(np.linspace(hyps_below_HWL[i][0],hyps_below_HWL[i][b[i]],length_linearline))  # Create a linear line between the highest value of the hyps and the value closest to -3 * amp (b)

c = np.array(c).tolist()                # Make a list from variable C in order to be able to sort each row from low to high
d = np.sort(c[:])                       # Make a copy of variable C in order be able to determine the index in the list of C after it's been sorted
e = []                                  # create a variable in order to save the value of C closest to 0.0 (mean water level, between intertidal low and intertidal high area)
f = []                                  # Create a new variable in order to save the indexed location of the value from C
g = []
aa = []
ba = []
ca = []
intertidal_high2 = []
h_int_low_max = []
h_int_high_max = []
h_sub_max = []
h_sub_deep_max = []
h_sub_shal_max = []

for i in range(points):
    e.append(find_nearest(d[i], -amp[i]))                       # Determine the value from the linear line C that is closest to 0.0 (msl; -amp)
    f.append(c[i].index(e[i]))                                  # Determine the indexed location of the value e in the list from hypsometry between amp and -2amp (c)
    g.append(c[i].index(e[i]))                                  # Determine the indexed location of the value e in the list from hypsometry between amp and -2amp (c)
    intertidal_low2 = (find_nearest(d[i], amp2[i]))             # Determine the value from the linear line C that is closest to -amp (low-intertidal to subtidal)
    intertidal_low.append(c[i].index(intertidal_low2))          # Determine the indexed location of the value intertidal_low2 in the list from hypsometry between amp and -2amp (c)
    intertidal_high2.append(find_nearest(d[i], 0.0))            # Determine the value from the linear line C that is closest to amp (low-intertidal to subtidal)
    intertidal_high.append(c[i].index(intertidal_high2[i]))     # Determine the indexed location of the value intertidal_high2 in the list from hypsometry between amp and -2amp (c)
    aa.append(round((f[i]/length_linearline) * b[i]))                   # The location of the value f divided by the location b which represents the width
    ba.append(round((intertidal_low[i]/length_linearline) * b[i]))
    ca.append(round((intertidal_high[i]/length_linearline) * b[i]))

    h_intertidal_low.append(abs(np.mean(hyps[i][aa[i]:ba[i]], axis=0)))       # Append the mean from the hypsometry values from index f to index intertidal_low, representing the depth between the higher and lower limit of the low intertidal area
    h_intertidal_high.append(((np.mean(hyps[i][ca[i]:aa[i]], axis=0))))       # Append the mean from the hypsometry values from index intertidal_high to index f, representing the depth between the higher and lower limit of the high intertidal area
    h_subtidal.append(abs(np.mean(hyps[i][ba[i]:w2[i]], axis=0)))             # Append the mean from the hypsometry values from index intertidal_low to the last value, representing the depth between the higher and lower limit of the subtidal area
    h_sub_deep.append(abs(np.mean(hyps[i][b[i]:w2[i]], axis=0)))
    h_sub_shal.append(abs(np.mean(hyps[i][ba[i]:b[i]], axis=0)))
    h_sub_deep_max.append((np.min(hyps[i][b[i]:w2[i]], axis=0)))
    
    try:    
        h_int_low_max.append((np.min(hyps[i][aa[i]:ba[i]], axis=0)))
    except:
        h_int_low_max.append(np.nan)

    try:           
        h_int_high_max.append(((np.min(hyps[i][ca[i]:aa[i]], axis=0))))
    except:
        h_int_high_max.append(np.nan)        

    try:           
        h_sub_max.append((np.min(hyps[i][ba[i]:w2[i]], axis=0)))
    except:
        h_sub_max.append(np.nan)          

    try:           
        h_sub_shal_max.append((np.min(hyps[i][ba[i]:b[i]], axis=0)))
    except:
        h_sub_shal_max.append(np.nan)          
    
    w_int_high = np.array(aa) - np.array(ca)                # width of the intertidal high area
    w_int_low = np.array(ba) - np.array(aa)                 # width of the intertidallow area

w_sub = w2 - np.array(ba)                               # width of the subtidal area
w_sub_deep = w2 - np.array(b)
w_sub_shal = np.array(b) - np.array(ba)   
w_int_low_rel = w_int_low / w2                          # relative width of the intertidal low area to the total width
w_int_high_rel = w_int_high / w2                        # relative width of the intertidal high area to the total width
w_sub_rel = w_sub / w2                                  # relative width of the subtidal area to the total width

h_int_high = np.array(h_intertidal_high)                # average depth of the high intertidal area
h_int_low = np.array(h_intertidal_low) - h_int_high     # average depth of the low intertidal area
h_sub = np.array(h_subtidal) - h_int_low                # average depth of the subtidal area

#%%
"""
Flow velocity calculations
"""

hyp_mean=[]                     # Array for modelled average depth per transect
u_trans=[]                      # Array for average flow velocity per transect, based on the modelled Tidal prism and the modelled average depth.
u_h=[]                          # Array for max. flow prediction based on model results
u_calib=[]                      # Array for constant to calculate velocities over depth, in which it is assumed that the velocity at 0m below high water level is 0 m/s
u_d2=[]                         # Array to store velocity calculations

for i in range(points):
    hyp_mean.append(np.nanmean(hyps_below_HWL[i,:]))        # Mean depth per transect [m]
    u_trans.append(abs((tp[i])/(hyp_mean[i]*t*w2[i])))      # Average flow velocity per transect, based on tp [m/s]
    u_calib.append(u_trans[i]/hyp_mean[i])                  # Assume that the velocity at 0m below high water level is 0 m/s, calculate constant to obtain this

hyp_d=(hyps_below_HWL.T-hyp_mean).T                         # Difference between the hypsometry profile and the average depth at that transect

hyps_below_HWL_for_u = -hyps_below_HWL
hyps_below_HWL_for_u[hyps_below_HWL_for_u<1]=1

u_h_max =0.0983+0.9126*(np.log10(hyps_below_HWL_for_u))     # max. flow prediction based on regression U=F(h) in model results [m/s]
u_h_mean = (u_h_max.T * np.linspace((2/np.pi),1,points)).T
u_h_mean = (u_h_max.T - (amp/max(amp))* (2/np.pi)*u_h_max.T).T

tp_depth =[]                         # Array to store tidal prism calculation based on flow velocity from depth

for i in range(points):
    tp_depth.append((-hyps_below_HWL[i,:])*u_h_mean[i,:]*t)
tp_d =  np.nansum(tp_depth,axis=1)   
    

for i in range(points):
    u_d2.append(u_calib[i]*hyp_d[i,:])                      # Difference in flow velocity between average and other points along the transect
u_d2 = np.array(u_d2)                                       # Convert list into matrix array

u_new=(u_d2.T+np.array(u_trans)).T                          # Calculated flow velocity, based on the tidal prism, deviation from average depth and relation between depth and flow velocity
u_new[u_new < 0] = 0                                        # Flow velocity minimum is 0.                         

u=-1*np.sort(u_new*-1)                                      # sorting of the flow velocity from high to low values

u_max = (u_new.T * np.linspace(2,1,points)).T
u_max_sorted=-1*np.sort(u_max*-1)

u_h_max_sorted = -1*np.sort(u_h_max*-1)
u_h_mean_sorted = -1*np.sort(u_h_mean*-1)

u_trans2=[]                                                 # average flow velocity per transect, based on the modelled flow velocities
u_max_new=[]                                                # Width averaged max flow velocity per transect, based on the modelled flow velocities
u_max_new2=[]                                                # Max flow velocity per transect, based on the modelled flow velocities
u_h_trans2=[]                                                 # average flow velocity per transect, based on the modelled flow velocities
u_h_max_new=[]
u_h_max_new2=[]

for i in range(points):
    u_trans2.append(np.mean(u_new[i,:w2[i]]))
    u_max_new.append(np.mean(u_max[i,:w2[i]]))          # Width averaged max flow velocity
    u_max_new2.append(np.max(u_max[i,:w2[i]]))
    u_h_trans2.append(np.mean(u_h_mean[i,:w2[i]]))
    u_h_max_new.append(np.mean(u_h_max[i,:w2[i]]))
    u_h_max_new2.append(np.max(u_h_max[i,:w2[i]]))          # Width averaged max flow velocity

"""
Flow velocity typical values for distinct zones
"""
u_int_low = []
u_int_high = []
u_sub = []
u_sub_deep = []
u_sub_shal = []
u_int_low_max = []
u_int_high_max = []
u_sub_max = []
u_sub_deep_max = []
u_sub_shal_max = []

for i in range(points):
    u_int_low.append((np.mean(u_h_mean[i][aa[i]:ba[i]], axis=0)))       # Append the mean from the hypsometry values from index f to index intertidal_low, representing the depth between the higher and lower limit of the low intertidal area
    u_int_high.append(((np.mean(u_h_mean[i][ca[i]:aa[i]], axis=0))))       # Append the mean from the hypsometry values from index intertidal_high to index f, representing the depth between the higher and lower limit of the high intertidal area
    u_sub.append((np.mean(u_h_mean[i][ba[i]:w2[i]], axis=0)))             # Append the mean from the hypsometry values from index intertidal_low to the last value, representing the depth between the higher and lower limit of the subtidal area
    u_sub_deep.append((np.mean(u_h_mean[i][b[i]:w2[i]], axis=0)))
    u_sub_shal.append((np.mean(u_h_mean[i][ba[i]:b[i]], axis=0)))
    u_sub_deep_max.append((np.max(u_h_max[i][b[i]:w2[i]], axis=0)))

    try:           
        u_int_low_max.append((np.max(u_h_max[i][aa[i]:ba[i]], axis=0)))
    except:
        u_int_low_max.append(np.nan)       

    try:           
        u_int_high_max.append(((np.max(u_h_max[i][ca[i]:aa[i]], axis=0))))
    except:
        u_int_high_max.append(np.nan)  
        
    try:           
        u_sub_max.append((np.max(u_h_max[i][ba[i]:w2[i]], axis=0)))
    except:
        u_sub_max.append(np.nan)          
        
    try:           
        u_sub_shal_max.append((np.max(u_h_max[i][ba[i]:b[i]], axis=0)))
    except:
        u_sub_shal_max.append(np.nan)  
        
#%%
"""Inundation duration""" 
duration = np.ones((points, w_max)) * np.nan                # relative inundation duration of the whole estuary
duration2 = np.ones((points, w_max)) * np.nan                # relative inundation duration of the whole estuary

for i in range(points):
    duration[i,:ba[i]] = 0.5-(0.5*(np.sin((2*np.pi)/(4*amp[i])*hyps[i,:ba[i]])))
    duration[i,ba[i]:w2[i]] = np.ones(w_sub[i])
    
duration = np.sort(abs(duration)*-1)*-1

#%%
"""Salinity:"""

if salin == 1:       
    """General parameters used in all salinity calculations"""
    la = (-length_est) * (1/(math.log((h_avg_river*min(w2[points-ten_perc:]))/(csa_x[0])))) # convergence length CSA
    a0 = csa_x[0]                              # Cross sectional area mouth [m^2]
    g = 9.81                                   # Gravitational constant
    d_rho = float(abs(rho_0-rho_1))            # Density difference salt and fresh water
    
    """Brockway"""
    beta = -(math.log((h_avg_river*w2[points-1])/(csa_x[0])))/(length_est) # CSA convergence length (Brockway, 2006)
    kx = 0.28 * q_r + 13 * np.mean(tr)                                     # longitudinal mixing coefficient (m2/s)
    s_brock = []                                                           # salinity according to formula by Brock (ppt)   
    for i in range(points):
        s_brock.append(S_m * math.exp(- (q_r / (beta * kx * a0 ) * (math.exp(beta*x[i])-1))))
    
    """Savenije"""
    v0 = u_h_max_new2[0]                    # maximum tidal velocity amplitude at the mouth (m/s)
    h_x_m = (h_avg_mouth)                   # average depth
    N = (q_r*t)/ tp[0]                      # Eq. 2 (Savenije, 1993), Canter Cremers' estuary number 
    F_d = (rho_1 * v0**2)/(d_rho*g*h_avg_mouth)   # Eq. 3 (Savenije, 1993), densimetric Froude number 
    N_r = N/F_d                             # p. 214 (Savenije, 1993), Estuarine Richardson number 
    excur = 1.080 * v0 * t /math.pi         # p. 205 (Savenije, 1993), tidal excursion
    
    D0 = 220*np.sqrt(40) * h_avg_mouth / (la ) * np.sqrt(N_r) * v0 * excur    # Eq. 25 (Savenije, 1993), Dispersion coefficient at the estuary mouth
    k = 0.16 * 10**-6 * (h_avg_mouth**0.69 * g**1.12 * t**2.24) / (tr_m**0.59 * lw**1.1 * w_m**0.13) # Eq. 19 in Savenije (1993), dispersion reduction rate
    if k>1: 
        k = 1                                        # max value is 1                         
   
    beta_Sav = (k * la * q_r) / (D0 * a0)            # Eq. 12 (Savenije, 1993), longitudinal variation in dispersion
    ddd0 = []                                        # = D/D0
    s_sav = np.ones(points) * np.nan                 # salinity according to Savenije (1993) (ppt)
    for i in range(points):
        ddd0.append(1 - beta_Sav * (math.exp(x[i] / la) - 1)) # Eq. 11 (Savenije, 1993)
        if ddd0[i] < 0:
            del ddd0[-1]
            break
        s_sav[i] = (S_m - S_r) * (ddd0[i]**(1 / k)) + S_r     # Eq. 10, salinity according to Savenije (1993) (ppt)

    """ Gisen"""
    infl_point = 0                              # location of the infliction point (m)
    infl_p = int(round(infl_point/dist))        # infliction point
    bf = w_r2                                   # river regime width
    hh1 = tr[infl_p]                            # tidal range at infliction point
    b1 = w[infl_p]                              # estuary width at infliction point
    c1 = 42                                     # Roughness (estimate)
    rs = (w_int_low[0]+w_int_high[0])/w_sub[0]  # Storage ratio: width tidal-flat / channel width
   
    D0_Gis = 1400        * h_avg_mouth / (la )       * np.sqrt(N_r) * v0 * excur    # Eq. 14, Gisen et al, 2015, Dispersion coefficient at the estuary mouth
    k1= 151.35*10**-6 * ((bf**0.3 * hh1**0.13 * t**0.97) / (b1**0.30 * c1**0.18 * v0**0.71 * lw**0.11 * h_avg_mouth**(-0.15) * rs**0.84)) # Eq. 39, Gisen et al, 2015
    if k1>1: 
        k1 = 1                                  # max value is 1      
    
    beta_Gis = (k1 * la * q_r) / (D0_Gis * a0)  # Eq. 12 (Savenije, 1993), longitudinal variation in dispersion
    ddd0_Gis = []                               # = D/D0
    s_gis = np.ones(points) * S_r               # salinity according to Gisen et al. (2015) (ppt)
    
    for i in range(points):
        ddd0_Gis.append(1 - beta_Gis * (math.exp(x[i] / la) - 1))
        if ddd0_Gis[i] < 0:
            del ddd0_Gis[-1]
            break
        s_gis[i] = (S_m - S_r) * (ddd0_Gis[i]**(1 / k1)) + S_r
    
    sal =[]
    for i in range(points):
        sal.append((s_sav[i]+s_gis[i])/2)

    salinity = (ww.T * sal).T

#%%
"""data output"""

if excel == 1:
    os.chdir('..\\zip_Estuarine_Morphology_Estimator\\Results')               # location to save data output
    print("Excel file with output data is being created")
    start_time5 = time.time()

    data_selection = pd.DataFrame({ 'Width':w,
                                    'Width intertidal high':w_int_high,
                                    'Width intertidal low':w_int_low,
                                    'Width subtidal area':w_sub,
                                    'Width shallow subtidal area':w_sub_shal,
                                    'Width deep subtidal area':w_sub_deep,
                                    'Average depth intertidal high':h_int_high,
                                    'Average depth intertidal low':h_int_low,
                                    'Average depth subtidal area':h_sub,
                                    'Maximum flow velocity intertidal high':u_int_high_max,
                                    'Average flow velocity intertidal high':u_int_high,
                                    'Maximum flow velocity intertidal low':u_int_low_max ,
                                    'Average flow velocity intertidal low':u_int_low ,
                                    'Maximum flow velocity subtidal area':u_sub_max ,
                                    'Average flow velocity subtidal area':u_sub ,
                                    'Maximum flow velocity shallow subtidal area':u_sub_shal_max ,
                                    'Average flow velocity shallow subtidal area':u_sub_shal,
                                    'Maximum flow velocity deep subtidal area':u_sub_deep_max ,
                                    'Average flow velocity deep subtidal area':u_sub_deep,
                                    'Salinity':sal ,
                                    'Continuity factor': cont_fac ,
                                    'Tidal prism based on area': tp2 ,
                                    'Tidal prism based on area (excursion length limit)': tp ,
                                    'Tidal prism based on flow velocities': tp_d ,
                                    'Salinity Sav': s_sav ,
                                    'Salinity Gisen': s_gis ,
                                    'Salinity Brockway': s_brock })
    
    ds=data_selection[['Width','Width intertidal high','Width intertidal low', 'Width subtidal area',
                       'Width shallow subtidal area', 'Width deep subtidal area', 'Average depth intertidal high',
                       'Average depth intertidal low', 'Average depth subtidal area', 
                       'Average flow velocity intertidal high', 'Average flow velocity intertidal low',
                       'Average flow velocity subtidal area', 'Average flow velocity shallow subtidal area', 
                       'Average flow velocity deep subtidal area',
                       'Maximum flow velocity intertidal high', 'Maximum flow velocity intertidal low',
                       'Maximum flow velocity subtidal area', 'Maximum flow velocity shallow subtidal area', 
                       'Maximum flow velocity deep subtidal area','Salinity','Continuity factor',
                       'Tidal prism based on area','Tidal prism based on area (excursion length limit)',
                       'Tidal prism based on flow velocities','Salinity Sav','Salinity Gisen','Salinity Brockway']]
    
    writer = pd.ExcelWriter("output_"+str(add)+".xlsx")
    hypsometry = pd.DataFrame(np.array(hyps_for_save), index= np.arange(0,points,1), columns= np.arange(0,w_max,1))
    hypsometry.to_excel(writer,'Hypsometry')
    u_pd = pd.DataFrame(np.array(u), index= np.arange(0,points,1), columns= np.arange(0,w_max,1))
    u_pd.to_excel(writer,'Flow velocities mean f(P)')
    u_max_sorted_pd = pd.DataFrame(np.array(u_max_sorted), index= np.arange(0,points,1), columns= np.arange(0,w_max,1))
    u_max_sorted_pd.to_excel(writer,'Flow velocities max f(P)')    
    u_h_mean_sorted_pd = pd.DataFrame(np.array(u_h_mean_sorted), index= np.arange(0,points,1), columns= np.arange(0,w_max,1))
    u_h_mean_sorted_pd.to_excel(writer,'Flow velocities mean f(h)')  
    u_h_max_sorted_pd = pd.DataFrame(np.array(u_h_max_sorted), index= np.arange(0,points,1), columns= np.arange(0,w_max,1))
    u_h_max_sorted_pd.to_excel(writer,'Flow velocities max f(h)')    
         
    durationw = pd.DataFrame(np.array(duration), index= np.arange(0,points,1), columns= np.arange(0,w_max,1))
    durationw.to_excel(writer,'Rel. inundation duration')
    ds.to_excel(writer,'data')
    writer.save()
    print("--- %s seconds for creating excel output---" % (time.time() - start_time5))
else:
    print("Excel file with output is not created as indicated in the document Input_variables, run continues")


#%%  ######################################################################################################################################
#%%
"""plot creation and saving"""
start_time2 = time.time()
if crea == 1:
    print("hypsometry graphs are being created")
    os.chdir('..\\zip_Estuarine_Morphology_Estimator\\Results\\Hypsometry_graphs')
    for i in range(points):
        plt.title('Hypsometry profile '+str(add)+' ('+ str(float(i)/points*length_est/1000)+'km)', fontsize = 14, fontweight = 'bold')
        plt.plot(np.arange(w_max), hyps[i,:],'b')
        plt.grid()
        plt.ylabel('Depth (m)')
        plt.xlabel('Width (m)')
        plt.legend()
        plt.savefig("hyps"+str(i)+".png", dpi = 100)
        plt.close()
    np.savetxt("hyps"+str(add)+".csv", hyps, delimiter=',')
    os.chdir('..')
    os.chdir('..')
    print("--- %s seconds plotting hypsometry graphs---" % (time.time() - start_time2))

#%%
start_time4 = time.time()
""" Graphs
In this paragraph all graphs other than the hypsometry are created and saved to the folder 'Results'
"""
if crea == 1:
    os.chdir('..\\zip_Estuarine_Morphology_Estimator\\Results')               # where are the figures saved

if excel == 0:
    os.chdir('..\\zip_Estuarine_Morphology_Estimator\\Results')               # location to save data output

matplotlib.rcParams.update({'font.size': 6})

#%%

#continuity factor?
#plt.title('Continuity factor ('+str(add)+')', fontsize = 14, fontweight = 'bold')
#plt.plot(x_km,[1]*(points), 'k')
#plt.plot(x_km[:(points)],cont_fac,'b')
##norm_excess = ([1]*(points-1)+(np.array( (w_excess[:points-1]) / w_ideal[:points-1])))
##diff_norm_excess = norm_excess.deriv()
#plt.plot(x_km[:(points-1)],[1]*(points-1)+np.gradient([1]*(points-1)+(np.array( (w_excess[:points-1]) / w_ideal[:points-1]))),'r--')
#plt.xlabel('Distance to mouth (km)')
#plt.ylabel('Downstream change in prism / change in width, dP/dW (-)')
#axes = plt.gca()
#axes.set_ylim([min(cont_fac)*0.95,max(cont_fac)*1.05])
#plt.grid(linewidth=0.2)
#plt.savefig(str(add)+'_continuity_factor.png', dpi=300)
#plt.close()

#tidal prism
plt.title('Tidal prism ('+str(add)+')', fontsize = 14, fontweight = 'bold')
plt.plot(x_km[:(points)],tp2, label='Surface area * amplitude')
plt.plot(x_km[:(points)],tp, label='Surface area * amplitude [excursion length]')
plt.plot(x_km[:(points)],tp_d,label='Predicted velocity * depth * time')
#norm_excess = ([1]*(points-1)+(np.array( (w_excess[:points-1]) / w_ideal[:points-1])))
#diff_norm_excess = norm_excess.deriv()
plt.xlabel('Distance to mouth (km)')
plt.ylabel('Tidal prism ($m^3$)')
axes = plt.gca()
axes.set_ylim([min(tp)*0.95,max(tp_d)*1.5])
plt.grid(linewidth=0.2)
plt.legend(loc='upper right')
plt.savefig(str(add)+'_tidal_prism.png', dpi=300)
plt.close()

#braiding index
plt.title('Braiding Index ('+str(add)+')', fontsize = 14, fontweight = 'bold')
plt.plot(x_km, bi)
plt.xlabel('Distance to mouth (km)')
plt.ylabel('Braiding Index (-)')
axes = plt.gca()
axes.set_ylim([0.9,max(bi)*1.1])
plt.grid(linewidth=0.2)
plt.savefig(str(add)+'_braiding_index.png', dpi=300)
plt.close()

# Bar width
plt.title('Bar width ('+str(add)+')', fontsize = 14, fontweight = 'bold')
plt.plot(x_km, w_bar, label='predicted partitioned bar width (m), Leuven et al. (2018)' )
plt.xlabel('Distance to mouth (km)')
plt.ylabel('Bar width (m)')
plt.grid(linewidth=0.2)
plt.legend(loc='upper right')
plt.savefig(str(add)+'_bar_width.png', dpi=300)
plt.close()

#%% Combined figure

# Bar width
plt.subplot(3,1,1)
plt.plot(x_km, w_bar, label='predicted partitioned bar width (m), Leuven et al. (2018)' )
#plt.xlabel('Distance to mouth (km)')
plt.ylabel('Bar width (m)')
plt.grid(linewidth=0.2)
plt.legend(loc='upper right')

# Braiding index
plt.subplot(3,1,2)
plt.plot(x_km, bi)
#plt.xlabel('Distance to mouth (km)')
plt.ylabel('Braiding Index (-)')
axes = plt.gca()
axes.set_ylim([0.9,max(bi)*1.1])
plt.grid(linewidth=0.2)

#flow velocity at average depth
plt.subplot(3,1,3)
plt.plot(x_km, u_h_trans2, '-', label='Width averaged mean')
plt.plot(x_km, u_h_max_new, '-', label='Width averaged max')
plt.plot(x_km, u_h_max_new2, '-', label='Max flow velocity per transect')
plt.legend(loc='upper right')
plt.grid(linewidth=0.2)
plt.xlabel('Distance to mouth (km)')
plt.ylabel('Flow velocity (m/s)')
plt.savefig(str(add)+"_bar_width_BI_velocity_combined.png", dpi = 300)
plt.close()

# plotting z (waarmee de vorm van de hyps bepaald wordt)
#plt.title('Z-value in hypsometry fit of Strahler (1952) ('+str(add)+')', fontsize = 14, fontweight = 'bold')
#plt.plot(x_km, z)
#plt.xlabel('Distance to mouth (km)')
#plt.ylabel('Z-value (-)')
#plt.grid(linewidth=0.2)
#plt.savefig(str(add)+"_z.png", dpi = 300)
#plt.close()

#%%

if salin==1:
    plt.suptitle('Salinity ('+str(add)+')', fontsize = 14, fontweight = 'bold')
    plt.plot(x_km, s_brock, label='Brockway et al., 2006')
    plt.plot(x_km, s_sav, 'b', label='Savenije, 1993')
    plt.plot(x_km, s_gis, 'y', label='Gisen et al., 2015')
    plt.plot(x_km, sal, 'g', label='Mean of Gisen et al. & Savenije')
    plt.legend(loc='upper right')
    plt.grid(linewidth=0.2)
    plt.xlabel('Distance to mouth (km)')
    plt.ylabel('Salinity (ppt)')
    plt.savefig(str(add)+"_salinity.png", dpi = 300)
    plt.close()
    
#%%    
    
#flow velocity at average depth
plt.suptitle('Flow velocity, width averaged ('+str(add)+')', fontsize = 14, fontweight = 'bold')
#plt.plot(x_km, u_trans2, '--', label='Width averaged mean')
#plt.plot(x_km, u_max_new, '--', label='Width averaged max')
#plt.plot(x_km, u_max_new2, '--', label='Max flow velocity per transect')
plt.plot(x_km, u_h_trans2, '-', label='Width averaged mean')
plt.plot(x_km, u_h_max_new, '-', label='Width averaged max')
plt.plot(x_km, u_h_max_new2, '-', label='Max flow velocity per transect')

plt.legend(loc='upper right')
plt.grid(linewidth=0.2)
plt.xlabel('Distance to mouth (km)')
plt.ylabel('Flow velocity (m/s)')
plt.savefig(str(add)+"_velocity.png", dpi = 300)
plt.close()

# Width
fig = plt.figure(figsize=(8,8))
fig.suptitle('Width different zones ('+str(add)+')', fontsize = 14, fontweight = 'bold')
gs = gridspec.GridSpec(5,1, height_ratios=[5,0.8,3,3,3])
ax = plt.subplot(gs[0])
ax.grid(linewidth=0.2)
ax.set_title('Cumulative width of the different areas', fontsize = 8)
plt.plot([],[],color='c', label='Width subtidal area', linewidth=5)
plt.plot([],[],color='y', label='Width intertidal low area', linewidth=5)
plt.plot([],[],color='g', label='Width intertidal high area', linewidth=5)
plt.stackplot(x_km, w_int_high, w_int_low, w_sub, colors=['g','y','c'])
plt.ylabel('Width (m)', fontsize = 10)
plt.xlabel('Distance to mouth (km)', fontsize = 10)
plt.legend(loc='upper right')
ax1 = plt.subplot(gs[1])
plt.plot(x_km, w_int_high_rel, 'w')
ax1.spines['bottom'].set_color('w')
ax1.spines['top'].set_color('w')
ax1.xaxis.label.set_color('w')
ax1.spines['left'].set_color('w')
ax1.spines['right'].set_color('w')
ax1.yaxis.label.set_color('w')
ax1.tick_params(axis='x', colors='w')
ax1.tick_params(axis='y', colors='w')
ax2 = plt.subplot(gs[2])
ax2.set_title('Relative width of the different areas', fontsize = 8)
plt.plot(x_km, w_int_high_rel, 'g', label='intertidal high')
plt.legend(loc='lower right')
ax3 = plt.subplot(gs[3])
plt.plot(x_km, w_int_low_rel, 'y', label='intertidal low')
plt.ylabel('Relative width (-)', fontsize = 10)
plt.legend(loc='lower right')
ax4 = plt.subplot(gs[4])
plt.plot(x_km, w_sub_rel, 'c', label='subtidal')
plt.legend(loc='lower right')
plt.xlabel('Distance to mouth (km)')
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax3.get_xticklabels(), visible=False)
plt.savefig(str(add)+"_zones_width.png", dpi=300)
plt.close()

#%%

# Depth
fig = plt.figure(figsize=(8,8))
fig.suptitle('Average depth below maximum water level ('+str(add)+')', fontsize = 14, fontweight = 'bold')
gs = gridspec.GridSpec(5,1, height_ratios=[5,0.8,3,3,3])
ax = plt.subplot(gs[0])
ax.grid(linewidth=0.2)
ax.set_title('Cumulative depth of the different areas', fontsize = 8)
plt.plot([],[],color='g', label='Depth intertidal high', linewidth=5)
plt.plot([],[],color='y', label='Depth intertidal low', linewidth=5)
plt.plot([],[],color='c', label='Depth subtidal area', linewidth=5)
plt.stackplot(x_km, abs(h_int_high), abs(h_int_low), abs(h_sub), colors=['g','y','c'])
plt.ylabel('Depth (m)', fontsize = 10)
plt.legend(loc='lower right')
plt.gca().invert_yaxis()
ax1 = plt.subplot(gs[1])
plt.plot(x_km, h_int_high, 'w')
ax1.spines['bottom'].set_color('w')
ax1.spines['top'].set_color('w')
ax1.xaxis.label.set_color('w')
ax1.spines['left'].set_color('w')
ax1.spines['right'].set_color('w')
ax1.yaxis.label.set_color('w')
ax1.tick_params(axis='x', colors='w')
ax1.tick_params(axis='y', colors='w')
ax2 = plt.subplot(gs[2])
ax2.set_title('Depth of the different areas (m)', fontsize = 8)
plt.plot(x_km, np.array(h_intertidal_high) * -1, 'g', label='intertidal high')
plt.legend(loc='lower right')
plt.gca().invert_yaxis()
ax3 = plt.subplot(gs[3])
plt.plot(x_km, (h_intertidal_low), 'y', label='intertidal low')
plt.ylabel('Depth (m)', fontsize = 10)
plt.legend(loc='lower right')
plt.gca().invert_yaxis()
ax4 = plt.subplot(gs[4])
plt.plot(x_km, (h_subtidal), 'c', label='subtidal')
plt.legend(loc='lower right')
plt.gca().invert_yaxis()
plt.xlabel('Distance to mouth (km)')
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax3.get_xticklabels(), visible=False)
plt.savefig(str(add)+"_zones_average_depth.png", dpi=450)
plt.close()

#%%
# average flow velocities
fig = plt.figure(figsize=(8,8))
fig.suptitle('Tidal flow velocity amplitude ('+str(add)+')', fontsize = 14, fontweight = 'bold')
plt.subplot(5,1,1)
plot1 = plt.plot(x_km, u_int_high_max, 'g', label='max intertidal high')
plot1 = plt.plot(x_km, u_int_high, 'g--', label='mean intertidal high')
#plot1 = plt.plot(x_km, u_max2_intertidal_high, 'g', label='max intertidal high')
plt.legend(loc='upper right')
plt.grid(linewidth=0.2)
plt.subplot(5,1,2)
plot2 = plt.plot(x_km, u_int_low_max, 'y', label='max intertidal low')
plot2 = plt.plot(x_km, u_int_low, 'y--', label='mean intertidal low')
#plot2 = plt.plot(x_km, u_max2_intertidal_low, 'y', label='max intertidal low')
plt.legend(loc='upper right')
plt.grid(linewidth=0.2)
plt.subplot(5,1,3)
plot3 = plt.plot(x_km, u_sub_max, 'c', label='max subtidal')
plot3 = plt.plot(x_km, u_sub, 'c--', label='mean subtidal')
#plot3 = plt.plot(x_km, u_max2_subtidal, 'c', label='max subtidal')
plt.ylabel('Average flow velocities (m/s)', fontsize = 10)
plt.legend(loc='upper right')
plt.grid(linewidth=0.2)
plt.subplot(5,1,4)
plot4 = plt.plot(x_km, u_sub_shal_max, 'b', label='max shallow subtidal')
plot4 = plt.plot(x_km, u_sub_shal, 'b--', label='mean shallow subtidal')
#plot4 = plt.plot(x_km, u_max2_sub_shal, 'b', label='max shallow subtidal')
plt.legend(loc='upper right')
plt.grid(linewidth=0.2)
plt.subplot(5,1,5)
plot5 = plt.plot(x_km, u_sub_deep_max, 'k', label='max deep subtidal')
plot5 = plt.plot(x_km, u_sub_deep, 'k--', label='mean deep subtidal')
#plot5 = plt.plot(x_km, u_max2_sub_deep, 'k', label='max deep subtidal')
plt.xlabel('Distance to mouth (km)', fontsize = 10)
plt.legend(loc='upper right')
plt.grid(linewidth=0.2)
plt.savefig(str(add)+"_zones_average_flow_velocity_.png", dpi=300)
plt.close()

#%% Previous 3 figures combined

fig = plt.figure(figsize=(24,8))
#fig.suptitle('Width different zones ('+str(add)+')', fontsize = 14, fontweight = 'bold')
gs = gridspec.GridSpec(5,3, height_ratios=[5,0.8,3,3,3])
ax = plt.subplot(gs[0])
ax.grid(linewidth=0.2)
ax.set_title('Cumulative width of the different areas', fontsize = 8)
plt.plot([],[],color='c', label='Width subtidal area', linewidth=5)
plt.plot([],[],color='y', label='Width intertidal low area', linewidth=5)
plt.plot([],[],color='g', label='Width intertidal high area', linewidth=5)
plt.stackplot(x_km, w_int_high, w_int_low, w_sub, colors=['g','y','c'])
plt.ylabel('Width (m)', fontsize = 10)
plt.xlabel('Distance to mouth (km)', fontsize = 10)
plt.legend(loc='upper right')
ax1 = plt.subplot(gs[3])
plt.plot(x_km, w_int_high_rel, 'w')
ax1.spines['bottom'].set_color('w')
ax1.spines['top'].set_color('w')
ax1.xaxis.label.set_color('w')
ax1.spines['left'].set_color('w')
ax1.spines['right'].set_color('w')
ax1.yaxis.label.set_color('w')
ax1.tick_params(axis='x', colors='w')
ax1.tick_params(axis='y', colors='w')
ax2 = plt.subplot(gs[6])
ax2.set_title('Relative width of the different areas', fontsize = 8)
plt.plot(x_km, w_int_high_rel, 'g', label='intertidal high')
plt.legend(loc='lower right')
ax3 = plt.subplot(gs[9])
plt.plot(x_km, w_int_low_rel, 'y', label='intertidal low')
plt.ylabel('Relative width (-)', fontsize = 10)
plt.legend(loc='lower right')
ax4 = plt.subplot(gs[12])
plt.plot(x_km, w_sub_rel, 'c', label='subtidal')
plt.legend(loc='lower right')
plt.xlabel('Distance to mouth (km)')
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax3.get_xticklabels(), visible=False)
#plt.savefig(str(add)+"_zones_width.png", dpi=300)
#plt.close()

#fig.suptitle('Average depth below maximum water level ('+str(add)+')', fontsize = 14, fontweight = 'bold')
#gs = gridspec.GridSpec(5,1, height_ratios=[5,0.8,3,3,3])
ax = plt.subplot(gs[1])
ax.grid(linewidth=0.2)
ax.set_title('Cumulative depth of the different areas', fontsize = 8)
plt.plot([],[],color='g', label='Depth intertidal high', linewidth=5)
plt.plot([],[],color='y', label='Depth intertidal low', linewidth=5)
plt.plot([],[],color='c', label='Depth subtidal area', linewidth=5)
plt.stackplot(x_km, abs(h_int_high), abs(h_int_low), abs(h_sub), colors=['g','y','c'])
plt.ylabel('Depth (m)', fontsize = 10)
plt.legend(loc='lower right')
plt.gca().invert_yaxis()
ax1 = plt.subplot(gs[4])
plt.plot(x_km, h_int_high, 'w')
ax1.spines['bottom'].set_color('w')
ax1.spines['top'].set_color('w')
ax1.xaxis.label.set_color('w')
ax1.spines['left'].set_color('w')
ax1.spines['right'].set_color('w')
ax1.yaxis.label.set_color('w')
ax1.tick_params(axis='x', colors='w')
ax1.tick_params(axis='y', colors='w')
ax2 = plt.subplot(gs[7])
ax2.set_title('Depth of the different areas (m)', fontsize = 8)
plt.plot(x_km, np.array(h_intertidal_high) * -1, 'g', label='intertidal high')
plt.legend(loc='lower right')
plt.gca().invert_yaxis()
ax3 = plt.subplot(gs[10])
plt.plot(x_km, (h_intertidal_low), 'y', label='intertidal low')
plt.ylabel('Depth (m)', fontsize = 10)
plt.legend(loc='lower right')
plt.gca().invert_yaxis()
ax4 = plt.subplot(gs[13])
plt.plot(x_km, (h_subtidal), 'c', label='subtidal')
plt.legend(loc='lower right')
plt.gca().invert_yaxis()
plt.xlabel('Distance to mouth (km)')
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax3.get_xticklabels(), visible=False)
#plt.savefig(str(add)+"_zones_average_depth.png", dpi=450)
#plt.close()

#fig = plt.figure(figsize=(8,8))
#fig.suptitle('Tidal flow velocity amplitude ('+str(add)+')', fontsize = 14, fontweight = 'bold')
ax5 = plt.subplot(gs[2])
plot1 = plt.plot(x_km, u_int_high_max, 'g', label='max intertidal high')
plot1 = plt.plot(x_km, u_int_high, 'g--', label='mean intertidal high')
#plot1 = plt.plot(x_km, u_max2_intertidal_high, 'g', label='max intertidal high')
plt.legend(loc='upper right')
plt.grid(linewidth=0.2)
ax6 = plt.subplot(gs[8])
plot2 = plt.plot(x_km, u_int_low_max, 'y', label='max intertidal low')
plot2 = plt.plot(x_km, u_int_low, 'y--', label='mean intertidal low')
#plot2 = plt.plot(x_km, u_max2_intertidal_low, 'y', label='max intertidal low')
plt.legend(loc='upper right')
plt.grid(linewidth=0.2)
ax7 = plt.subplot(gs[11])
plot3 = plt.plot(x_km, u_sub_max, 'c', label='max subtidal')
plot3 = plt.plot(x_km, u_sub, 'c--', label='mean subtidal')
#plot3 = plt.plot(x_km, u_max2_subtidal, 'c', label='max subtidal')
plt.ylabel('Maximum and average flow velocities (m/s) per zone', fontsize = 10)
plt.legend(loc='upper right')
plt.grid(linewidth=0.2)
#ax8 = plt.subplot(gs[11])
#plot4 = plt.plot(x_km, u_sub_shal_max, 'b', label='max shallow subtidal')
#plot4 = plt.plot(x_km, u_sub_shal, 'b--', label='mean shallow subtidal')
#plot4 = plt.plot(x_km, u_max2_sub_shal, 'b', label='max shallow subtidal')
#plt.legend(loc='upper right')
#plt.grid(linewidth=0.2)
ax9 = plt.subplot(gs[14])
plot5 = plt.plot(x_km, u_sub_deep_max, 'k', label='max deep subtidal')
plot5 = plt.plot(x_km, u_sub_deep, 'k--', label='mean deep subtidal')
#plot5 = plt.plot(x_km, u_max2_sub_deep, 'k', label='max deep subtidal')
plt.xlabel('Distance to mouth (km)', fontsize = 10)
plt.legend(loc='upper right')
plt.grid(linewidth=0.2)
plt.savefig(str(add)+"_zones_combined.png", dpi=300)
plt.close()

# Total and subtidal width
#plt.suptitle('Width different areas ('+str(add)+')', fontsize = 14, fontweight = 'bold')
#plt.plot(x_km, w, 'r', label='Total width')
#plt.plot(x_km, w_sub, 'b', label='Width subtidal area')
#plt.plot(x_km, w_ideal,'c', label='Ideal width subtidal area')
#plt.legend(loc='upper right')
#plt.grid(linewidth=0.2)
#plt.xlabel('Distance to mouth (km)')
#plt.ylabel('Width (m)')
#plt.savefig(str(add)+"_width_v2.png", dpi = 300)
#plt.close()

#plt.suptitle('Depth below high water level ('+str(add)+')', fontsize = 14, fontweight = 'bold')
##plt.plot(x_km,w_sub/1000, 'k', label='Width subtidal area',linewidth=0.5)
#plt.contourf(x_km,np.arange(0,w_max,1)/1000,ii.T)
#cbar=plt.colorbar()
#plt.xlabel('Distance to mouth (km)')
#plt.ylabel('Width (km)')
#plt.grid(linewidth=0.2)
#cbar.ax.set_ylabel('bed elevation (m)', rotation = 90)
#plt.savefig(str(add)+"_Depth_v1", dpi = 300)
#plt.close()
 
#%%

plt.suptitle('Depth below high water level ('+str(add)+')', fontsize = 14, fontweight = 'bold')

if (w_max % 2 == 0): #even 
    plt.contourf(x_km,np.arange(0,int(w_max/2),1)/1000,ii.T[::2])
else: #odd
    plt.contourf(x_km,np.arange(0,int(w_max/2)+1,1)/1000,ii.T[::2])

plt.contourf(x_km,np.arange(-int(w_max/2),0,1)/1000,np.fliplr(ii).T[1::2])
#plt.plot(x_km,(np.array(w_sub)*0.5)/1000, 'k', label='Width subtidal area',linewidth=0.5)
#plt.plot(x_km,(np.array(w_sub)*-0.5)/1000, 'k',linewidth=0.5)
cbar=plt.colorbar()
plt.xlabel('Distance to mouth (km)')
plt.ylabel('Width (km)')
plt.grid(linewidth=0.2)
cbar.ax.set_ylabel('Bed elevation (m)', rotation = 90)
plt.savefig(str(add)+"_Depth_v2.png", dpi = 300)
plt.close()

#cm=plt.cm.get_cmap('jet')
#plt.suptitle('Average flow velocity ('+str(add)+')', fontsize = 14, fontweight = 'bold')
#plt.contourf(x_km,np.arange(0,w_max,1)/1000,u.T,cmap=cm)
#cbar=plt.colorbar()
#plt.xlabel('Distance to mouth (km)')
#plt.ylabel('Width (km)')
#plt.grid(linewidth=0.2)
#cbar.ax.set_ylabel('Flow velocity (m/s)', rotation = 90)
#plt.savefig(str(add)+"_u_v1.png", dpi = 300)
#plt.close()
v=np.linspace(0,1.8,10, endpoint=True)


#cm=plt.cm.get_cmap('jet')
#plt.suptitle('Average flow velocity based on tidal prism ('+str(add)+')', fontsize = 14, fontweight = 'bold')
#if (w_max % 2 == 0): #even 
#    plt.contourf(x_km,np.arange(0,int(w_max/2),1)/1000,u.T[::2],v,cmap=cm)
#else:
#    plt.contourf(x_km,np.arange(0,int(w_max/2)+1,1)/1000,u.T[::2],v,cmap=cm)    
#plt.contourf(x_km,np.arange(-int(w_max/2),0,1)/1000,np.fliplr(u).T[1::2],v,cmap=cm)
#cbar=plt.colorbar(ticks=v)
#plt.xlabel('Distance to mouth (km)')
#plt.ylabel('Width (km)')
#plt.grid(linewidth=0.2)
#cbar.ax.set_ylabel('Flow velocity (m/s)', rotation = 90)
#plt.savefig(str(add)+"_velocity_mean_tp.png", dpi = 300)
#plt.close()

cm=plt.cm.get_cmap('jet')
plt.suptitle('Average flow velocity based on depth ('+str(add)+')', fontsize = 14, fontweight = 'bold')
if (w_max % 2 == 0): #even 
    plt.contourf(x_km,np.arange(0,int(w_max/2),1)/1000,u_h_mean_sorted.T[::2],v,cmap=cm)
else:
    plt.contourf(x_km,np.arange(0,int(w_max/2)+1,1)/1000,u_h_mean_sorted.T[::2],v,cmap=cm)    
plt.contourf(x_km,np.arange(-int(w_max/2),0,1)/1000,np.fliplr(u_h_mean_sorted).T[1::2],v,cmap=cm)
cbar=plt.colorbar(ticks=v)
plt.xlabel('Distance to mouth (km)')
plt.ylabel('Width (km)')
plt.grid(linewidth=0.2)
cbar.ax.set_ylabel('Flow velocity (m/s)', rotation = 90)
plt.savefig(str(add)+"_velocity_mean_h.png", dpi = 300)
plt.close()


#cm=plt.cm.get_cmap('jet')
#plt.suptitle('Maximum flow velocity ('+str(add)+')', fontsize = 14, fontweight = 'bold')
#plt.contourf(x_km,np.arange(0,w_max,1)/1000,u_max_sorted.T, cmap=cm)
#cbar=plt.colorbar()
#plt.legend()
#plt.xlabel('Distance to mouth (km)')
#plt.ylabel('Width (km)')
#plt.grid(linewidth=0.2)
#cbar.ax.set_ylabel('Maximum flow velocity (m/s)', rotation = 90)
#plt.savefig(str(add)+"_u_max_v1.png", dpi = 300)
#plt.close()

#cm=plt.cm.get_cmap('jet')
#p1 = plt.suptitle('Maximum flow velocity based on tidal prism ('+str(add)+')', fontsize = 14, fontweight = 'bold')
#if (w_max % 2 == 0): #even 
#    plt.contourf(x_km,np.arange(0,int(w_max/2),1)/1000,u_max_sorted.T[::2],v,cmap=cm)
#else:
#    plt.contourf(x_km,np.arange(0,int(w_max/2)+1,1)/1000,u_max_sorted.T[::2],v,cmap=cm)
#plt.contourf(x_km,np.arange(-int(w_max/2),0,1)/1000,np.fliplr(u_max_sorted).T[1::2],v,cmap=cm)
#cbar=plt.colorbar(ticks=v)
#plt.legend()
#plt.grid(linewidth=0.2)
#plt.xlabel('Distance to mouth (km)')
#plt.ylabel('Width (km)')
#cbar.ax.set_ylabel('Maximum flow velocity (m/s)', rotation = 90)
#plt.savefig(str(add)+"_velocity_max_tp.png", dpi = 300)
#plt.close()

cm=plt.cm.get_cmap('jet')
p2 = plt.suptitle('Maximum flow velocity based on depth ('+str(add)+')', fontsize = 14, fontweight = 'bold')
if (w_max % 2 == 0): #even 
    plt.contourf(x_km,np.arange(0,int(w_max/2),1)/1000,u_h_max_sorted.T[::2],v,cmap=cm)
else:
    plt.contourf(x_km,np.arange(0,int(w_max/2)+1,1)/1000,u_h_max_sorted.T[::2],v,cmap=cm)
plt.contourf(x_km,np.arange(-int(w_max/2),0,1)/1000,np.fliplr(u_h_max_sorted).T[1::2],v,cmap=cm)
cbar=plt.colorbar(ticks=v)
plt.legend()
plt.grid(linewidth=0.2)
plt.xlabel('Distance to mouth (km)')
plt.ylabel('Width (km)')
cbar.ax.set_ylabel('Maximum flow velocity (m/s)', rotation = 90)
plt.savefig(str(add)+"_velocity_max_h.png", dpi = 300)
plt.close()

if salin==1:
    plt.suptitle('Salinity profile ('+str(add)+')', fontsize = 14, fontweight = 'bold')
    cm = plt.cm.get_cmap('jet')
    #plt.contourf(x_km,np.arange(0,w_max,1)/1000,salinity.T,100,cmap=cm)
    if (w_max % 2 == 0): #even 
        plt.contourf(x_km,np.arange(0,int(w_max/2),1)/1000,salinity.T[::2],150,cmap=cm)
    else:
        plt.contourf(x_km,np.arange(0,int(w_max/2)+1,1)/1000,salinity.T[::2],150,cmap=cm)

    plt.contourf(x_km,np.arange(-int(w_max/2),0,1)/1000,np.fliplr(salinity).T[1::2],150,cmap=cm)
    #plt.plot(x_km,(np.array(w_sub)*0.5)/1000, 'k', label='Width subtidal area',linewidth=0.5)
    #plt.plot(x_km,(np.array(w_sub)*-0.5)/1000, 'k',linewidth=0.5)
    cbar=plt.colorbar()
    plt.legend()
    plt.grid(linewidth=0.2)
    plt.xlabel('Distance to mouth (km)')
    plt.ylabel('Width (km)')
    cbar.ax.set_ylabel('Salinity (ppt)', rotation = 90)
    plt.savefig(str(add)+"_Salinity_profile.png", dpi = 300)
    plt.close()


plt.suptitle('Relative inundation time ('+str(add)+')', fontsize = 14, fontweight = 'bold')
cm=plt.cm.get_cmap('plasma_r')
if (w_max % 2 == 0): #even 
    plt.contourf(x_km,np.arange(0,int(w_max/2),1)/1000,duration.T[::2],10,cmap=cm)
else:
    plt.contourf(x_km,np.arange(0,int(w_max/2)+1,1)/1000,duration.T[::2],10,cmap=cm)
plt.contourf(x_km,np.arange(-int(w_max/2),0,1)/1000,np.fliplr(duration).T[1::2],10,cmap=cm)
plt.grid(linewidth=0.2)
cbar=plt.colorbar()
plt.xlabel('Distance to mouth (km)')
plt.ylabel('Width (km)')
cbar.ax.set_ylabel('Relative duration of inundation (\times tidal period)', rotation = 90)
plt.savefig(str(add)+"_inundation_time.png", dpi = 300)
plt.close()

#%% Combine main plots in one figure

#plt.suptitle(''+str(add)+'', fontsize = 14, fontweight = 'bold')
fig = plt.figure(figsize=(5,4))

gs = gridspec.GridSpec(4,1,height_ratios=[2,2,2,2])
ax = plt.subplot(gs[0])
#plt.subplot(4,1,1)
if (w_max % 2 == 0): #even 
    plt.contourf(x_km,np.arange(0,int(w_max/2),1)/1000,ii.T[::2])
else: #odd
    plt.contourf(x_km,np.arange(0,int(w_max/2)+1,1)/1000,ii.T[::2])

plt.contourf(x_km,np.arange(-int(w_max/2),0,1)/1000,np.fliplr(ii).T[1::2])
#plt.plot(x_km,(np.array(w_sub)*0.5)/1000, 'k', label='Width subtidal area',linewidth=0.5)
#plt.plot(x_km,(np.array(w_sub)*-0.5)/1000, 'k',linewidth=0.5)
cbar=plt.colorbar()
#plt.xlabel('Distance to mouth (km)')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',     # ticks along the bottom edge are o…
    labelbottom=False) # labels along the bottom edge are off
plt.ylabel('Width (km)')
#plt.grid(linewidth=0.2)
cbar.ax.set_ylabel('Bed elevation (m)', rotation = 90)

ax2 = plt.subplot(gs[1])
#plt.subplot(4,1,2)
cm=plt.cm.get_cmap('plasma_r')
if (w_max % 2 == 0): #even 
    plt.contourf(x_km,np.arange(0,int(w_max/2),1)/1000,duration.T[::2],10,cmap=cm)
else:
    plt.contourf(x_km,np.arange(0,int(w_max/2)+1,1)/1000,duration.T[::2],10,cmap=cm)
plt.contourf(x_km,np.arange(-int(w_max/2),0,1)/1000,np.fliplr(duration).T[1::2],10,cmap=cm)
#plt.grid(linewidth=0.2)
cbar=plt.colorbar()
#plt.xlabel('Distance to mouth (km)')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',     # ticks along the bottom edge are o…
    labelbottom=False) # labels along the bottom edge are off
plt.ylabel('Width (km)')
cbar.ax.set_ylabel('Inundation (*T)', rotation = 90)

ax3 = plt.subplot(gs[2])
#plt.subplot(4,1,3)
cm=plt.cm.get_cmap('Blues')
if (w_max % 2 == 0): #even 
    plt.contourf(x_km,np.arange(0,int(w_max/2),1)/1000,u_h_max_sorted.T[::2],v,cmap=cm)
else:
    plt.contourf(x_km,np.arange(0,int(w_max/2)+1,1)/1000,u_h_max_sorted.T[::2],v,cmap=cm)
plt.contourf(x_km,np.arange(-int(w_max/2),0,1)/1000,np.fliplr(u_h_max_sorted).T[1::2],v,cmap=cm)
cbar=plt.colorbar(ticks=v)
plt.legend()
#plt.grid(linewidth=0.2)
#plt.xlabel('Distance to mouth (km)')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',     # ticks along the bottom edge are o…
    labelbottom=False) # labels along the bottom edge are off
plt.ylabel('Width (km)')
cbar.ax.set_ylabel('Peak flow (m/s)', rotation = 90)

v2=np.linspace(0,35,8, endpoint=True)
ax4 = plt.subplot(gs[3])
#plt.subplot(4,1,4)
if salin==1:
    cm = plt.cm.get_cmap('YlOrRd')
    #plt.contourf(x_km,np.arange(0,w_max,1)/1000,salinity.T,100,cmap=cm)
    if (w_max % 2 == 0): #even 
        plt.contourf(x_km,np.arange(0,int(w_max/2),1)/1000,salinity.T[::2],150,cmap=cm)
    else:
        plt.contourf(x_km,np.arange(0,int(w_max/2)+1,1)/1000,salinity.T[::2],150,cmap=cm)

    plt.contourf(x_km,np.arange(-int(w_max/2),0,1)/1000,np.fliplr(salinity).T[1::2],150,cmap=cm)
    #plt.plot(x_km,(np.array(w_sub)*0.5)/1000, 'k', label='Width subtidal area',linewidth=0.5)
    #plt.plot(x_km,(np.array(w_sub)*-0.5)/1000, 'k',linewidth=0.5)
    cbar=plt.colorbar(ticks=v2)
    plt.legend()
    #plt.grid(linewidth=0.2)
    plt.xlabel('Distance to mouth (km)')
    plt.ylabel('Width (km)')
    cbar.ax.set_ylabel('Salinity (ppt)', rotation = 90)

plt.savefig(str(add)+"_overview.png", dpi = 300)
plt.close()

direc=os.getcwd()

#%%

os.chdir('..')                      #return to the original (starting) working directory
print("--- %s seconds for creating graphs---" % (time.time() - start_time4))
print("--- %s seconds for the total run---" % (time.time() - start_time))
print('The model run has now finished. The results can be found in the folder "Results", ('+str(direc)+')')
