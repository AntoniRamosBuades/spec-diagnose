import os
import sys
import glob
import re
import h5py
import numpy as np

from segment_utils import FindLatestSegments, LoadH5_from_segments
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline


#Identical to ImportRun but loading also the waves, which makes it much slower
def ImportRunAndWaves(path_to_ev, Lev, tmin=-1e10, tmax=1e10):
    """ImportRun

Load some important files for a certain Ev/Lev*, and populate a
dictionary with the imported data as follows
"""
    D={}
    segs,tstart,termination=FindLatestSegments(path_to_ev,Lev, tmin=tmin, tmax=tmax)
    D['segs']=segs
    D['tstart']=tstart
    print("tstart={}".format(tstart))
    D['termination']=termination
    D['Horizons']   =LoadH5_from_segments(segs,"ApparentHorizons/Horizons.h5")
    D['AhA']=       LoadDat_from_segments(segs,"ApparentHorizons/AhA.dat")
    D['AhB']=       LoadDat_from_segments(segs,"ApparentHorizons/AhB.dat")
    D['AdjustGrid']=LoadH5_from_segments(segs,"AdjustGridExtents.h5")
    D['rh_FiniteRadii_CodeUnits'] = LoadH5_from_segments(segs,"GW2/rh_FiniteRadii_CodeUnits.h5")
    D['GhCeLinf'] = LoadDat_from_segments(segs,"ConstraintNorms/GhCe_Linf.dat")
    D['sep']      = LoadDat_from_segments(segs,"ApparentHorizons/HorizonSepMeasures.dat")
    return D


# compute the frequency from the waveform (numpy array) using the simplest finite difference approach
# The outcome is an array of length = len(hlm)-1
def compute_freq(hlm, deltaT):
    freq = np.diff(np.unwrap(np.angle(hlm)))/deltaT

    return freq


# compute the frequency from the waveform  using the scipy spline interpolation (higher order method than the previous one)
# The outcome is an array of length = len(hlm)
def compute_freqInterp(time, hlm):

    philm=np.unwrap(np.angle(hlm))

    intrp = InterpolatedUnivariateSpline(time,philm)
    omegalm = intrp.derivative()(time)

    return omegalm



# function to parse the modes, compute amplitude and phase of the outermost extraction radius
# Example format ModeList:   ModeList=[[2, 2], [2, 1], [3, 1], [3, 2], [3, 3], [4, 1], [4, 2], [4, 3], [4, 4],  [5, 5], [6, 6],  [7, 7],  [8, 8]]
# posradius : position of the extraction radius we want to look at
# trelax :  relaxation time. This time will be skipped from the start of the waveform.
def readModes(Lev3, ModeList, posradius, trelax=150, align = True):

    radii = list(Lev3['rh_FiniteRadii_CodeUnits'].keys())
    #print(radii)
    radius=radii[posradius]
    print("Extract waveforms from the outermost extraction radius R = ",radius)

    #trelax=300.0 # Set some arbitrary buffer of time to avoid junk radiation

    hlm={}
    amplm={}
    phaselm={}
    omegalm={}

    for l,m in ModeList:

        data = Lev3['rh_FiniteRadii_CodeUnits'][radius]['Y_l'+str(l)+'_m'+str(m)]
        dataKeys = list(data.keys())


        tdiff = float(radius.replace('R', ''))+trelax

        time = data[dataKeys[0]][:,0]

        # apply relaxation time
        shiftedInd = np.where(time>time[1]+tdiff)[0]
        time=time[shiftedInd]

        dt=time[2]-time[1]

        rehlm = data[dataKeys[1]][:,1]
        imhlm = data[dataKeys[2]][:,1]
        hlm[l,m]= rehlm - 1j*imhlm
        hlm[l,m] = hlm[l, m][shiftedInd]

        #omegalm[l,m] = compute_freq(hlm[l,m],dt)

        # compute phase, amplitude and frequencies
        phaselm[l,m] = np.unwrap(np.angle(hlm[l,m]))
        omegalm[l,m] = compute_freqInterp(time, hlm[l,m])
        amplm[l,m] = np.abs(hlm[l,m])

        iAmax, Amax = np.argmax(amplm[2,2]), np.max(amplm[2,2])
        tAmax = time[iAmax]

        if align == True:
            time -= tAmax # Set peak to t=0

    return time, hlm, amplm, omegalm, phaselm


# Function to compute the orbital phase and frequency from AHs of the simulation

def compute_OrbPhaseOmega(Lev3):

    timeDyn = Lev3['Horizons']['AhA']['CoordCenterInertial']['x'][:,0]
    xA = Lev3['Horizons']['AhA']['CoordCenterInertial']['x'][:,1]
    yA = Lev3['Horizons']['AhA']['CoordCenterInertial']['y'][:,1]
    zA = Lev3['Horizons']['AhA']['CoordCenterInertial']['z'][:,1]

    xB = Lev3['Horizons']['AhB']['CoordCenterInertial']['x'][:,1]
    yB = Lev3['Horizons']['AhB']['CoordCenterInertial']['y'][:,1]
    zB = Lev3['Horizons']['AhB']['CoordCenterInertial']['z'][:,1]

    x = xA - xB
    y = yA - yB
    z = zA -zB
    #rvec = np.concatenate((x,y,z),axis=None)

    # Multiply by 2 the argument of unwrap and divide the outcome by two as suggested in https://stackoverflow.com/questions/52293831/unwrap-angle-to-have-continuous-phase
    phaseorb = np.unwrap(2*np.arctan(y/x))/2

    intrp_phase = InterpolatedUnivariateSpline(timeDyn, phaseorb)
    omegaorb  = intrp_phase.derivative()(timeDyn)

    return timeDyn, phaseorb, omegaorb

# Function to calculate the difference between Levs of the orbital frequency and orbital phase

def phaseOmegaOrbdiff(time_common, timeDyn_Lev1, phaseorb_Lev1, timeDyn_Lev2, phaseorb_Lev2):


    intrp_Lev1 = InterpolatedUnivariateSpline(timeDyn_Lev1, phaseorb_Lev1)
    intrp_Lev2 = InterpolatedUnivariateSpline(timeDyn_Lev2, phaseorb_Lev2)

    phaseorbdiff = intrp_Lev2(time_common)-intrp_Lev1(time_common)
    intrp_phaseorbdiff = InterpolatedUnivariateSpline(time_common, phaseorbdiff)
    omegaorbdiff = intrp_phaseorbdiff.derivative()(time_common)

    return phaseorbdiff, omegaorbdiff


# Function to calculate the difference between Levs of the phase and frequencies of the modes

def phaseOmegalmdiff(time_common, time_Lev1, phaselm_Lev1, time_Lev2, phaselm_Lev2):

    phaselmdiff = {}
    omegalmdiff = {}

    assert( len(list(phaselm_Lev1.keys())) == len(list(phaselm_Lev2.keys()))),"Levs have different mode arrays!"


    for l,m in list(phaselm_Lev1.keys()):


        intrp_Lev1 = InterpolatedUnivariateSpline(time_Lev1, phaselm_Lev1[l,m])
        intrp_Lev2 = InterpolatedUnivariateSpline(time_Lev2, phaselm_Lev2[l,m])

        phaselmdiff[l,m] = intrp_Lev2(time_common)-intrp_Lev1(time_common)

        intrp_phaselmdiff = InterpolatedUnivariateSpline(time_common, phaselmdiff[l,m])
        omegalmdiff[l,m] = intrp_phaselmdiff.derivative()(time_common)


    return phaselmdiff, omegalmdiff
