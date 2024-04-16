"""

Backus-Gilbert utils script.

"""

import numpy as np
import pandas as pd
import os
import gvar as gv

import matplotlib as mpl

dons_channels = ['spp_0','spp_i','sxx_0','sxx_i','pp_i','xx_i','pp_A','xx_A']
symbs = ['$\eta_B$ (l-l)','$\\Upsilon$ (l-l)','$\eta_B$ (s-s)','$\\Upsilon$ (s-s)',
         '$\chi_{b1}$ (l-l)','$\chi_{b1}$ (s-s)','$h_b$ (l-l)','$h_b$ (s-s)']

symbs2 = ['$\eta_B$','$\\Upsilon$','$\eta_B$','$\\Upsilon$ ',
         '$\chi_{b1}$','$\chi_{b1}$','$h_b$ ','$h_b$']

__Tc__ = gv.gvar(167,3)  #Tc in MeV
__escale__ = gv.gvar(6.079,0.013)  #1/a_t in GeV
__eshift__ = gv.gvar(7.463,0)  #NRQCD e-shift in GeV


def to_a(value,scale=__escale__,shift=__eshift__):
    """Converts a value in GeV to lattice units."""
    
    return (np.asarray(value)-shift)/scale

def to_gev(value,scale=__escale__,shift=__eshift__):
    """Converts a value in lattice units to GeV."""
    
    return (np.asarray(value)*scale)+shift

def get_channel(path_to_file, channel, Nt):
    """Extracts correlator data from a correlator output file."""
    
    confs = open(path_to_file, 'r')
    
    raw = confs.readlines()
    
    # Get list of available channels
    channels = [raw[i*(Nt+1)].rstrip('\n') for i in range(len(raw)%Nt)]
    
    # Calculate position in configuration file
    step = channels.index(channel)
    
    nmin = step * (Nt+1) + 1
    nmax = nmin + Nt
    
    data = raw[nmin:nmax]

    return np.asarray(data, dtype = np.float64)

def get_valid_selection(path_to_confs, channel):
    """Fetches the valid filenames in path_to_confs given cstring."""
   
    filenames = os.listdir(path_to_confs)
    valid_names = []
    
    for name in filenames:
        if name.startswith(f"{channel[0]}onia"):
            valid_names.append(name)
            
    return valid_names

def get_sample(path_to_confs, channel, n_samples, Nt, t1=0, t2='Nt', return_samples=False):
    """Fetches the mean and stdev of n_samples of the correlator configs.
    
    Args:
        path_to_confs: The path to the Gen2l configurations
        cstring:       Binary string form of the channel (i.e. 's01' for spp_i, etc.)
        n_samples:     Number of bootstrap samples
        Nt:            Temporal lattice size
        return_samples: (bool) Return samples selected
    """

    if t2 == 'Nt':
        t2 = Nt
    
    # Get the names of all valid configuration files
    valid_names = get_valid_selection(path_to_confs, channel)
   
    if n_samples == -1: # Select all valid configs
        choices = valid_names
    else:
        choices = np.random.choice(valid_names, n_samples)
    
    # Collect samples
    samples = []
    for cfile in choices:
        path = f"{path_to_confs}/{cfile}"
        
        samples.append(get_channel(path, channel, Nt)[t1:t2])
        
    mean = np.mean(samples,axis=0)
    std = np.std(samples,axis=0)
    cov = np.cov(np.asarray(samples).T)
      
    if return_samples:
        return mean,std,cov,samples
    else:
        return mean,std,cov

def get_covmat(path_to_outdat):
    """Extracts the covariance matrix from an `output.dat` BG input file."""
    
    Nt = int(pd.read_csv(path_to_outdat,sep=',',header=None,nrows=1).values[0])
    
    covmat = np.zeros((Nt,Nt))
    cov = pd.read_csv(path_to_outdat,sep=',',header=None,skiprows=2+Nt).values[:]

    counter = Nt
    for i in range(Nt):
        covmat[i,i] = cov[i]
    
    emode = 1
    if emode == 1:
        for i in range(Nt):
            for j in range(i+1,Nt):
                covmat[i,j] = covmat[j,i] = cov[counter]
                counter += 1
                
    return covmat

def get_zeros(x,y):
    """Returns the zero-crossings (turning-points) of a function y=f(x)."""
    
    locations = []
    index = []
    
    for pos,loc in enumerate(x):
        if pos < len(x)-1:
            if y[pos]*y[pos+1] <= 0:
                locations.append(loc)
                index.append(pos)
                
    return np.asarray(index),np.asarray(locations)

def is_peak(loc,y):
    """Determines whether y[loc] is a peak."""
    return np.gradient(np.gradient(y))[np.asarray(loc)] < 1