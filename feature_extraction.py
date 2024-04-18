import numpy as np
import scipy.optimize as so

import backus_utils as utils

# # Feature Extraction

def normal(x,A,mu,sigma):
    return A*np.exp(-0.5*((x-mu)/sigma)**2)

def get_peaks(x,y):
    """Returns the peaks of a function y=f(x)."""
    
    dy = np.gradient(y)
    
    locations = []
    index = []
    
    for pos,loc in enumerate(x):
        if pos < len(x)-1:
            if dy[pos]*dy[pos+1] <= 0 and y[pos] > 0:
                locations.append(loc)
                index.append(pos)
                
    return np.asarray(index),np.asarray(locations)

def main(rho,err,w,channel=None,Nt=None,gen2l_path=None,return_popt=False,w_feat=None,method="leading",fitfunc=normal,return_info=False,upper_fit=True):
    """Extract the ground state feature for some spectrum rho with error err.
    
    Arguments:
    
    rho -- the spectrum
    err -- the error in the spectrum
    w -- the energy (x-axis) corresponding to the spectrum
    channel -- the channel for determining the feature location (if not specified)
    Nt -- lattice extent
    gen2l_path -- the path to the gen2l configs
    return_popt -- flag to directly return fitting parameters
    w_feat -- feature location for fit
    method -- fitting method for Gaussian (leading, trailing or full fits)
    fitfunc -- fitting function, default is Gaussian
    return_info -- return information corresponding to extracted feature
    upper_fit -- flag to toggle fitting to upper half of spectrum. If float, fits upper nth fraction.
    """
    
    if w_feat == None and channel == None:
        raise Exception("Must specify either w_feat or channel")
        
    elif channel != None:
        # Estimate w_feat from effective mass
        mean, _, _ = utils.get_sample(gen2l_path,channel,n_samples=-1,Nt=Nt)
        
        meff = -np.gradient(np.log(mean))
        w_feat = meff[-1]
        
    
    #get values of w and indices corresponding to peaks
    peak_index, peaks = get_peaks(w,rho)
    
    # calculate distance between peaks and the feature point
    peak_distance = abs(w_feat-w[peak_index])      
    feat_index = np.where(peak_distance == min(peak_distance))[0] # Location of the ground state
       
    if len(feat_index) > 1:
        feat_index = feat_index[0]
    
    peak_loc = peak_index[feat_index]
    
    if feat_index - 1 < 0:
        previous_peak = 0
    else:
        previous_peak = peak_index[feat_index-1]
    
    try:
        next_peak = peak_index[feat_index + 1]
    except IndexError:
        next_peak = len(w)-1
    
    if method == "full":
        peak_mask = np.logical_and(w > w[previous_peak], w <= w[next_peak])
    elif method == "leading":
        peak_mask = np.logical_and(w > w[previous_peak],w <= w[peak_loc])
    elif method == "trailing":
        peak_mask = np.logical_and(w > w[peak_loc], w <= w[next_peak])
    
    if upper_fit == True:
        mask = np.logical_and(rho > 0.5*rho[peak_loc],peak_mask)  # Fitting upper half
    elif isinstance(upper_fit, float):
        mask = np.logical_and(rho > (1-upper_fit)*rho[peak_loc],peak_mask)  # Fitting upper fraction
    else:
        mask = peak_mask # Do not curtail fit 
    
    if err is None:
        sigma = None
    elif len(np.shape(err)) > 1:
        sigma = err[mask,mask]
    else:
        sigma = err[mask]
    
    popt,pcov = so.curve_fit(fitfunc,w[mask],rho[mask],sigma=sigma,absolute_sigma=True,maxfev=1200)

    if popt[2] > max(w)-min(w):
        popt = [-1,-1,-1]
        raise Exception("Estimated width exceeds reconstruction window.")
    if popt[1] > max(w) or popt[1] < min(w):
        popt = [-1,-1,-1]
        raise Exception("Estimated mass outside reconstruction window.")

    # # Fit Results

    mu = popt[1]
    FWHM = 2.355*abs(popt[2])

    mu_err = np.sqrt(np.diag(pcov))[1]
    FWHM_err = 2.355*np.sqrt(np.diag(pcov))[2]
    
    peak = rho[peak_loc]
    prev_peak = rho[previous_peak]

    if peak > 2*prev_peak:
        weight = 1 #accept fit
    else:
        weight = 0 #reject fit
        
    output = []
        
    if return_popt:
        output.extend([popt,pcov])
    if return_info:
        output.extend([mask])
        
    if return_popt or return_info:
        return output
    else:
        return mu,mu_err,FWHM,FWHM_err,weight
