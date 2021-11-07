#!/usr/bin/env python3

'''
imfit-MCM
---------
Author:Thato Manamela  
Version: 2021

-------------------------------------------------------------------------------------------------
This code samples a 2D Gaussian function to accurately estimate the flux density and source size
parameters of stacked radio galaxies by implementing MCMC (Emcee software Package). 
-------------------------------------------------------------------------------------------------
'''

import numpy as np
import matplotlib
import matplotlib as mpl
import os
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pylab as plt  
import astropy.io.fits as pf
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS

import scipy.optimize as opt
import emcee
from multiprocessing import Pool, cpu_count
import functions as ft
import pandas  as pd
import configparser
import sys



config = configparser.ConfigParser()
config.read(sys.argv[-1])

PATH = config.get('pipeline', 'data_path') 
image, header =pf.getdata(PATH+config.get('flux_Lumi_SFR_culculations', 'stackedimage'), header = True)
image = image[0,0,:,:]
wcs = WCS(header).celestial
wcs.celestial 
mean, median, std = sigma_clipped_stats(image, sigma=3.0)
sigma = std

try:
    pixscale = header['CDELT2']*3600 # arcsec
except KeyError:
    pixscale = header['CD2_2']*3600 # arcsec



class PreMCMC_fit(object):
    """
    Predicts initial 2D Gaussian paramters that will later be used in an MCMC fit
    
    Parameter
    --------
    data : 2D array
         Image Data array to be fitted.
    """
    
    def __init__(self, data):
        self.data = data
    
    
    def twoD_Gaussian_model(self, xdata, amplitude, xo, yo, sigma_x, sigma_y, offset):
        """
        Function to fit, returns 2D gaussian function as 1D array
        """
        (x, y) = xdata
        xo = float(xo)
        yo = float(yo)     
        g = offset + amplitude*np.exp( - (((x-xo)**2)/(2*sigma_x**2) + ((y-yo)**2)/(2*sigma_y**2)))                                   
        return g.ravel()

    def pre_mcmc_fit(self):
        """
        Get Parameters of a blob by 2D gaussian fitting
        
        Returns: 
           amplitude, xo, yo, sigma_x, sigma_y, PA, offset.
        """
        xx = np.linspace(0, self.data.shape[1], self.data.shape[1])
        yy = np.linspace(0, self.data.shape[0], self.data.shape[0])
        x, y = np.meshgrid(xx, yy)
        xdata = np.vstack((x.ravel(),y.ravel()))
        #Parameters: amp, xpos, ypos, sigmaX, sigmaY, PA, offset
        initial_guess = (1, self.data.shape[1]/2, self.data.shape[0]/2, 10, 10, 1)
        popt, pcov = opt.curve_fit(self.twoD_Gaussian_model, xdata, 
                                   self.data.ravel(), p0=initial_guess)
        return popt



class Gaussian_Model(object):
    """
    Dinfines the Likelihood function, priors on parameters 
    and the Posterior function for MCMC.
    
    Parameters
    ----------
    data : 2D array 
         Image Data array to be fitted.
    sigma : float
          the standard deviation of the data points or the image RMS
    """
    
    def __init__(self,data, sigma):
        self.data = data
        self.sigma = sigma
    
    def twoD_Gaussian(self, image_data, amplitude, xo, yo, sigma_x, sigma_y, offset):
        """
        Function to fit, returns 2D gaussian function as 1D array
        """
        x     = np.arange(0, image_data.shape[0])
        y     = np.arange(0, image_data.shape[1])
        (x, y) = np.meshgrid(x,y)
        xo = float(xo)                                                              
        yo = float(yo)                                                              
        g = offset + amplitude*np.exp( - (((x-xo)**2)/(2*sigma_x**2) + ((y-yo)**2)/(2*sigma_y**2)))                                   
        return g.ravel()

    def loglikelihood(self,theta):
        """
        The natural logarithm of the joint likelihood.

        Parameter
        --------
        theta : tuple
              a sample containing individual parameter values
        """

        # unpack the model parameters from the tuple
        amplitude, xo, yo, sigma_x, sigma_y, offset = theta
        
        # evaluate the model
        
        md = self.twoD_Gaussian(self.data, amplitude, xo, yo, sigma_x, sigma_y, offset)

        # return the log likelihood
        return -0.5*np.sum(((md - self.data.ravel())/self.sigma)**2)

    def logprior(self,theta):
        """
        The natural logarithm of the prior probability.

        Parameter
        --------
        theta : tuple
              a sample containing individual parameter values
        """

        # unpack the model parameters from the tuple
        amplitude, xo, yo, sigma_x, sigma_y, offset = theta
        
        if (amplitude > 0) & (0 < xo < self.data.shape[0]) & (0 < yo < self.data.shape[1]) & (sigma_x > 0) & (sigma_y > 0):
            return  0.0         # set prior to 1 (log prior to 0) if in the range and zero (-inf) outside the range 


        return -np.inf

    def logposterior(self,theta):
        """
        The natural logarithm of the joint posterior.

        Parameter
        --------
        theta : tuple
              a sample containing individual parameter values
        """

        lp = self.logprior(theta) # get the prior

        # if the prior is not finite return a probability of zero (log probability of -inf)
        if not np.isfinite(lp):
            return -np.inf

        # return the likeihood times the prior (log likelihood plus the log prior)
        return lp + self.loglikelihood(theta)
    
    def __call__(self, theta):
        return self.logposterior(theta)

    
def main_mcmc(p0, ndim, lnprob, nwalkers = 12 , niter = 500):
    """
    Run the MCMC sampler.

    Parameters
    ----------
    p0 : array
       intial parameters determined from the PreMCMC-fit class
    ndim : int
         the number of parameters being fitted for.
    lnprob : object
           The Posterior probability function 
    nwalkers: int
            Required integer number of walkers to use in ensemble.
    niter: int
        Number of steps for walkers. Typically at least a few hundreds (but depends on dimensionality).
        Low nrun (<100?) will underestimate the errors.
    """
    num_CPU = cpu_count()
    with Pool(num_CPU) as pool: # Threading the MCMC for faster execution ...
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)

        print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, 500)
        sampler.reset()

        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter)

    return sampler, pos, prob, state


# PreMCMC-fit:
popt = PreMCMC_fit(image) 
initial =popt.pre_mcmc_fit() # inintial paramters

# The actual MCMC:
ndim = len(initial)
nwalkers = 240
p0 = [initial*(1.+1.e-3*np.random.randn(ndim)) for i in range(nwalkers)]
sampler, pos, prob, state = main_mcmc(p0, ndim, Gaussian_Model(image,sigma), nwalkers, niter = 1200) 
#----------------------
# Plot of MCMC results:
#----------------------
# Corner plot for each parameter explored by the walkers.


ft.corner_plot(sampler, pixscale)

# Trace plot
ft.plot_trace(sampler, nwalkers, pixscale)

# data - model = residual data plot
ft.data_model_residual_comparisson_plot(image, header, wcs, sigma, sampler)


#----------------------
# Saving results to file:
#----------------------
# Converting peak flux to Integrated, sigmaX, sigmaY FWHM_y, FWHM_x:

FWHM_x = np.percentile((2.*sampler.flatchain[:,3]*(2.*np.log(2.))**0.5)*pixscale, 50) # Minor-axis in arcsec
FWHM_y = np.percentile((2.*sampler.flatchain[:,4]*(2.*np.log(2.))**0.5)*pixscale, 50) #  Major-axis in arcsec

FWHM_x_err = ((np.percentile((2.*sampler.flatchain[:,3]*(2.*np.log(2.))**0.5)*pixscale, 84)-np.percentile((2.*sampler.flatchain[:,3]*(2.*np.log(2.))**0.5)*pixscale, 16))/2.) # Minor-axis in arcsec
FWHM_y_err = ((np.percentile((2.*sampler.flatchain[:,4]*(2.*np.log(2.))**0.5)*pixscale, 84)-np.percentile((2.*sampler.flatchain[:,4]*(2.*np.log(2.))**0.5)*pixscale, 16))/2.) # Minor-axis in arcsec

Int_flux = np.percentile(ft.Integrated_flux(image,header,sigma, sampler)*1e3, 50) #mJy

Int_flux_err = ((np.percentile(ft.Integrated_flux(image,header,sigma, sampler)*1e3, 84)-np.percentile(ft.Integrated_flux(image,header,sigma, sampler)*1e3, 16))/2.) #mJy

results_dictionary = {'Int_flux (mJy)': Int_flux, 'Int_flux_err (mJy)': Int_flux_err, 'Minor (arcsec)': FWHM_x, 
                      'Minor_err (arcsec)': FWHM_x_err , 'Major (arcsec)': FWHM_y ,'Major_err (arcsec)': FWHM_y_err, 
                      'size (arcsec^2)': FWHM_y*FWHM_x , 'size_err (arcsec^2)': FWHM_y_err*FWHM_x_err}

df = pd.DataFrame(results_dictionary, index=[0])
df.to_csv(PATH+'fit_results.csv')