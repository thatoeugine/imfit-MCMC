#!/usr/bin/env python3

'''
Author: Thato Manamela
'''
import numpy as np
import corner
import matplotlib.pylab as plt
import astropy.units as u
import matplotlib
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
from astropy.coordinates import SkyCoord
from matplotlib.patches import Ellipse
import configparser
import sys
import seaborn as sns

sns.set_style("whitegrid")


params = {'legend.fontsize': 12,
          'axes.labelsize': 15,
          'axes.titlesize': 15,
          'axes.linewidth': 0.5,
          'axes.edgecolor': '.15',
          'xtick.labelsize' :15,
          'ytick.labelsize': 15,
          'mathtext.fontset': 'cm',
          'mathtext.rm': 'serif',
          'mathtext.bf': 'serif:bold',
          'mathtext.it': 'serif:italic',
          'mathtext.sf': 'sans\\-serif',
         }
matplotlib.rcParams.update(params)





config = configparser.ConfigParser()
config.read(sys.argv[-1])

PATH = config.get('pipeline', 'data_path') 





def corner_plot(sampler, pixscale, plotname):
    """
    Corner plot for each parameter explored by the walkers.
    """
    labels = ['$I_0$ [$\mu$Jy/beam]', '$X_0$ [arcsec]','$Y_0$ [arcsec]', 'sigmaX [arcsec]', 'sigmaY [arcsec]', 'offset [$\mu$Jy/beam]']

    posterior_samples = np.array([sampler.flatchain[:,0]*1e6,sampler.flatchain[:,1]*pixscale, 
                      sampler.flatchain[:,2]*pixscale,sampler.flatchain[:,3]*pixscale,
                      sampler.flatchain[:,4]*pixscale, sampler.flatchain[:,5]*1e6])          
    fig = corner.corner(posterior_samples.T, labels=labels,show_titles=True,
                    label_kwargs={"fontsize": 13}, title_kwargs={"fontsize": 13}, quantiles=[0.16, 0.5, 0.84]);
    fig.savefig(plotname, dpi = 400)
    plt.clf()
    plt.close(fig)



def plot_trace(sampler, nwalkers, pixscale, plotname):
    """
    Plot the trace of walkers for every steps
    """
    labels = ['$I_0$ [$\mu$Jy/beam]', '$X_0$ [arcsec]','$Y_0$ [arcsec]', 'sigmaX [arcsec]', 'sigmaY [arcsec]', 'offset [$\mu$Jy/beam]']

    multiplier = [1e6, pixscale, pixscale, pixscale, pixscale, 1e6]

    fig, ax = plt.subplots(len(labels), sharex=True)
    ax[0].set_title("Number of walkers: "+str(nwalkers), fontsize=20)
    for i in range(len(ax)):
        ax[i].plot(sampler.chain[:, :, i].T*multiplier[i], "-k", alpha=0.2)
        ax[i].set_ylabel(labels[i])
        fig.set_size_inches(2*10,15)
        ax[i].tick_params(labelsize=15)
    
    plt.xlabel("steps", fontsize=20)
    plt.savefig(plotname, dpi = 400)
    plt.clf()
    plt.close(fig)


def twoD_Gaussian(image_data, amplitude, xo, yo, sigma_x, sigma_y, offset):
        """
        Function to fit, returns 2D gaussian function as 1D array
        """
        x     = np.arange(0, image_data.shape[0])
        y     = np.arange(0, image_data.shape[1])
        (x, y) = np.meshgrid(x,y)
        xo = float(xo)                                                              
        yo = float(yo)                                                              
        g = offset + amplitude*np.exp( - (((x-xo)**2)/(2*sigma_x**2) + ((y-yo)**2)/(2*sigma_y**2)))                                   
        return g


def data_model_residual_comparisson_plot(image, header, wcs, sigma, sampler, plotname):
    """
    data - model = residual data plot
    Parameters:
    ----------
    image : array
            Image data array
    header : Astropy header of the input image
    sigma : float
            the standard deviation of the data points or the image RMS
    sampler : array
            MCMC sampler
    """

    # calculating median of parameters
    bestamp = np.median(sampler.flatchain[:,0]) 
    bestx0 = np.median(sampler.flatchain[:,1]) 
    besty0 = np.median(sampler.flatchain[:,2]) 
    bestsigx = np.median(sampler.flatchain[:,3])
    bestsigy = np.median(sampler.flatchain[:,4]) 
    bestoffset = np.median(sampler.flatchain[:,5]) 
    bestmodel = twoD_Gaussian(image, bestamp, bestx0, besty0, bestsigx, bestsigy, bestoffset) # generate median model

    #Creating 1 row and 3 columns grid
    gs = matplotlib.gridspec.GridSpec(1, 3) 
    fig = plt.figure(figsize=(38,9))

    '''Plot 1'''
    ax1=plt.subplot(gs[0,0], projection=wcs)
    im1 = ax1.imshow(image*1e6,cmap='Spectral_r', origin='lower', vmax = image.max()*1e6, vmin = image.min()*1e6)
    ax1.set_xlabel('RA(J2000)', fontsize = 20)
    ax1.set_ylabel('DEC(J2000)', fontsize = 20)
    cbar1 = plt.colorbar(im1, pad=0.0)
    cbar1.set_label(label = 'Flux Density  ($\mu$Jy/beam)', fontsize = 20)
    cbar1.ax.tick_params(labelsize=18)
    ax1.contour(image*1e6, colors='white',levels = [sigma,2*sigma,3*sigma,4*sigma,5*sigma],linewidths= 0.4)
    ax1.set_title("Image", fontsize = 20, weight = 'bold')
    ax1.tick_params(labelsize=20, axis = 'both') 


    
    #Beam size
    def draw_ellipse(ax, width, height, angle):
        """
        Draw an ellipse of the size of the Telescope's Primary Beam
        """
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredEllipse
        ae = AnchoredEllipse(ax.get_transform('fk5'), width, height, angle,
                             loc='lower left', pad=0.5, borderpad=0.4, frameon=True)

        ax.add_artist(ae)

    bmaj, bmin, angle = header['BMAJ'], header['BMIN'], header['BPA']

    draw_ellipse(ax1, bmaj, bmin, angle)

    '''Plot 2'''
    ax2=plt.subplot(gs[0,1], projection=wcs)
    im2 = ax2.imshow(bestmodel*1e6,cmap='Spectral_r', origin='lower')
    ax2.set_xlabel('RA(J2000)',  fontsize = 20)
    ax2.set_ylabel('DEC(J2000)',  fontsize = 20)
    cbar2 = plt.colorbar(im2, pad=0.0)
    cbar2.set_label(label = 'Flux Density  ($\mu$Jy/beam)', fontsize = 20)
    cbar2.ax.tick_params(labelsize=20)
    ax2.set_title("Model", fontsize = 20, weight = 'bold')
    ax2.tick_params(labelsize=20, axis = 'both') 



    '''Plot 3'''
    ax3=plt.subplot(gs[0,2], projection=wcs)
    im3 = ax3.imshow((image-bestmodel)*1e6,cmap='PuOr_r', origin='lower')
    ax3.set_xlabel('RA(J2000)', fontsize = 20)
    ax3.set_ylabel('DEC(J2000)', fontsize = 20)
    cbar3 = plt.colorbar(im3, pad=0.0)
    cbar3.set_label(label = 'Flux Density  ($\mu$Jy/beam)', fontsize = 20)
    cbar3.ax.tick_params(labelsize = 20)
    ax3.contour((image-bestmodel)*1e6, colors = 'black',levels = [sigma,2*sigma,3*sigma,4*sigma,5*sigma],linewidths = 0.4)
    ax3.set_title("Residual", fontsize = 20, weight = 'bold')
    ax3.tick_params(labelsize=20, axis = 'both') 
    
    plt.savefig(plotname, dpi = 400)
    

    
def Integrated_flux(image, header, sigma, sampler):
    """
    Computes the total flux density of a detection
    
    Parameters:
    ----------
    image : 2D array
            Image data array
    header : Astropy header of the input image
    sigma : float
            the standard deviation of the data points or the image RMS
    sampler : array
            MCMC sampler
    
    returns the image total flux density/integrated in Jy

    """
    pixscale = abs(header['CDELT1'])*u.deg
    bmaj      = (2.*sampler.flatchain[:,4]*(2.*np.log(2.))**0.5)*pixscale
    bmin      = (2.*sampler.flatchain[:,3]*(2.*np.log(2.))**0.5)*pixscale

    beammaj        = bmaj/(2.*(2.*np.log(2.))**0.5) # Convert to sigma
    beammin        = bmin/(2.*(2.*np.log(2.))**0.5) # Convert to sigma
    pix_area  = abs(header['CDELT1']*header['CDELT2'])*u.deg*u.deg
    beam_area = 2.*np.pi*1.0*beammaj*beammin
    beam2pix  = beam_area/pix_area
    
    Fluxsum = []
    for i in range(len(image.ravel())):
        if image.ravel()[i] >= sigma: # count Fluxes that are above the noise level simga
            Fluxsum.append(image.ravel()[i])
    Int_Flux = np.sum(Fluxsum)/beam2pix # Source https://www.eaobservatory.org/jcmt/faq/how-can-i-convert-from-mjybeam-to-mjy/
    return Int_Flux.value
