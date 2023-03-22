# imfit-MCMC

## Introduction 
This software is created to automate the flux density, and source size of stacked radio galaxies. This is done by fitting the radio stacked image to a two dimensional (2D) Gaussian model using Bayesian Inference. From the resulting fit, the integrated flux density and source size are calculated analytically.

### Requirements:

[ILIFU](http://docs.ilifu.ac.za/#/) cluster

### To run
```python
python imfit_mcmc.py Params.ini
```

### Notes:
1) The code requires the images to be in '.fit' extension.
2) Images should be in the data-path
