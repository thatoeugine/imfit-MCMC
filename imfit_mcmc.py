import sys
import configparser

# Loading the parameters file:
config = configparser.ConfigParser()
config.read(sys.argv[1])


# ===============================================================
#                          RUN MCMC
# ===============================================================

if config.getboolean('pipeline', 'RUN'):
    from Imfit_MCMC.imfit_runner import runImfitMCMC

    runImfitMCMC(config, sys.argv[1])



else:
    print('Something went wrong!')
