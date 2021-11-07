import os
import sys
import configparser


#Loading the parameters file:
config = configparser.ConfigParser()
config.read(sys.argv[1])
    

#======================================================================================
#                                     Stacking
#======================================================================================

if config.getboolean('pipeline', 'dostacking'):
    from codes.runner import runStack
    runStack(config, sys.argv[1])

#======================================================================================
#                            Calculations Post-stacking
#======================================================================================

if config.getboolean('pipeline', 'docalulations'):
    from codes.runner import runFlux_n_Size_esti, runFlux_esti, runSize_convert, runLumi_SFR
    from Imfit_MCMC.imfit_runner import runImfitMCMC
    runImfitMCMC(config, sys.argv[1])
    runFlux_esti(config, sys.argv[1])
    runSize_convert(config, sys.argv[1])
    runLumi_SFR(config, sys.argv[1])


else:
    print('All Tasks Complete')
