"""
@author: Thato Manamela
Uinversity of Pretoria, Department of Physics, Radio Asronomy Research Group.
"""
#____________________________________________________________________________________________


# This code runs each code as module

#____________________________________________________________________________________________



import os

SOURCE_FINDING_CONTAINER = 'singularity exec /data/exp_soft/containers/sourcefinding_py3.simg'

run_dir = os.getcwd()




def runImfitMCMC(config, parameter_filename):
    

    cmd = '{0} python {1}/Imfit_MCMC/imfit_MCMC.py {1}/{2}'.format(SOURCE_FINDING_CONTAINER,run_dir,parameter_filename)
    print(cmd)
    os.system(cmd)

