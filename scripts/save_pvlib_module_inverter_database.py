# -*- coding: utf-8 -*-
'''
    This is a script for saving the database to be used by PVLib for
    modules and inverters models. This was necessary to keep the 
    database up to date with the latest database version from SAM
    while updating the out-dated original database from PVLib.
    This script uses the tabulate package: pip install tabulate
'''
import numpy as np
import pandas as pd
import pathlib
import bz2
import pickle as cPickle
import pvlib
from tabulate import tabulate

from emhass.retrieve_hass import RetrieveHass
from emhass.optimization import Optimization
from emhass.forecast import Forecast
from emhass.utils import get_root, get_yaml_parse, get_days_list, get_logger

# the root folder
root = str(get_root(__file__, num_parent=2))
# create logger
logger, ch = get_logger(__name__, root, save_to_file=False)

if __name__ == '__main__':
    save_new_files = True
    logger.info('Reading original outdated database from PVLib')
    cec_modules_0 = pvlib.pvsystem.retrieve_sam('CECMod')
    cec_inverters_0 = pvlib.pvsystem.retrieve_sam('cecinverter')
    logger.info('Reading the downloaded database from SAM')
    cec_modules = pvlib.pvsystem.retrieve_sam(path=root + '/data/CEC Modules.csv')
    cec_modules = cec_modules.loc[:, ~cec_modules.columns.duplicated()] # Drop column duplicates
    cec_inverters = pvlib.pvsystem.retrieve_sam(path=root + '/data/CEC Inverters.csv')
    cec_inverters = cec_inverters.loc[:, ~cec_inverters.columns.duplicated()] # Drop column duplicates
    logger.info('Updating and saving databases')
    cols_to_keep = [elem for elem in list(cec_modules_0.columns) if elem not in list(cec_modules.columns)]
    cec_modules = pd.concat([cec_modules, cec_modules_0[cols_to_keep]], axis=1)
    logger.info('Modules databases')
    print(tabulate(cec_modules.head(20).iloc[:,:5], headers='keys', tablefmt='psql'))
    logger.info('Inverters databases')
    print(tabulate(cec_inverters.head(20).iloc[:,:3], headers='keys', tablefmt='psql'))
    if save_new_files:
        with bz2.BZ2File(root + '/src/emhass/data/cec_modules.pbz2', "w") as f: 
            cPickle.dump(cec_modules, f)    
    if save_new_files:
        with bz2.BZ2File(root + '/src/emhass/data/cec_inverters.pbz2', "w") as f: 
            cPickle.dump(cec_inverters, f)
    