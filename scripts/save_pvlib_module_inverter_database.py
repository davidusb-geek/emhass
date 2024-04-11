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

if __name__ == '__main__':

    # the root folder
    root = str(get_root(__file__, num_parent=2))
    emhass_conf = {}
    emhass_conf['config_path'] = pathlib.Path(root) / 'config_emhass.yaml'
    emhass_conf['data_path'] = pathlib.Path(root) / 'data/'
    emhass_conf['root_path'] = pathlib.Path(root)

    # create logger
    logger, ch = get_logger(__name__, emhass_conf, save_to_file=False)

    save_new_files = True
    logger.info('Reading original outdated database from PVLib')
    cec_modules_0 = pvlib.pvsystem.retrieve_sam('CECMod')
    cec_inverters_0 = pvlib.pvsystem.retrieve_sam('cecinverter')
    logger.info('Reading the downloaded database from SAM')
    cec_modules = pvlib.pvsystem.retrieve_sam(path=emhass_conf['data_path'] / 'CEC Modules.csv')
    cec_modules = cec_modules.loc[:, ~cec_modules.columns.duplicated()] # Drop column duplicates
    cec_inverters = pvlib.pvsystem.retrieve_sam(path=emhass_conf['data_path'] / 'CEC Inverters.csv')
    cec_inverters = cec_inverters.loc[:, ~cec_inverters.columns.duplicated()] # Drop column duplicates
    logger.info('Updating and saving databases')
    cols_to_keep_modules = [elem for elem in list(cec_modules_0.columns) if elem not in list(cec_modules.columns)]
    cec_modules = pd.concat([cec_modules, cec_modules_0[cols_to_keep_modules]], axis=1)
    cols_to_keep_inverters = [elem for elem in list(cec_inverters_0.columns) if elem not in list(cec_inverters.columns)]
    cec_inverters = pd.concat([cec_inverters, cec_inverters_0[cols_to_keep_inverters]], axis=1)
    logger.info(f'Number of elements from old database copied in new database for modules = {len(cols_to_keep_modules)}')
    logger.info(f'Number of elements from old database copied in new database for inverters = {len(cols_to_keep_inverters)}')
    logger.info('Modules databases')
    print(tabulate(cec_modules.head(20).iloc[:,:5], headers='keys', tablefmt='psql'))
    logger.info('Inverters databases')
    print(tabulate(cec_inverters.head(20).iloc[:,:3], headers='keys', tablefmt='psql'))
    if save_new_files:
        with bz2.BZ2File(emhass_conf['root_path'] + '/src/emhass/data/cec_modules.pbz2', "w") as f: 
            cPickle.dump(cec_modules, f)    
    if save_new_files:
        with bz2.BZ2File(emhass_conf['root_path'] + '/src/emhass/data/cec_inverters.pbz2', "w") as f: 
            cPickle.dump(cec_inverters, f)
    