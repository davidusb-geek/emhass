"""
This is a script for saving the database to be used by PVLib for
modules and inverters models. This was necessary to keep the
database up to date with the latest database version from SAM
while updating the out-dated original database from PVLib.
This script uses the tabulate package: pip install tabulate
"""

import bz2
import pathlib
import pickle as cPickle

import pandas as pd
import pvlib
from tabulate import tabulate

from emhass.utils import get_logger, get_root

# the root folder
root = pathlib.Path(str(get_root(__file__, num_parent=2)))
emhass_conf = {}
emhass_conf["data_path"] = root / "src/emhass/data/"  # NOTE CUSTOM DATA PATH
emhass_conf["root_path"] = root / "src/emhass/"
emhass_conf["docs_path"] = root / "docs/"
emhass_conf["config_path"] = root / "config.json"
emhass_conf["defaults_path"] = emhass_conf["root_path"] / "data/config_defaults.json"
emhass_conf["associations_path"] = emhass_conf["root_path"] / "data/associations.csv"

# create logger
logger, ch = get_logger(__name__, emhass_conf, save_to_file=False)

if __name__ == "__main__":
    save_new_files = True

    logger.info("Reading original outdated database from PVLib")
    cec_modules_0 = pvlib.pvsystem.retrieve_sam("CECMod")
    cec_inverters_0 = pvlib.pvsystem.retrieve_sam("cecinverter")

    logger.info("Reading the downloaded database from SAM")
    cec_modules = pvlib.pvsystem.retrieve_sam(path=str(emhass_conf["data_path"] / "CEC Modules.csv"))
    cec_modules = cec_modules.loc[:, ~cec_modules.columns.duplicated()]  # Drop column duplicates
    cec_inverters = pvlib.pvsystem.retrieve_sam(path=str(emhass_conf["data_path"] / "CEC Inverters.csv"))
    cec_inverters = cec_inverters.loc[:, ~cec_inverters.columns.duplicated()]  # Drop column duplicates

    logger.info("Reading custom EMHASS database")
    cec_modules_emhass = pvlib.pvsystem.retrieve_sam(path=str(emhass_conf["data_path"] / "emhass_modules.csv"))
    cec_inverters_emhass = pvlib.pvsystem.retrieve_sam(path=str(emhass_conf["data_path"] / "emhass_inverters.csv"))
    strait_str = "================="
    logger.info(strait_str)
    logger.info(strait_str)

    logger.info("Updating and saving databases")

    # Modules
    cols_to_keep_modules = [elem for elem in list(cec_modules_0.columns) if elem not in list(cec_modules.columns)]
    cec_modules = pd.concat([cec_modules, cec_modules_0[cols_to_keep_modules]], axis=1)
    logger.info(
        f"Number of elements from old database copied in new database for modules = {len(cols_to_keep_modules)}:"
    )
    cols_to_keep_modules = [elem for elem in list(cec_modules_emhass.columns) if elem not in list(cec_modules.columns)]
    cec_modules = pd.concat([cec_modules, cec_modules_emhass[cols_to_keep_modules]], axis=1)
    logger.info(
        f"Number of elements from custom EMHASS database copied in new database for modules = {len(cols_to_keep_modules)}:"
    )
    print(
        tabulate(
            cec_modules_emhass[cols_to_keep_modules].head(20).iloc[:, :5],
            headers="keys",
            tablefmt="psql",
        )
    )
    logger.info(strait_str)
    logger.info(strait_str)

    # Inverters
    cols_to_keep_inverters = [elem for elem in list(cec_inverters_0.columns) if elem not in list(cec_inverters.columns)]
    cec_inverters = pd.concat([cec_inverters, cec_inverters_0[cols_to_keep_inverters]], axis=1)
    logger.info(
        f"Number of elements from old database copied in new database for inverters = {len(cols_to_keep_inverters)}"
    )
    cols_to_keep_inverters = [
        elem for elem in list(cec_inverters_emhass.columns) if elem not in list(cec_inverters.columns)
    ]
    cec_inverters = pd.concat([cec_inverters, cec_inverters_emhass[cols_to_keep_inverters]], axis=1)
    logger.info(
        f"Number of elements from custom EMHASS database copied in new database for inverters = {len(cols_to_keep_inverters)}"
    )
    print(
        tabulate(
            cec_inverters_emhass[cols_to_keep_inverters].head(20).iloc[:, :5],
            headers="keys",
            tablefmt="psql",
        )
    )
    logger.info(strait_str)
    logger.info(strait_str)
    logger.info("Modules databases")
    print(tabulate(cec_modules.head(20).iloc[:, :5], headers="keys", tablefmt="psql"))
    logger.info("Inverters databases")
    print(tabulate(cec_inverters.head(20).iloc[:, :3], headers="keys", tablefmt="psql"))
    if save_new_files:
        with bz2.BZ2File(emhass_conf["data_path"] / "cec_modules.pbz2", "w") as f:
            cPickle.dump(cec_modules, f)
    if save_new_files:
        with bz2.BZ2File(emhass_conf["data_path"] / "cec_inverters.pbz2", "w") as f:
            cPickle.dump(cec_inverters, f)
