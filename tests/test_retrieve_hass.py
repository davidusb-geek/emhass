#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from unittest import TestCase
import pandas as pd
import pytz

from emhass.retrieve_hass import retrieve_hass
from emhass.utils import get_root, get_root_2pardir, get_yaml_parse, get_days_list

class TestRetrieveHass(TestCase):

    def setUp(self):
        try:
            root = get_root()
            retrieve_hass_conf, _, _ = get_yaml_parse(root)
        except:
            root = get_root_2pardir()
            retrieve_hass_conf, _, _ = get_yaml_parse(root)
        
        self.retrieve_hass_conf = retrieve_hass_conf
        self.days_list = get_days_list(self.retrieve_hass_conf['days_to_retrieve'])
        self.var_list = [self.retrieve_hass_conf['var_load'], self.retrieve_hass_conf['var_PV']]
        
    def test_get_data(self):
        self.rh = retrieve_hass(self.retrieve_hass_conf['hass_url'], self.retrieve_hass_conf['long_lived_token'], 
                           self.retrieve_hass_conf['freq'], self.retrieve_hass_conf['time_zone'])
        self.rh.get_data(self.days_list, self.var_list,
                         minimal_response=False, significant_changes_only=False)
        self.assertIsInstance(self.rh.df_final, type(pd.DataFrame()))
        self.assertIsInstance(self.rh.df_final.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.rh.df_final.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(len(self.rh.df_final.columns), len(self.var_list))
        self.assertEqual(self.rh.df_final.index.isin(self.days_list).sum(), len(self.days_list))
        self.assertEqual(self.rh.df_final.index.freq, self.retrieve_hass_conf['freq'])
        self.assertEqual(self.rh.df_final.index.tz, pytz.UTC)
        self.df_raw = self.rh.df_final.copy()
        
    def test_prepare_data(self):
        self.rh.prepare_data(self.retrieve_hass_conf['var_load'], load_negative = self.retrieve_hass_conf['load_negative'],
                             set_zero_min = self.retrieve_hass_conf['set_zero_min'], 
                             var_replace_zero = self.retrieve_hass_conf['var_replace_zero'], 
                             var_interp = self.retrieve_hass_conf['var_interp'])
        self.assertIsInstance(self.rh.df_final, type(pd.DataFrame()))
        self.assertEqual(self.rh.df_final.index.isin(self.days_list).sum(), self.df_raw.index.isin(self.days_list).sum())
        self.assertEqual(len(self.rh.df_final.columns), len(self.df_raw.columns))
        self.assertEqual(self.rh.df_final.index.freq, self.retrieve_hass_conf['freq'])
        self.assertEqual(self.rh.df_final.index.tz, self.retrieve_hass_conf['time_zone'])
        if self.retrieve_hass_conf['load_negative']:
            self.assertEqual(self.rh.df_final[self.retrieve_hass_conf['var_load']+"_positive"].isnull().sum(), 0)
        
if __name__ == '__main__':
    trh=TestRetrieveHass()
    trh.setUp()
    trh.test_get_data()
    trh.test_prepare_data()