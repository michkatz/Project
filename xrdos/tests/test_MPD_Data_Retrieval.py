import unittest

import pandas as pd
from xrdos import MPD_Data_Retrieval as MPD
import os.path


class test_xrdos(unittest.TestCase):

    def test_Retrieve_data(self):
        MPD.Retrieve_data(0, 0.001, 'raw')
        assert os.path.isfile('./raw.csv') is True
        return

    def test_Extract_data(self):
        df = pd.read_csv('raw.csv', sep='\t')
        results = MPD.Extract_data(df.iloc[0])
        assert isinstance(results, pd.DataFrame)
        return

    def test_Reformat_data(self):
        df = pd.read_csv('raw.csv', sep='\t')
        results = MPD.Reformat_data(df.iloc[0])
        assert isinstance(results, dict)
        return

    def test_Produce_data(self):
        MPD.Produce_data('raw', 'processed')
        assert os.path.isfile('./processed.csv') is True
        return

    def test_MPD_Data(self):
        results = MPD.MPD_Data(0, 0.001, 'raw', 'processed')
        assert isinstance(results, pd.DataFrame)
        return
