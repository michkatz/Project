import pandas as pd
from xrdos import MPD_Data_Retrieval as MPD
import os.path


def test_Retrieve_data():
    MPD.Retrieve_data(0, 0.001, 'raw')
    assert os.path.isfile('./raw.csv') is True


def test_Extract_data():
    df = pd.read_csv('raw.csv', sep='\t')
    results = MPD.Extract_data(df.iloc[0])
    assert isinstance(results, pd.DataFrame)


def test_Reformat_data():
    df = pd.read_csv('raw.csv', sep='\t')
    results = MPD.Reformat_data(df.iloc[0])
    assert isinstance(results, dict)


def test_Produce_data():
    MPD.Produce_data('raw', 'processed')
    assert os.path.isfile('./processed.csv') is True


def test_MPD_Data():
    results = MPD.MPD_Data(0, 0.001, 'raw', 'processed')
    assert isinstance(results, pd.DataFrame)
