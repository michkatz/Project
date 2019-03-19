
# coding: utf-8

# In[19]:


import pandas as pd
import MPD_Data_Retriaval as MPD
import os.path
from unittest import mock


# In[16]:


def Test_Retrieve_data():
    MPD.Retrieve_data(0, 0.001, 'raw')
    assert os.path.isfile('./raw.csv') == True


# In[8]:


def Test_Extract_data():
    df = pd.read_csv('raw.csv', sep='\t')
    results = MPD.Extract_data(df.iloc[0])
    assert isinstance(results, pd.DataFrame)


# In[20]:


def Test_Reformat_data():
    df = pd.read_csv('raw.csv', sep='\t')
    results = MPD.Reformat_data(df.iloc[0])
    assert isinstance(results, dict)


# In[15]:


def Test_Produce_data():
    MPD.Produce_data('raw', 'processed')
    assert os.path.isfile('./processed.csv') == True


# In[18]:


def Test_MPD_Data():
    results = MPD.MPD_Data(0, 0.001, 'raw', 'processed')
    assert isinstance(results, pd.DataFrame)

