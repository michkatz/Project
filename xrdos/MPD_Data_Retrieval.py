import numpy as np
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
from pydash import py_
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
mpdr = MPDataRetrieval()  # or mpdr = MPDataRetrieval(api_key='YOUR_KEY')


def Retrieve_data(bg_lower, bg_upper, raw_name):
    """
    Downloads data from the MPD

    Parameters
    ----------
    bg_lower : Int or float
      Lower bound of bandgap for the materials to be collected
    bg_upper : Int or float
      Upper bound of bandgap for the materials to be collected
    raw_name : Str
      Desired file name for raw data
    """

    properties = ['material_id', 'xrd.Cu', 'band_gap', 'efermi']
    criteria = {"band_gap": {'$gt': bg_lower, '$lt': bg_upper},
                "efermi": {'$exists': True},
                'xrd.Cu': {'$exists': True}}

    res_ids = mpdr.get_dataframe(criteria=criteria,
                                 properties=["material_id"]
                                 ).index.tolist()
    results = pd.DataFrame()

    for chunk in tqdm(py_.chunk(res_ids, 50)):  # Grabs 50 materials at a time
        temp_data = mpdr.get_dataframe(
            criteria={
                "material_id": {
                    "$in": chunk}},
            properties=properties)
        results = results.append(temp_data)

    results.to_csv(raw_name + '.csv', sep='\t')


def Extract_data(MPD_data_row):
    """
    Extracts the relevant XRD data from the dictionary obtained from MPD

    Parameters
    ----------
    MPD_data_row : Pandas dataframe
         A row of data for a single material from the full MPD dataframe

    Returns
    -------
    clean_df: Pandas dataframe
        The top 10 XRD peaks and their corresponding
        two theta values for the material
    """

    # Extracting out the amplitude and two theta values
    # from the dictionary contained inside the received data
    # then turning it into a pandas dataframe
    dirty_df = pd.DataFrame(
        literal_eval(
            MPD_data_row['xrd.Cu'])['pattern'], columns=literal_eval(
            MPD_data_row['xrd.Cu'])['meta'])
    dirty_df.drop(['hkl', 'd_spacing'], axis=1, inplace=True)

    # Sorting the peaks into the top 10 with the highest peaks
    dirty_df.sort_values('amplitude', ascending=False, inplace=True)
    dirty_df.reset_index(drop=True, inplace=True)
    clean_df = dirty_df[:10]

    return clean_df


def Reformat_data(MPD_data_row):
    """
    Reformats the cleaned data obtained from
    the extract_data function into a dictionary

    Parameters
    ----------
    MPD_data_row : Pandas dataframe
         A row of data for a single material from the full MPD dataframe

    Returns
    -------
    clean_df: Pandas dataframe
        The top 10 XRD peaks and their corresponding two theta
        values for the material
    """

    # Cleaning data and creating empty dictionary
    clean_df = Extract_data(MPD_data_row)
    mat_dict = {}

    # Loop to assign each data point to a key and stores it within the
    # dictionary
    for i in range(0, 20):
        if i < 10:
            amp_key = ('amplitude_' + str(i))
            mat_dict[amp_key] = clean_df['amplitude'][i]

        else:
            theta_key = ('two_theta_' + str(i - 10))
            mat_dict[theta_key] = clean_df['two_theta'][i - 10]

    return mat_dict


def Produce_data(raw_name, processed_name):
    """
    Produces the XRD and DOS data for all the materials passed to the function

    Parameters
    ----------
    raw_name : Str
      Desired file name for raw data
    processed_name : Str
      Desired file name for processed data
    """

    MPD_data_raw = pd.read_csv(raw_name + '.csv', sep='\t')

    # Creating prelimanry containers for XRD and DOS data
    xrd_data = {}
    dos_data = MPD_data_raw.drop(['xrd.Cu'], axis=1)
    dos_data.set_index(['material_id'], inplace=True)

    # Loop to run through each row of the dataframe
    # tqdm is used only to monitor progress during testing. MAY NEED TO REMOVE
    for i in tqdm(range(len(MPD_data_raw))):

        # Conditional to skip over materials with less than 10 XRD peaks
        # or no fermi energies
        if len(
            literal_eval(
                MPD_data_raw.iloc[i]['xrd.Cu'])['pattern']) >= 10 and np.isnan(
                MPD_data_raw.iloc[i]['efermi']) is False:

            # Obtaining and storing the XRD data for a material into a
            # dictionary
            ID = MPD_data_raw.iloc[i]['material_id']
            mat_dict = Reformat_data(MPD_data_raw.iloc[i])
            xrd_data[ID] = mat_dict

        else:

            # Replaces rows that failed the conditional with NaN
            # This is for easy removal od the rows
            dos_data.iloc[i] = float('nan')

    # Creating the final dataframe from the obtained XRD and DOS dataframes
    dos_df = dos_data.dropna()
    xrd_df = pd.DataFrame.from_dict(xrd_data, orient='index')
    full_df = pd.concat([xrd_df, dos_df], axis=1, sort=False)

    full_df.to_csv(processed_name + '.csv', sep='\t')


def MPD_Data(bg_lower, bg_upper, raw_name, processed_name):
    """
    Gathers, cleans, and produces the MPD data

    Parameters
    ----------
    bg_lower : Int or float
      Lower bound of bandgap for the materials to be collected
    bg_upper : Int or float
      Upper bound of bandgap for the materials to be collected
    raw_name : Str
      Desired file name for raw data
    processed_name : Str
      Desired file name for processed data

    Returns
    -------
    MPD_data: Pandas dataframe
        Contains the peak intensities, two theta values,
        band gap, and fermi energy for all materials collected

    Notes
    -----
    The two .csv files will be saved in the directory the code used in
    """

    Retrieve_data(bg_lower, bg_upper, raw_name)
    Produce_data(raw_name, processed_name)

    MPD_data = pd.read_csv((processed_name + '.csv'), sep='\t', index_col=0)

    return MPD_data
