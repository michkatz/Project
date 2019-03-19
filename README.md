<p align="center">
  <img src=images/logo.png>
</p>

[![Build Status](https://travis-ci.org/michkatz/xrdos.svg?branch=master)](https://travis-ci.org/michkatz/xrdos)

xrdos is a Python toolkit that can be used to estimate the density of state (DOS) of new materials using x-ray diffraction (XRD) data. The toolkit uses a neural network machine learning model to develop a relationship between XRD peaks and a number of electronic properties related to denisty of state. 

Currently the program can estimate bandgap and the Fermi energy of the material. The training data is pulled from the Materials Project Database. 

This toolkit is maintained by a group of gradaute students at the University of Washington in Seattle. 

# Use Cases
This toolkit is designed for three major use cases. More detailed explanations can be found in `Software_Design` under `docs`. 
### 1. Upload XRD data into the toolkit. 
 At the current state of development this data needs to be cleaned prior to uploading into the toolkit. The ten most intense peaks and their corresponding 2-theta values should be organized in a 20 X 1 Numpy array, with intensities listed first and corresponding 2-theta values following. 
    
### 2. Use XRD data from the Materials Project Database to trian the algorithm.

### 3. Estimate the bandgap and Fermi energy of the material based on XRD peaks. 

# Installation
## Dependencies

xrdos requires:
 
* Python (v>)
* Keras (v>)
* Matminer (v>)
* Pandas (v>)
* Numpy (v>)
* Scikit Learn (v>)


## User Installation
xrdos is in the process of becoming pip installable. Until this is successfully set up, this program can be used by cloning the git respository. 


# Data Tips
The data can be very time consuming to download for the Materials Project Database. The module `get_data_xrdos` can be used to download the data with any desired adjustments, but the data used to train the model is already included as a csv in this repository. 

If the data is downloaded directly from the Materials Project Database, we suggest going through this arduous process only once and saving the data as a csv. 


# Contact

katzm@uw.edu
