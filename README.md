<p align="center">
  <img src=images/logo.png>
</p>

[![Build Status](https://travis-ci.org/michkatz/xrdos.svg?branch=master)](https://travis-ci.org/michkatz/xrdos)

xrdos is a Python toolkit that can be used to estimate the density of state (DOS) of new materials using x-ray diffraction (XRD) data. The toolkit uses a neural network machine learning model to develop a relationship between XRD peaks and a number of electronic properties related to denisty of state. 

Currently the program can estimate bandgap and the Fermi energy of the material. The training data is pulled from the Materials Project Database. 

This toolkit is maintained by a group of gradaute students at the University of Washington in Seattle. 

See Software Design under `docs` for more information. 

# Installation
## Dependencies

xrdos requires:
 
* Python (v>3.6)
* Keras (v>2.2.4)
* TensorFlow (v>1.13.1)
* Matminer (v> 0.5.1)
* Pandas (v>0.24.1)
* Numpy (v>1.16.2)
* Scikit Learn (v>0.20.2)


## User Installation
xrdos is in the process of becoming pip installable. Until this is successfully set up, this program can be used by cloning the git respository. 


# Data Tips
The data can be very time consuming to download for the Materials Project Database. The module `get_data_xrdos` can be used to download the data with any desired adjustments, but the data used to train the model is already included as a csv in this repository. 

If the data is downloaded directly from the Materials Project Database, we suggest going through this arduous process only once and saving the data as a csv. 


# Contact

katzm@uw.edu
