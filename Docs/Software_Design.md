# Software Design

The purpose of this software program is to predict the Density of State (DOS) and other electronic properties of new materials based on an X-Ray Diffraction (XRD) datafiles, using a machine learning algorithm. Below we describe the use cases within the program and the component to accomodate that use case. 

### Mine data from the Materials Project Database
   The XRD and DOS data off of the Materials Project Database (MPD) will be the source of training and test datasets.
   
   ***MatMiner*** is the software tool used to access the data on the MPD.



### Uploading JSON (XRD) data file
XRD data is in the format of a .json file. When this program the user wants to predict the DOS of their material, they will upload a .json file into the program. 

***TBD*** on the program we use or create to do that. 


### Cleaning the data 
In either data source, the data will need to be cleaned and edited such that it is presentable to the machine learning model.

***TBD*** on the method we use to accomplish this. It will likely depend heavily on the machine learning tool we choose. 


### Separating training dataset from test dataset
This is fairly simple, and the data will just be divided into 80% training and 20% test datasets. 

***TBD*** on if we use a tool to do this or if we just write our own code. 


### Running the training set through a machine learning algorithm 
This will train the algorithm to predict the DOS with XRD data. 

***Scikitlearn, Keras, and Tensorflow*** are possible candidates for the machine learning algorithm we use. If we end up running the data through multiple tools to select the best one, we may end up using all three.


### Running the test set through the same alogrithm 
This will test the algorithm to make sure it can accruately predict the DOS with XRD data. 

***Scikitlearn, Keras, and Tensorflow*** are possible candidates for the machine learning algorithm we use. If we end up running the data through multiple tools to select the best one, we may end up using all three.


#### Possible Use Case: Running both the training set and test set through multiple machine learning algorithms to determine the best model

