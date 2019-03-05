# Software Design

The purpose of this software program is to predict the Density of State (DOS) and other electronic properties of new materials based on an X-Ray Diffraction (XRD) datafiles, using a machine learning algorithm. Below we describe the use cases within the program and the component to accommodate that use case.

## 1.	Upload XRD data into the program
- a. Mine data from the Materials Project Database
     - i.	Inputs: JSON files from MPD
     - ii.	Outputs: Pandas dataframe
- b. Cleaning the data
     - i.	Inputs: Pandas dataframe with all the messy data
     - ii.	Outputs: Pandas dataframe with only the needed data in a format that is readable to the ML program

## 2.	Use the data to train the algorithm
- a.	Separating training dataset from test dataset
    - i.	Input: Cleaned Pandas dataframe
    - ii. Output: Two separate sets for training and testing
- b.	Running the training set through a machine learning algorithm
    - i.	Input: Training set with both XRD and DOS data
    - ii.	Output: Estimated DOS
- c.	Running the test set through the same algorithm
    - i.	Input: Testing set with both XRD and DOS data
    - ii.	Output: Estimated DOS
- d.	Verifying the statistical reliability of the predicted results
    - i.	Input: Test-Estimated DOS and actual DOS
    - ii.	Output: Statistical comparisons

## 3.	Use the data to predict the DOS of a new XRD file
- a.	Running the unknown set through the same algorithm
    - i.	Input: XRD JSON file
    - ii.	Output: DOS JSON file
- b.	Visualize the predicted data
    - i.	Input: DOS JSON file
    - ii.	Output: A graph
- c.	Present data is a save-able format
    - i.	Input: DOS JSON file
    - ii.	Output: DOS csv file
