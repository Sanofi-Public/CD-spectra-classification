# CD_Classification-Classification_of_CD_spectra_using_linear_and_non_linear_information
This reposiroty is made to host a workflow that create a reference spectrum for a set of CD spectra and evaluate it using linear and non-linear information.
## Input
This workflow take on input **2** files, one containing data from CD spectra and one associating the name of each spectra to a theoretical class. This theoretical class can be designed by the means of your choice, spectral aspect or theorétical structure of your spectrum. The datafiles can be organised in columns or in lines with at the beginning of each column, or line, the name of the spectra corresponding to the ones in the reference file where it is associated to a class.
## What does it does?
### First step
At first, the workflow will calculate a reference for your spectra using SVD, a calculation methode that is kind of a PCA.
### Second step
Then, it evaluate the calculated reference using two metrics.
One is the *correlation* between your referece and each spectra of the dataset. This gives us an information for the *linear information* that is carried by both your reference and each spectra of the dataset.
The second is the *mutual information* between your reference and each spectra of the dataset. Instead of *linear correspondance* this information gives us information about ***non*** *linear correspondance* between the reference and the dataset.
### Third step
The third step is the calculation of a Score consisting of the product of the two metrics calculated before. This score can be used in several ways to determine if rather your reference is a good representation of your class or not.
## Output
This code gives as an output three files :
*  a reference spectrum organized in line, on the same wavelengths of the dataset
*  a file containing the correlation between the edited reference and your dataset
*  a third one combining correlation and mutual information between your dataset and your reference
# Remarks
* This code does not deal with wavelength selection, it's to the user discretion to take care of having the values on every wavelengths in their data files on every spectrum.
* This code has been developped on CD spectra but can be used on every spectral data showing both linear and non linear correspondances.
