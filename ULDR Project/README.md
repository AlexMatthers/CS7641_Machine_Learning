README for Unsupervised Learning and Dimensionality Reduction Project:

Download the folder titled "ULDR Project" from this github link:

Ensure that Python 3.8.0 is installed and that the executable is add to the path.
Ensure that the following libraries are installed:
    numpy
    scipy
    sklearn
    matplotlib
    pandas

From a command line interface navigate into the directory downloaded from github.
Next run the following command 'python primary.py'.
An option to delete an existing 'Results' directory will be presented (if one exists) if a fresh data run is desired.
An option to choose the scaling method will be presented. The options offered are the RobustScaler, StandardScaler, and QuantileTransformer from the sklearn library.
    The StandardScaler removes the mean and scales the data to unit variance.
    The RobustScaler uses percentiles to work similarly to StandardScaler but ignoring extreme outliers.
    The QuantileTransformer (specifically uniform) applies a non-linear transform to map the probability density function of each feature to a uniform distribution. This bounds the feature data to the range [0, 1].
An option is presented to choose which dataset to run.
Basic clustering and dimensional reduction are applied then an option to apply clustering on the dimensionally reduces data is offered.
A final option to run the neural network experiments is offered only in the case that the 'Epileptic Seizure Recognition' dataset is run.
