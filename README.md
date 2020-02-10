CS7641 ML - Assignment 1 - Supervised Learning - README FILE

From this github (https://github.com/AlexMatthers/CS7641_Machine_Learning) download both the Data folder and the run.py file. 
Place both folder and file in the same directory.
Ensure that Python 3.8 or higher is installed on the system.
Ensure that Python is in the system path and callable from the command line.
Ensure that the numpy, pandas, matplotlib, and sklearn libraries are installed.
From a command prompt navigate to the directory mentioned above. Enter the command 'python run.py'.
The experiment script will run (for about 5-10 hours with 12 threads) and generate a 'Results' folder. Within this folder will be two sub-folders, one for each dataset, and within those will be the generated graphs (in svg format) and data reports for each algorithm.

The random seed, number of threads to use, and verbosity of the script are set on lines 17, 18, and 20 respectively. The verbosity is off with a value of 0 and fully on with a value of 10.

The algorithms to run can be changed in the list on line 41. Copy the existing list, comment it out, and then keep only the desired algorithm abbreviations.
