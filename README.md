# Young-new Driver Accompaniment Program 

This code reproduces the results 
of the Casual Inference Project examining the casual effects of 
the changes in legislation relating to 
the Young-new Driver Accompaniment Program.


To setup the environment  run `conda env create -f environment.yml`.

Then to reproduce the results and figures simply run `python main.py`.

The analysis related items will be automatically created and saved under the `analysis` folder, and the results related items under the `results` folder.

The two R scripts are used for the density test and for computing the CI for one of the methods.
It assumes the package `rdrobust` and `rddensity` to be installed.

Both data files - `full_data.xlsx` and `num_issued_licenses_per_year.xlsx`
were first generated on-demand by the Central Bureau of Statistics of Israel
(Tools and databases - Statistics Generator of road accidents with casualties), extracted as a CSV file 
and then preprocessed by us to fit the specific task. 