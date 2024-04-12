# NSM_5HTR_analysis
[![DOI](https://zenodo.org/badge/628047111.svg)](https://zenodo.org/badge/latestdoi/628047111)

## Citation

To cite this work (datasets, methods, code, results), please refer to the article:

Dissecting the functional organization of the C. elegans serotonergic system at whole-brain scale

Ugur Dag*, Ijeoma Nwabudike*, Di Kang*, Matthew A. Gomes, Jungsoo Kim, Adam A. Atanas, Eric Bueno,
Cassi Estrem, Sarah Pugliese, Ziyu Wang, Emma Towlson, Steven W. Flavell

Cell 2023; doi: https://doi.org/10.1016/j.cell.2023.04.023

*Equal Contribution


## Data files
The processed data files for our two whole-brain imaging strains (SWF415 and SWF702) are available under the `data` directory.
Receptor expression, neuron class ID and publicly available connectome data are also provided.

## Scripts
Run `nsm_analysis.ipynb` to compute each neuron's temporal correlation with NSM for individual animals.
Run `generate_figure7.ipynb` to relate each neuron's functional relationship with NSM to receptor expression.
Both scripts are available under the `notebook` directory. 

## Installation
Please follow the instructions to install all prerequisite packages for behavioral tracking, image registration and signal processing: https://github.com/flavell-lab/AtanasKim-Cell2023
The functions that are used specifically for this analysis are available under `code`.

## Results
The outputs of `nsm_analysis.ipynb` for multiple animals of each imaging strain are compiled as a Julia dictionary under the `results` directory.
You may reproduce figure 7 of the above paper by directly plugging in a julia dictionary without re-running `nsm_analysis.ipynb` yourself.
