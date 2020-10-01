# Code for the paper "Using mobility data in the design of optimal lockdown strategies for the COVID-19 pandemic in England" 

The preprint can be found at: 

https://arxiv.org/pdf/2006.16059.pdf

Please also check the companion website, where updates on subsequent research will be described: 

https://optimallockdown.github.io/Covid19inEngland/

## Content
The content of this repository is as follows: 

- `data` contains the data used in the paper for model fit and optimal control. It contains both the raw data and the one formatted for using our code.
- `plot_routines` contains some scripts to produce plots in the paper.
- `results` contains the `.jrl` files used to store the inference results in the ABCpy library, together with some figures.
- `src` contains source code, with model definition and various utilities functions.
- `Datasets.ipynb` is a jupyter notebook describing the data sources and the data processing operations before fitting the model. 
- `inference_SEI4RD_england_data.py` is the script performing model fit; can be used with MPI parallelization by uncommenting one line. 
- `optimal_control_posterior_mean.py` is the script performing optimal control under uncertainty, together with the definition of several functions to perform optimal control in specific cases. 

## Requirements

The following should install the required packages:

```
pip3 install -r requirements.txt
```

According to your python3 installation, you may require `pip` instead of `pip3`.


TODO check data, may need to add more!
TODO check plot routines as well. 