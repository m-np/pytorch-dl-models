<h1 align = "center">HOWTO</h1>

<div align = "justify">

The template provides a minimal approach for getting started with an AI/ML project, and has hardly any dependencies required. However, the [`notebooks/BOILERPLATE.ipynb`](notebooks/BOILERPLATE.ipynb) provides popular import and its configurations (like `pandas`, `numpy`, `scikit-learn` and `pytorch`). A high level directory overview is as follows:

```
├───config          : store all configuration files
│
├───data            : responsible for all data handling, or contains raw data
│   └───processed   : contains processed data (like combined/normalized dataframes, tables, etc.)
│
├───notebooks       : contains notebooks for building models 
│
├───src             : source directory
│   └───models      : directory containing model definations
│
└───utils           : utilities directory containing functions and/or submodules
```

</div>
