# MSTDP implementation in PyTorch

## Project details

This repository contains a PyTorch implementation of the MSTDP presented in "Where to Go Next Day: Multi-Scale Spatial-Temporal Model for Mid-term Human Mobility Prediction"

## Requirements

- python==3.8.5
- pytorch==1.13.1
- torch_geometric==2.3.1
- numpy
- pandas
- geopandas
- shapely
- scipy
- tqdm

## Documentation

- data: file directory for datasets.
- models: file directory to define the model and its modules. 
- results: file directory to store experimental results.
- utils: file directory with various tools. 
- data_prepare.py: used to prepare the trajectory and graph datasets. 
- main.py: the entrance to the main process.

## How to deploy

### Data prepare

It is recommended to adaptively modify the code based on the specific dataset employed, in order to accomplish the construction of trajectories and graphs.

```
python data_prepare.py --[Parameter set] 
```

### Train model

A detailed description of the parameters is provided in `.\utils\parameter_settings.py`. It is advised to adaptively modify parameters when applying to a new dataset, such as `traj_len, lid_size, tract_size`, etc.

```
python main.py --[Parameter set] 
```

​    