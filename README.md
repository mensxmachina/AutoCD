# AutoCD
Towards Automated Causal Discovery

This repository contains the code for the paper:\
Towards Automated Causal Discovery: a case study on 5G telecommunication data\
Konstantina Biza, Antonios Ntroumpogiannis, Sofia Triantafillou, Ioannis Tsamardinos\
https://arxiv.org/pdf/2402.14481.pdf


## Overview
AutoCD is a causal discovery framework that aims to fully automate the application of causal discovery.

It can be applied to a plethora of real-world problems with :
* cross-sectional or temporal data
* high-dimensional data
* unmeasured confounders
* mixed data types

AutoCD consists of three modules:
1. **Automated Feature Selection (AFS)**
      - reduces the dimensionality of the problem, by selecting a set of features that optimize a user-defined target
2. **Causal Learning (CL)**
      - learns a causal model over the selected features
3. **Causal Reasoning and Visualization (CRV)**
      - visualizes and interprets the learned causal model, as a response to a set of user-defined queries 

## Packages
AutoCD uses the following publicly avalaible implementations
* MXM R package: https://CRAN.R-project.org/package=MXM 
  - feature selection
* Tetrad project: https://github.com/cmu-phil/tetrad
  - causal discovery on cross-sectional and temporal data
  - data simulation
* Tigramite project: https://github.com/jakobrunge/tigramite
  - causal discovery on temporal data
  - data simulation

It also needs the following python packages:
* scikit-learn
* pandas
* numpy
* py4cytoscape
* JPype1
* networkx

AutoCD visualizes the graphs using the Cytoscape platform: https://cytoscape.org/

## Notes
You need to download R, Java and Cytoscape to run AutoCD.\
Make sure that Python, R, Java and Cytoscape are installed in the same folder (e.g. Program Files)


## Contact
kbiza@csd.uoc.gr
