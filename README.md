# DRL For Complex Navigation in Dynamic 3D Digital environments

Code to reproduce the experiments from the PhD Thesis "Deep Reinforcement Learning For Complex Navigation in Dynamic 3D Digital Environments" by Dale Grant.
See installation and run procedures below.

## Installation
### Anaconda
* Install Anaconda/Miniconda.
* Create a conda environment: ```conda create --name drl_nav python=3.8.10```
* Install dependencies in mlagents.yaml via anaconda import.
* Install remaining dependencies in the conda environment using ```pip install -r requirements.txt```
* Install pytorch 1.8.2 for Cuda 11.1 using ```pip install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html```

### Unity
* Unity version 2022.2.12f1
* MLAgents package derived from Release 18 branch.
* Install Windows Dedicated Server module to create Windows executables for training.
* Build an .exe into a Build folder. 

## To train a model
To train a model on your computer, we run the RunPolicy python script.
The following example is for 3 training runs, using config file "sac.yaml", for the "DRLNavigation" environment at a 20x time-scale.

```
python RunPolicy.py -r 3 -c 'sac.yaml' -e "DRLNavigation" -t 20
```

## To Generate Training Results Figures
Figure plotting can be either selected as an input parameter for the RunPolicy main method or as a separate call to the Plotting.py main function.

```
python Plotting.py -r "Results" -d "Navigation" -n "Plot 1" "Plot 2" -t "Plot Title" -s 0.5
```



