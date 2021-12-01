# RLTaskOffloading

## Prerequisites 
Baselines requires python3 (>=3.5) with the development headers. You'll also need system packages CMake, OpenMPI, graphviz and zlib. Those can be installed as follows
### Ubuntu 
    
```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
sudo apt-get install graphviz
```

### Configure the virtual environment:
It is better to use virtual environment (e.g., Anaconda) to run the code. About how to install anaconda please refer to the official website: https://www.anaconda.com

Once you have anaconda installed, run 

```bash
conda env create -f environment.yaml
```

to create the virtual environment. The current version of the code only support tensorflow 1.x (>=1.5).  

### Run the code
We implemented two DRL-based algorithms for task offloading: DRLTO and DDQNTO. 

To train and evaluate DRLTO under different scenarios, run
```bash
# train and evaluate DRLTO with different number of tasks and LO target.
python train.py --algo DRLTO --scenario Number --goal LO --dependency True 
# train and evaluate DRLTO with different number of tasks and EE target.
python train.py --algo DRLTO --scenario Number --goal EE --dependency True 
# train and evaluate DRLTO with different transmission rate and LO target.
python train.py --algo DRLTO --scenario Trans --goal LO --dependency True 
# train and evaluate DRLTO with different transmission rate and EE target.
python train.py --algo DRLTO --scenario Trans --goal EE --dependency True 
```

To train DRLTO without considering task dependency, run
```bash
# train and evaluate DRLTO with different number of tasks and LO target without considering dependency
python train.py --algo DRLTO --scenario Number --goal LO --dependency False 
```

To train and evaluate DDQNTO under different scenarios, run 
```bash
# train and evaluate DDQNTO with different number of tasks and LO target. In DDQNTO we do not consider the dependency.
python train.py --algo DDQNTO --scenario Number --goal LO --dependency False 
# train and evaluate DDQNTO with different number of tasks and EE target.
python train.py --algo DDQNTO --scenario Number --goal EE --dependency False 
# train and evaluate DDQNTO with different transmission rate and LO target.
python train.py --algo DDQNTO --scenario Trans --goal LO --dependency False 
# train and evaluate DDQNTO with different transmission rate and EE target.
python train.py --algo DDQNTO --scenario Trans --goal EE --dependency False 
```

The running results can be found in the log folder (default path of log folder is './log/Result')

To evaluate the heuristic algorithms, run
```bash
python evaluate_heuristic_algo.py --scenario Number --goal LO
python evaluate_heuristic_algo.py --scenario Number --goal EE
python evaluate_heuristic_algo.py --scenario Trans --goal LO
python evaluate_heuristic_algo.py --scenario Trans --goal EE
```

### Related publication
If you are interested in this work, please cite the paper

```bash
@article{Wang2021Depedent,
  author={Wang, Jin and Hu, Jia and Min, Geyong and Zhan, Wenhan and Zomaya, Albert and Georgalas, Nektarios},
  journal={IEEE Transactions on Computers}, 
  title={Dependent Task Offloading for Edge Computing based on Deep Reinforcement Learning}, 
  year={2021},
  doi={10.1109/TC.2021.3131040}}
```