# RLTaskOffloading

## Prerequisites 
Baselines requires python3 (>=3.5) with the development headers. You'll also need system packages CMake, OpenMPI, graphviz and zlib. Those can be installed as follows
### Ubuntu 
    
```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
sudo apt-get install graphviz
```

### Third party python lib requirements:
It is better to use virtual environment (e.g., Anaconda) to run the code

1. tensorflow-gpu(version >= 1.5):
```bash
	pip install tensorflow-gpu==1.14
```
2. graphviz
```bash
	pip install graphviz
```

3. pydotplus
```bash
	pip install pydotplus
```

4. gym
```bash
	pip install gym
```

### Run the code
```bash
	python offloading_ppo.py
```

Get the result from the log file defined in offloading_ppo.py. 

Related paper:

Wang J, Hu J, Min G, Zhan W, et al. Computation offloading in multi-access edge computing using a deep sequential model based on reinforcement learning[J]. IEEE Communications Magazine, 2019, 57(5): 64-69.
## Prerequisites 
