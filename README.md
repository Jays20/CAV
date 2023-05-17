# <p align="center"> Implementing a Deep Reinforcement Learning Model in a CAV environment </p>

## About the Project

This project aims to merge a vehicle onto a motorway in a CAV environment. It utilises a reinforcement learning approach in order to achieve a successful merge.  

The project involves three components:

1. CARLA Environment setup
2. Variational Autoencoder
3. Proximal Policy Optimization

## Project Setup (Installations)

1. Initialise a Conda environment using Python version **v3.7.+**
2. Activate the Conda environment
3. Install dependencies: `pip install -r requirements.txt`
4. Update **poetry**: `cd poetry/ && poetry update`
5. Download the **CARLA server (0.9.8)** + **Additional Maps** and run the server

## Built With

* [Python](https://www.python.org/downloads/release/python-370/) - Programming language
* [PyTorch](https://pytorch.org/) - Open source machine learning framework
* [CARLA](http://carla.org/) - An urban driving simulator
* [Poetry](https://python-poetry.org/) - Packaging and dependency manager
* [Tensorboard](https://www.tensorflow.org/tensorboard) - Visualization toolkit

# Methodology

Architectural layout encapsulating the three most essential components: 

1. CARLA Simulation. 
2. VAE. 
3. PPO Agent.

<p align="center"> <img width="500" src="info/diagrams/Methodology.png"></p>
<p align="center"> Architectural Methodology</p>


## How to Run

## Running a Trained Agent

With the project, we provide you two pretrained PPO agents, one for each town (Town 02 & Town 07).
The preTrained serialized files for this model are placed in `preTrained_models/PPO/<town>` folder.

```
python continuous_driver.py --exp-name ppo --train False
```

By deafult we are on Town 07 but we can changed it to Town 02 with the following argument addition:

```
python continuous_driver.py --exp-name ppo --train False --town Town02
```

## Training a New Agent

In order to train a new agent use the following command:

```
python continuous_driver.py --exp-name ppo
```

This will start training an agent with the default parameters, and checkpoints will be written to `checkpoints/PPO/<town>/` and the other metrics will be logged into `logs/PPO/<town>/`. Same as above, by default we're training on Town07 but we can change it to Town02 with this argument addition `--town Town02`.

### How our Training looks like.

<p align="center"><img width="550" src="info/gifs/town 7.gif"> </p>
<p align="center">Town 7</p>
<div>
</div>
<p align="center"><img width="550" src="info/gifs/town 2.gif"> </p>
<p align="center">Town 2</p>

## Variational AutoEncoder

The Variational Autoencoder (VAE) training process starts by driving around automatically and manually, collecting 12,000 160x80 semantically segmented images we will be using for training. Then, we will use the SS image as the input to the variational autoencoder (h ∗ 𝑤 ∗ 𝑐 = 38400 input units). VAE’s weights are frozen while our DRL network trains.

<p align="center"><img width="550" src="info/diagrams/VAE.png"> </p>
<p align="center"> Variational Autoencoder </p>

### Once we have trained our VAE, we can use the following command to check out the reconstructed images:

```
cd autoencoder && python reconstructor.py
```
<p align="center"><img width="350" src="info/diagrams/VAE Reconstruction.png"> </p>
<p align="center"> Original Image to Reconstructed Image </p>


## Project Architecture Pipeline (Encoder to PPO)

The folloing diagram depicts the VAE+PPO training pipeline. Note: all the variable names are missing the subscript 𝑡.

<p align="center"><img width="720" src="info/diagrams/PPO Network (extended).png"> </p>
<p align="center"> VAE + PPO training pipeline </p>


# File Overview

| File                          | Description                                                                                                           |
| ------------------------------| --------------------------------------------------------------------------------------------------------------------- |
| continuous_driver.py          | Script for training/testing our continuous agent e.g. PPO agent                                                       |                         |
| encoder_init.py               | script that uses the trained Encoder to turn the incoming images (states) into latent space                           |
| parameters.py                 | Contains the hyper-paramters of the Project                                                                           |
| simulation/connection.py      | Carla Environment class that makes the connection with the CARLA server                                               |
| simulation/environment.py     | CARLA Environment class that contains most of the Environment setup functionality (gym inspired class structure)      |
| simulation/sensors.py         | Carla Environment file that contains all the agent's sensor classes (setup)                                           |
| simulation/settings.py        | Carla Environement file that contains environment setup parameters                                                    |
| runs/                         | Folder containing Tensorboard plots/graphs                                                                            |
| preTrained_models/ppo         | Folder containing pre-trained models' serialized files                                                                |
| networks/on_policy/agent.py   | Contains code of our PPO agent                                                                                        |
| networks/on_policy/ppo.py     | Contains code of our PPO network                                                                                      |
| logs/                         | Folder containing the logged metrics of our agent while training                                                      |
| info/                         | Folder containing figures, gifs, diagrams, & documentation of the project                                             |
| checkpints/                   | Folder containing serialized parameters of our agent saved while training                                             |
| carla/                        | Folder containing CARL egg file, that is used in order to make connection with the server                             |
| autoencoder/                  | Folder containing the code for our Variational Autoencoder (VAE)                                                      |


## To view the training progress/plots in the Tensorboard:

```
tensorboard --logdir runs/
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details