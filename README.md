# Continuous Control with Deep Reinforcement Learning

![torcs](slides/torcs.gif)

## Requirements
The codebase itself uses Python 3.5+ with Tensorflow 1.2 and numpy. For the reinforcement learning environments the [OpenAI Gym](https://gym.openai.com/) and a [modified](src/lib/torcs) [version](docker/Dockerfile.torcs) of [gym_torcs](https://github.com/ugo-nama-kun/gym_torcs) is used.

## Usage
In order to not need to globally install torcs (and to use an [intel optimized Python and Tensorflow build](https://hub.docker.com/r/ahoereth/tensorflow/)) this repository uses [docker/moby](https://github.com/moby/moby) and [docker-compose](https://github.com/docker/compose). When the two are installed one just needs to run the following command to start the torcs training:

```bash
docker-compose up
```

### Without docker
If you want to run torcs, use docker. For playing around with other Gym environments you can install the requirements using `pip install -r requirements.txt` and use `run.py` -- checkout the source, should be self explanatory.
