# Continuous Control with Deep Reinforcement Learning

![torcs run 1](docs/torcs.gif)
![torcs run 2](docs/torcs2.gif)

## Usage
The codebase itself uses Python 3.5+ with Tensorflow 1.2 and numpy. For the reinforcement learning environments the [OpenAI Gym](https://gym.openai.com/) and a [modified](src/lib/torcs) [version](docker/Dockerfile.torcs) of [gym_torcs](https://github.com/ugo-nama-kun/gym_torcs) is used.

### With docker
In order to not need to globally install torcs (and to use an [intel optimized Python and Tensorflow build](https://hub.docker.com/r/ahoereth/tensorflow/)) we make use of [docker/moby](https://github.com/moby/moby) and [docker-compose](https://github.com/docker/compose). When the two are installed one just needs to run the following command to start the torcs training:

```bash
docker-compose up
```

This will start a total of 4 containers exposing multiple services on different ports. The ones of particular interest:

- Watch torcs in your browser: `localhost:6901` -- password: `tftorcs`. You can change the view with `F2`.
- Checkout live stats on tensorboard: `localhost:6006`
- Start the containers using `CMD='jupyter notebook --allow-root' docker-compose up` and access the notebook on `localhost:8888`

Docker also enables one to easily run everything on a remote instance using [docker-machine](https://github.com/docker/machine) -- perfect when exposing everything as web services as we do above.

### Without docker
If you want to run torcs, use docker. For playing around with other Gym environments you can install the requirements using `pip install -r requirements.txt` and use `run.py` -- checkout the source, should be self explanatory.

### Loading checkpoints & changing hyperparameters
Because its kind of messy to pass arguments to python files through docker-compose, one needs to edit the run scripts directly for changing anything.

By default `docker-compose up` will run `torcs.py` -- to use `run.py` instead one can use `CMD='python3 run.py' docker-compose-up`. By editing the designated file one can tweak hyper parameters and load checkpoints -- both should be self explanatory. `torcs.py` is currently configured to the hyper parameters which match [this checkpoint](https://drive.google.com/open?id=0B-k-3UScwy54MDg3bmZQTFJQcUk) -- extract the zip such that the `runs` folder is in the base of this repository and remove the comment from the `CHECKPOINT` line in `torcs.py`, run `docker-compose up` and visit `localhost:6901`.
