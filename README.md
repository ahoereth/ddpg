# Continuous Control with Deep Reinforcement Learning

## Usage
To start the environment using docker simply run the following command:

    docker-compose up notebook tensorboard

It will print the required authentication key for the notebook running on `localhost:8888` during the end of the startup process. Tensorboard will be available through `localhost:6006`.

Alternatively one can run the following on an AWS GPU instance for using GPU accelerated TensorFlow.

    docker-compose up notebook-aws tensorboard

The fastest way to get an instance up is using a prepared AMI (an AWS EC2 image) with the following command (adjust to your needs):

    docker-machine create letsplay \
        --driver amazonec2 \
        --amazonec2-region eu-west-1 \
        --amazonec2-zone b \
        --amazonec2-instance-type p2.xlarge \
        --amazonec2-request-spot-instance \
        --amazonec2-spot-price 0.4 \
        --amazonec2-root-size 32 \
        --amazonec2-ami ami-71e4d817

### Without docker

    pip install -r requirements.txt
