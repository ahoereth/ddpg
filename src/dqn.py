from .lib import Model


class DQN(Model):

    def __init__(
        self,
        env_name,
        *,
        batchsize=32,
        learning_rate=1e-4,
        gamma=0.99,
        **kwargs,
    ):
        self.learning_rate = learning_rate
        self.gamma = gamma
        super(DQN, self).__init__(batchsize=batchsize, **kwargs)

    def make_network(self, act_states, states, actions, rewards, terminals,
                     states_, training, action_bounds):
