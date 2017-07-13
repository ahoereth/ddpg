import random
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

import tensorflow as tf


class Trainer(Thread):
    def __init__(self, train_step, save, training_queue, simulation_queue,
                 update_frequency=1):
        super(Trainer, self).__init__(target=self.train_loop, daemon=True)
        self.train = train_step
        self.save = save
        self.training_queue = training_queue
        self.simulation_queue = simulation_queue
        self.update_frequency = update_frequency

    def train_loop(self):
        """Train network(s)."""
        while True:  # Train forever.
            try:
                self.training_queue.get(timeout=2)
            except Empty:
                break
            if random.random() < .01:  # Write logs sometimes.
                step = self.train(create_summary=True)
                print('{} steps done.'.format(step))
            else:
                step = self.train()
            if step % 1000 == 0:  # Save model from time to time.
                self.save(step)

            # Every update step allows `update_frequency` environment steps.
            for _ in range(self.update_frequency):
                self.simulation_queue.put(1)  # Blocks if queue is full.
