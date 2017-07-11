import random
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread

import tensorflow as tf


class Trainer:
    def __init__(self, train_step, save, simulation_queue, update_frequency=1):
        self.train = train_step
        self.save = save
        self.simulation_queue = simulation_queue
        self.update_frequency = update_frequency
        self.worker = Thread(target=self.worker, daemon=True)

    def worker(self):
        """Train network(s)."""
        while True:  # Train forever. Train steps are limited by agent:
            if random.random() < .01:  # Write logs sometimes.
                step = self.train(summarize=True)
            else:
                step = self.train()
            if step % 1000 == 0:  # Save model from time to time.
                self.save()

            # Every update step allows `update_frequency` environment steps.
            for _ in range(self.update_frequency):
                self.simulation_queue.put(1)  # Blocks if queue is full.

    def start(self):
        if not self.worker.is_alive():
            self.worker.start()
