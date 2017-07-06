import random
from datetime import datetime
from pathlib import Path
from queue import Queue

import tensorflow as tf


class Trainer:
    def __init__(self, session, train_op, memory, simulation_queue, save, step,
                 summaries, *, update_frequency=1, batchsize=32):
        self.session = session
        self.train_op = train_op
        self.simulation_queue = simulation_queue
        self.save = save
        self.step = step
        self.update_frequency = update_frequency
        self.batchsize = batchsize

    def worker(self):
        """Train network(s)."""
        while True:  # Train forever. Train steps are limited by agent:
            if random.random() < .01:  # Write logs sometimes.
                summary, step, _ = self.session.run([self.summaries,
                                                     self.step,
                                                     self.train_op])
                self.writer.add_summary(summary, step)
            else:
                self.session.run(self.train_op)
            if step % 1000 == 0:  # Save model from time to time.
                self.save(step)

            # Every update step allows `update_frequency` environment steps.
            for _ in range(self.update_frequency):
                self.simulation_queue.put(1)  # Blocks if queue is full.

    def train(self):
        """Manage feeding and training, allow one more train step.

        Starts as many feed and train threads as required in order to
        fully take advantage of the available computation resources.
        """
        # Only train when there is a memory and when the memory contains
        # enough experiences.
        assert self.memory is not None
        if len(self.memory) < self.batchsize:
            return

        # Allow one more training step.
        self.train_queue.put(1)

        # Start feed threads such that train threads always have data.
        if self.session.run(self.queue_size) < self.batchsize:
            thread = Thread(target=self.feed_thread, daemon=True)
            thread.start()
            self._feed_threads.append(thread)
            self.log('misc/threads/feed', len(self._feed_threads))

        # Start training threads if there is a training backlog.
        if self.train_queue.qsize() > 20:
            thread = Thread(target=self.train_thread, daemon=True)
            thread.start()
            self._train_threads.append(thread)
            self.log('misc/threads/train', len(self._train_threads))
