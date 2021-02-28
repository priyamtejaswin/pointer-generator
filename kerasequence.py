#!/usr/bin/env python
"""
created at: Sun 28 Feb 2021 01:51:09 AM EST
created by: Priyam Tejaswin

Testing the Keras Sequence and SequenceEnquer.
"""


import random
import tensorflow as tf


class DummySequence(tf.keras.utils.Sequence):
    """
    Just for testing ...
    """
    def __init__(self, x_set, y_set, batch_size):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size

    
    def __len__(self):
        """
        Returns the number of batches in the sequence.
        """
        return len(self.x) // self.batch_size

    
    def __getitem__(self, index):
        """
        Gets batch at position `index`.
        """
        lower = index * self.batch_size
        upper = (index + 1) * self.batch_size
        batch_x = self.x[lower : upper]
        batch_y = self.y[lower : upper]

        return [str(_) for _ in batch_x], batch_y


    # def on_epoch_end(self):
    #     """
    #     Called at every epoch end.
    #     How does it know tho ...?
    #     """
    #     combined = list(zip(self.x, self.y))
    #     random.shuffle(combined)
    #     self.x, self.y = zip(*combined)
    #     print("shuffled")



x = list(range(10))
y = [i**2 for i in x]

dataset = DummySequence(x, y, 2)
print(len(dataset))

ordenq = tf.keras.utils.OrderedEnqueuer(dataset, shuffle=True)
ordenq.start(workers=3, max_queue_size=15)

count = 0
for a, b in ordenq.get():
    print(a, b)
    count += 1
    if count >= 10:
        print("Count is %d; exiting." % count)
        break

ordenq.stop()