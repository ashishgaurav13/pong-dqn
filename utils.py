from collections import deque
import numpy as np
from scipy.misc import imresize
import cv2

# A simple experience buffer which returns randomly sampled transitions
class ExperienceBuffer:

    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        # Return a batch of batch_size samples, randomly chosen
        random_indices = np.random.choice(len(self.buffer), batch_size,
            replace=False)
        return np.array(self.buffer)[random_indices]

    def __len__(self):
        return len(self.buffer)


# Convert state to image
def obs2img(s, image_name):
    cv2.imwrite(image_name, s)

# Downsample Montezuma Revenge state
def downsample(s):
    s = s[30:, :, :]
    resized = imresize(s, (84, 84, 3))
    grayscale = np.mean(resized, axis=2).astype(np.uint8)
    return np.resize(grayscale, (1, 84, 84))

# https://stackoverflow.com/questions/17866724/python-logging-print-statements-while-having-them-print-to-stdout
class Tee(object):

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()