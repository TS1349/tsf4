import torch

from model import tsf_base_4c_32f

NUM_CLASSES = 4
model = tsf_base_4c_32f(NUM_CLASSES)

BATCH_SIZE = 2
CHANNELS = 4
FRAMES = 32
IMAGE_SIZE = 224
SIZE = (BATCH_SIZE, CHANNELS, FRAMES, IMAGE_SIZE, IMAGE_SIZE)

x = torch.rand(SIZE)
y = model(x)

print(y.size())
