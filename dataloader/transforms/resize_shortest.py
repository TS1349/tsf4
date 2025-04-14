from torchvision.transforms.v2.functional import resize

from ..utils import torch_random_int

class ResizeShortest(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        h, w = sample.shape[:2]
        new_length =self.size

        if (h <= w):
            w /= h # save ratio
            h = new_length
            w = round(w * new_length)
        else:
            h /=  w # save ratio
            w = new_length
            h = round(h * new_length)

        new_sample = resize(inpt = sample,
                           size = [h, w])

        return new_sample
