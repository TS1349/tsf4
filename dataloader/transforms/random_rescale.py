from torchvision.transforms.v2.functional import resize

from .utils import torch_random_int

class RandomRescale(object):
    def __init__(self, shorter_side_lengths = [256, 320]):
        self.shorter_side_lengths = shorter_side_lengths
        self.num_choices = len(shorter_side_lengths)

    def _random_length(self):
        random_choice = torch_random_int(low=0, high= self.num_choices)
        return self.shorter_side_lengths[random_choice]

    def __call__(self, sample):
        h, w = sample.video.shape[:2]

        new_length = self._random_length()

        if (h <= w):
            w /= h # save ratio
            h = new_length
            w = round(w * new_length)
        else:
            h /=  w # save ratio
            w = new_length
            h = round(h * new_length)

        new_video = resize(inpt = sample.video,
                           size = [h, w])
        sample["video"] = new_video
        return sample
