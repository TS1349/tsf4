from ..utils import torch_random_int

class FullCenterCrop(object):
    def __init__(self, crop_size=(224,224)):
        self.crop_height, self.crop_width = crop_size

    def _random_ul_corner(self, height, width):
        height_range = height - self.crop_height
        width_range = width - self.crop_width

        assert (height_range > 0)
        assert (width_range > 0)

        random_height = torch_random_int(low = 0,
                                         high = height_range)

        random_width = torch_random_int(low = 0,
                                        high = width_range)

        return (random_height, random_width)


    def __call__(self, sample):
        height, width = sample.video.shape[:2]
        random_height, random_width = self._random_ul_corner(height, width)
        new_video = sample.video[...,
            random_height: random_height + self.crop_height,
            random_width: random_width + self.crop_width]

        sample["video"] = new_video

        return sample
