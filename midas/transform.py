import numpy as np
import cv2
import math


class NormalizeImage(object):
    """
    Normalize Image by given mean & std.
    """

    def __init__(self, mean, std):
        self._mean = mean
        self._std = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self._mean) / self._std

        return sample


class PrepareForNet(object):
    """
    prepare sample for usage as network input
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image, np.float32)

        if "mask" in sample:
            sample["mask"] = np.ascontiguousarray(sample["mask"], np.float32)

        if "disparity" in sample:
            sample["disparity"] = np.ascontiguousarray(sample["disparity"], np.float32)

        if "depth" in sample:
            sample["depth"] = np.ascontiguousarray(sample["depth"], np.float32)

        return sample


class Resize(object):
    """
    Resize sample to given size
    """

    def __init__(self,
                 width,
                 height,
                 resize_target=True,
                 keep_aspect_ratio=False,
                 ensure_multiple_of=1,
                 resize_method="lower_bound",
                 image_interpolation_method=cv2.INTER_AREA):
        """
        Init Function
        <<<
        - width:int > desired output width.
        - height:int > desired output height.
        - resize_target:boolean (optional) > Resize full sample (image, mask, target)/ Resize image only [True].
        - keep_aspect_ration:boolean (optional) > keep aspect ratio (output might not have desired size) [False].
        - ensure_multiple_of:int (optional) > Output width & height is constrained of this parameter [1].
        - resize_method:str (optional) > "lower_bound": output will be at least as large as the given size.
                                         "upper_bound": output will be at max as large as give size. (Output size might be smaller than given size.)
                                         "minimal": Scale as least as possible. (Output size might be smaller than given size.)
        - image_interpolation_method:cv2Object (optional): method for image interpolation.
        >>>
        """
        self._width = width
        self._height = height
        self._resize_target = resize_target
        self._keep_aspect_ratio = keep_aspect_ratio
        self._multiple_of = ensure_multiple_of
        self._resize_method = resize_method
        self._image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self._multiple_of) * self._multiple_of).astype(int)

        if max_val is None and y > max_val:
            y = (np.floor(x / self._resize_method) * self._multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self._multiple_of) * self._multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height & width
        scale_height = self._height / height
        scale_width = self._width / width

        if self._keep_aspect_ratio:
            if self._resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self._resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self._resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(f"resize_method {self._resize_method} not implemented")
        if self._resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, min_val=self._height)
            new_width = self.constrain_to_multiple_of(scale_width * width, min_val=self._width)

        elif self._resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, max_val=self._height)
            new_width = self.constrain_to_multiple_of(scale_width * width, max_val=self._width)

        elif self._resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)

        else:
            raise ValueError(f"resize_method {self._resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(sample["image"].shape[1], sample["image"].shape[0])

        # resize sample
        sample["image"] = cv2.resize(sample["image"],
                                     (width, height),
                                     interpolation=self._image_interpolation_method)

        if self._resize_target:
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(sample["disparity"],
                                                 (width, height),
                                                 interpolation=cv2.INTER_NEAREST)

            if "depth" in sample:
                sample["depth"] = cv2.resize(sample["depth"],
                                             (width, height),
                                             interpolation=cv2.INTER_NEAREST)

            sample["mask"] = cv2.resize(sample["mask"].astype(np.float32),
                                        (width, height),
                                        interpolation=cv2.INTER_NEAREST)
            sample["mask"] = sample["mask"].astype(bool)

        return sample
