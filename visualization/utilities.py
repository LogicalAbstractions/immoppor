import torch
import cv2
from numpy import uint8


def _to_opencv_(tensor: torch.Tensor, dtype=None, scale: float = None) -> torch.Tensor:
    final_tensor = tensor

    if len(tensor.shape) == 4:
        if tensor.shape[0] != 1:
            raise ArithmeticError("4D tensors with batching must have 1 as batch size")

        width = tensor.shape[2]
        height = tensor.shape[3]
        channels = tensor.shape[1]

        final_tensor = tensor.reshape((channels, width, height))

    transposed_tensor = final_tensor.detach().numpy().transpose(1, 2, 0)

    if scale is not None:
        transposed_tensor = transposed_tensor * scale

    if dtype is not None:
        transposed_tensor = transposed_tensor.astype(dtype)

    return transposed_tensor


def display_tensor_as_image(self: torch.Tensor, dtype=None, scale: float = None):
    cv2.imshow("image", _to_opencv_(self, dtype, scale))
    cv2.waitKey()


def save_tensor_as_image(self: torch.Tensor, filename: str, dtype, scale: float):
    cv2.imwrite(filename, _to_opencv_(self, dtype, scale))
