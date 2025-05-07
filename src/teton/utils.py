"""
Utility functions.
"""

from itertools import islice

import numpy as np
from av.container.input import InputContainer
from numpy.lib.stride_tricks import as_strided
from numpy.typing import NDArray

LABELS_TO_KEEP = (
    "answering questions",
    "applauding",
    "applying cream",
    "arranging flowers",
    "bandaging",
    "bending back",
    "blowing nose",
    "blowing out candles",
    "braiding hair",
    "brush painting",
    "brushing hair",
    "brushing teeth",
    "celebrating",
    "clapping",
    "cleaning floor",
    "cleaning toilet",
    "cleaning windows",
    "cracking neck",
    "crying",
    "curling hair",
    "cutting nails",
    "dining",
    "doing aerobics",
    "doing laundry",
    "doing nails",
    "drawing",
    "drinking",
    "drumming fingers",
    "eating burger",
    "eating cake",
    "eating carrots",
    "eating chips",
    "eating doughnuts",
    "eating hotdog",
    "eating ice cream",
    "eating spaghetti",
    "eating watermelon",
    "exercising arm",
    "exercising with an exercise ball",
    "faceplanting",
    "filling eyebrows",
    "finger snapping",
    "fixing hair",
    "folding clothes",
    "folding napkins",
    "folding paper",
    "gargling",
    "getting a haircut",
    "gymnastics tumbling",
    "hugging",
    "ironing",
    "kissing",
    "laughing",
    "lunge",
    "making a sandwich",
    "making bed",
    "making tea",
    "marching",
    "massaging back",
    "massaging feet",
    "massaging legs",
    "massaging person's head",
    "mopping floor",
    "moving furniture",
    "opening bottle",
    "opening present",
    "peeling apples",
    "petting animal (not cat)",
    "petting cat",
    "pumping fist",
    "pushing cart",
    "pushing wheelchair",
    "reading book",
    "reading newspaper",
    "ripping paper",
    "setting table",
    "shaking hands",
    "shaking head",
    "sneezing",
    "sniffing",
    "squat",
    "sticking tongue out",
    "stretching arm",
    "stretching leg",
    "sweeping floor",
    "taking a shower",
    "tapping pen",
    "tasting food",
    "texting",
    "trimming or shaving beard",
    "tying bow tie",
    "tying tie",
    "unboxing",
    "using computer",
    "using remote controller (not gaming)",
    "washing dishes",
    "washing feet",
    "washing hair",
    "washing hands",
    "watering plants",
    "whistling",
    "writing",
    "yawning",
)


def batch_and_crop_frames(
    container: InputContainer,
    minimum_crop: int,
    square_crop: bool,
    window_size: int = 16,
    seconds_between_frames: float = 0.2,
) -> list[NDArray[np.uint8]]:
    """
    Decode the video with PyAV decoder, and do a center crop.

    Args:
        container: PyAV container.
        minimum_crop: the minimum number of pixels to use in the dynamic crop
        square_crop: use a central square crop instead of a dynamic one
        window_size: the number of frames in each window
        seconds_between_frames: the (approximate) number of seconds between frames used during training
    Returns:
        List of frames of shape (window_size, crop, crop, 3).
    """
    n_frames = container.streams.video[0].frames
    assert container.duration is not None
    fps = container.duration / n_frames / 10000
    frame_sample_rate = round(fps * seconds_between_frames)
    # Get frames
    container.seek(0)
    frames = np.stack(
        [
            x.to_ndarray(format="rgb24")
            for x in islice(container.decode(video=0), 0, None, frame_sample_rate)
        ]
    )

    # Get indices of batches
    frame_indices = np.arange(len(frames))
    original_stride = frame_indices.strides[0]
    overlap = window_size // 2
    num_windows = 1 + (frame_indices.size - window_size) // overlap
    strides = (overlap * original_stride, original_stride)
    indices = as_strided(
        frame_indices, shape=(num_windows, window_size), strides=strides
    )

    batch_frames = []
    for idxs in indices:
        chunk = frames[idxs]
        if square_crop is True:
            cx = round(chunk.shape[1] / 2)
            cy = round(chunk.shape[2] / 2)
            x0 = round(cx - minimum_crop / 2)
            x1 = round(cx + minimum_crop / 2)
            y0 = round(cy - minimum_crop / 2)
            y1 = round(cy + minimum_crop / 2)
            crop = chunk[:, x0:x1, y0:y1]
        else:
            crop = _dynamic_crop(frames=chunk, minimum_crop=minimum_crop)
        batch_frames.append(crop)

    return batch_frames


def _dynamic_crop(frames: NDArray[np.uint8], minimum_crop: int) -> NDArray[np.uint8]:
    """
    Create a dynamic crop of a list of frames.

    Find bounding boxes of a person and do a square crop surrounding them.

    Args:
        frames: full frames
        minimum_crop: the minimum number of pixels to use in the dynamic crop
    Returns:
        Cropped frames
    """
    W, H = frames.shape[1:3]
    # Find dynamic regions
    variance = frames.var(0).sum(-1)
    threshold = np.percentile(variance, 99.2)
    mask = variance > threshold
    # find initial crop region
    row_indices = np.where(mask.any(1))[0]
    col_indices = np.where(mask.any(0))[0]
    x0, x1 = row_indices[0], row_indices[-1]
    y0, y1 = col_indices[0], col_indices[-1]

    # Determine target square crop
    height = y1 - y0 + 1
    width = x1 - x0 + 1
    new_side = max(minimum_crop, height, width)

    # Compute center of the existing rectangle
    cy = (y0 + y1) / 2
    cx = (x0 + x1) / 2

    # Compute new square’s integer bounds, growing symmetrically
    half = new_side / 2
    new_y0 = round(cy - half)
    new_y1 = new_y0 + new_side
    new_x0 = round(cx - half)
    new_x1 = new_x0 + new_side

    # Clip to the mask’s domain
    if new_y0 < 0:
        new_y1 -= new_y0
        new_y0 = 0
    if new_x0 < 0:
        new_x1 -= new_x0
        new_x0 = 0
    if new_y1 > H:
        new_y0 -= new_y1 - H
        new_y1 = H
    if new_x1 > W:
        new_x0 -= new_x1 - W
        new_x1 = W

    # Edgecase where full side would be included
    new_x0 = max(new_x0, 0)
    new_y0 = max(new_y0, 0)
    new_x1 = min(new_x1, W)
    new_y1 = min(new_y1, H)

    # Check for edge cases
    assert new_y0 <= y0
    assert new_x0 <= x0
    assert new_y1 >= y1
    assert new_x1 >= x1
    assert (new_y1 - new_y0) >= (y1 - y0)
    assert (new_x1 - new_x0) >= (x1 - x0)
    assert (new_y1 - new_y0) == new_side or (new_y1 - new_y0) == H
    assert (new_x1 - new_x0) == new_side or (new_x1 - new_x0) == W

    # bgr_frame = cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)
    # cv2.rectangle(bgr_frame, (new_x0, new_y0), (new_x1, new_y1), (0, 0, 255), 2)
    # cv2.imwrite(f"frame.png", bgr_frame)

    return frames[:, new_x0:new_x1, new_y0:new_y1, :]


def create_label_mask(labels: list[str]) -> NDArray[np.bool]:
    """
    Create a mask to only use a subset of the Kinetics-400 labels.

    Args:
        labels: the original labels
    Returns:
        a label mask
    """
    assert set(LABELS_TO_KEEP).issubset(labels)
    mask = np.ones(len(labels), dtype=np.bool)
    for i, label in enumerate(labels):
        if label in LABELS_TO_KEEP:
            mask[i] = False
    return mask
