from av.container.input import InputContainer
import tempfile
from itertools import islice
import av
from av.filter import Graph
from numpy.typing import NDArray
import numpy as np
from numpy.lib.stride_tricks import as_strided

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

def batch_and_crop_frames(container: InputContainer, crop_pixels: int, window_size: int=16, seconds_between_frames: float=0.2) -> NDArray[np.uint8]:
    """
    Decode the video with PyAV decoder, and do a center crop

    Args:
        container: PyAV container.
        window_size: the number of frames in each window
        crop_pixels: the number of pixels to use in the square centered crop
        seconds_between_frames: the (approximate) number of seconds between frames used during training
    Returns:
        Frames of shape (num_windows, window_size, crop_pixels, crop_pixels, 3).
    """
    n_frames = container.streams.video[0].frames
    fps = container.duration / n_frames / 10000
    frame_sample_rate = round(fps * seconds_between_frames)
    # Get frames
    container.seek(0)
    frames = []
    for frame in islice(container.decode(video=0), 0, None, frame_sample_rate):
        full_frame = frame.to_ndarray(format="rgb24")
        start_i = (full_frame.shape[0] - crop_pixels) // 2
        start_j = (full_frame.shape[1] - crop_pixels) // 2
        cropped_frame = full_frame[start_i: start_i + crop_pixels, start_j: start_j + crop_pixels]
        frames.append(cropped_frame)

    # Get indices of batches
    frame_indices = np.arange(len(frames))
    original_stride = frame_indices.strides[0]
    overlap = window_size // 2
    num_windows = 1 + (frame_indices.size - window_size) // overlap
    strides = (overlap * original_stride, original_stride)
    indices = as_strided(frame_indices, shape=(num_windows, window_size), strides=strides)

    return np.stack([[frames[index] for index in batch] for batch in indices])

def create_label_mask(labels: list[str]) -> NDArray[np.bool]:
    """
    Create a mask to only use a subset of the Kinetics-400 labels

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

