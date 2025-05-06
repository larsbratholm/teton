from pydantic import BaseModel, Field
from typing import Literal
import argparse


# TODO support timesformer as well
class Arguments(BaseModel):
    """
    Dataclass for model options.

    Args:
        video_path: the location of the video input file
        crop_pixels: the number of pixels to include in a square crop
        model_size: which VideoMAE to use.
    """
    video_path: str
    crop_pixels: int = Field(default=1400, ge=224)
    model_size: Literal["small", "base", "large", "huge"] = "base"
    batch_size: int = Field(default=1, ge=1)

def parse_args() -> Arguments:
    parser = argparse.ArgumentParser(description="Classify actions with VideoMAE")
    parser.add_argument(
        "video_path",
        type=str,
        help="Location of video input file.",
    )
    parser.add_argument(
        "--crop_pixels",
        type=int,
        help="Number of pixels to include in a square crop."
    )

    parser.add_argument(
        "--model_size",
        type=str,
        choices=["small", "base", "large", "huge"],
        help="Model size to use.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size to use",
    )


    args = Arguments(**{key: value for key, value in vars(parser.parse_args()).items() if value is not None})

    return args
