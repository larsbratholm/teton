"""
Command-line arguments.
"""

import argparse
from typing import Literal

from pydantic import BaseModel, Field


class Arguments(BaseModel):
    """
    Dataclass for model options.

    Args:
        video_path: the location of the video input file
        minimum_crop: the minimum number of pixels to include in the crops
        model_size: which VideoMAE to use.
        square_crop: Use a central square crop instead of a dynamic one
    """

    video_path: str
    minimum_crop: int = Field(default=600, ge=224)
    model_size: Literal["small", "base", "large", "huge"] = "huge"
    square_crop: bool = False


def parse_args() -> Arguments:
    """
    Parse command-line arguments.

    Returns:
        Options for the VideoMAE classification
    """
    parser = argparse.ArgumentParser(description="Classify actions with VideoMAE")
    parser.add_argument(
        "video_path",
        type=str,
        help="Location of video input file.",
    )
    parser.add_argument(
        "--minimum_crop",
        type=int,
        help="Minimum number of pixels to include in the crop.",
    )

    parser.add_argument(
        "--model_size",
        type=str,
        choices=["small", "base", "large", "huge"],
        help="Model size to use.",
    )

    parser.add_argument("--square_crop", action="store_true")

    args = Arguments(
        **{
            key: value
            for key, value in vars(parser.parse_args()).items()
            if value is not None
        }
    )

    return args
