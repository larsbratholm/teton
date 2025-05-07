"""
Classify actions in video with VideoMAE.
"""

import av
import torch
from transformers import AutoImageProcessor, VideoMAEForVideoClassification

from ..utils import batch_and_crop_frames, create_label_mask
from .args import Arguments, parse_args


def main(args: Arguments) -> None:
    """
    Classify actions in video with VideoMAE.

    Args:
        args: command-line arguments
    """
    # Load model
    model_name = f"MCG-NJU/videomae-{args.model_size}-finetuned-kinetics"
    image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
    model = VideoMAEForVideoClassification.from_pretrained(model_name)
    model.eval()

    # Load video
    container = av.open(args.video_path)

    # Generate batches
    all_videos = batch_and_crop_frames(
        container=container,
        minimum_crop=args.minimum_crop,
        square_crop=args.square_crop,
        seconds_between_frames=4 / 25,
    )

    label_mask = create_label_mask(model.config.id2label.values())
    for window, video in enumerate(all_videos):
        inputs = image_processor(
            list(video),
            return_tensors="pt",
        )

        outputs = model(**inputs)
        logits = outputs.logits
        logits[:, label_mask] = -torch.inf

        # model predicts one of the 400 Kinetics-400 classes
        predicted_label = logits.argmax(-1).item()
        print(f"Window {window}:", model.config.id2label[predicted_label])


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
