import av
import torch
import numpy as np
import glob
import math

from transformers import AutoImageProcessor, VideoMAEForVideoClassification
from huggingface_hub import hf_hub_download

from ..utils import batch_and_crop_frames, create_label_mask
from .args import parse_args, Arguments


def main(args: Arguments):
    # Load model
    model_name = f"MCG-NJU/videomae-{args.model_size}-finetuned-kinetics"
    image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
    model = VideoMAEForVideoClassification.from_pretrained(model_name)
    model.eval()

    # Load video
    container = av.open(args.video_path)

    # Generate batches
    all_videos = batch_and_crop_frames(container=container, crop_pixels=args.crop_pixels, seconds_between_frames=4/25)

    n_chunks = all_videos.shape[0]
    n_batches = math.ceil(n_chunks / args.batch_size)
    window = 0
    for batch in range(n_batches):
        batch_video = all_videos[args.batch_size * batch: args.batch_size * (batch + 1)]
        inputs = image_processor(list(batch_video.reshape((-1,) + batch_video.shape[2:])), return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].view((batch_video.shape[0], batch_video.shape[1],) + inputs["pixel_values"].shape[2:])

        outputs = model(**inputs)
        logits = outputs.logits
        label_mask = create_label_mask(model.config.id2label.values())
        logits[:, label_mask] = -torch.inf

        # model predicts one of the 400 Kinetics-400 classes
        predicted_labels = logits.argmax(-1).numpy()
        for label in predicted_labels:
            window += 1
            print(f"Window {window}:", model.config.id2label[label])
    #label_mask = create_label_mask(model.config.id2label.values())
    #for window, video in enumerate(batch_video):
    #    inputs = image_processor(list(video), return_tensors="pt")

    #    outputs = model(**inputs)
    #    logits = outputs.logits
    #    logits[:, label_mask] = -torch.inf

    #    predicted_label = logits.argmax(-1).item()
    #    print(f"Window {window}:", model.config.id2label[predicted_label])

if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
