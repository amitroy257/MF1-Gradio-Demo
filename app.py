"""Gradio demo for MF-1 (Multimodal Flow, hustvl/Multimodal-Flow).

Built the way the Gradio quickstart shows: plain Python functions wrapped with
gr.Interface, one tab per task, launched with share=True for a public link.

    export MF_CHECKPOINT=/path/to/MF_weights/MF/sft
    export MF_ASSETS_ROOT=/path/to/assets      # folder that contains scale_rae_decoder/
    python app.py --share

Author: Amit Bikram Roy
"""

import argparse
import os
import random
import traceback
from dataclasses import replace

import gradio as gr
import torch
from torchvision.transforms.functional import to_pil_image

from mf.data.images import preprocess_image
from mf.inference.pipeline import MFPipeline
from mf.inference.protocol import GenerationConfig

# ---------------------------------------------------------------------------
# Load the model once
# ---------------------------------------------------------------------------

CHECKPOINT = os.environ.get("MF_CHECKPOINT")
if not CHECKPOINT:
    raise SystemExit("Set MF_CHECKPOINT to the MF checkpoint folder, for example MF_weights/MF/sft")

# MF-1 runs in bf16. Older GPUs such as the T4 (Colab, Kaggle) have no bf16
# hardware, so there we keep the weights in fp32 and run the math in fp16.
NATIVE_BF16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (8, 0)
DTYPE = torch.bfloat16 if NATIVE_BF16 else torch.float16

pipe = MFPipeline.from_checkpoint(CHECKPOINT, device="cuda", weights="ema")
if DTYPE is torch.float16:
    pipe.bundle.model.float()
    pipe.bundle.text_decoder.float()
    pipe._sampler = replace(pipe._sampler, amp_dtype=torch.float16)

DEFAULTS = pipe.default_sampler_config
VISION = pipe.config.codecs.vision


def _encoders_to_dtype():
    """MF builds its encoders in bf16 on first use. On GPUs without bf16 we run
    the text encoder in fp16 and keep the SigLIP2 vision encoder in fp32: it runs
    outside autocast and is small, and fp32 avoids any fp16 overflow there."""
    if DTYPE is torch.float16:
        if pipe.bundle._text_encoder is not None:
            pipe.bundle._text_encoder.to(torch.float16)
        if pipe.bundle._vision_encoder is not None:
            pipe.bundle._vision_encoder.to(torch.float32)


def _seed(seed):
    seed = int(seed)
    return random.randint(0, 2**31 - 1) if seed < 0 else seed


# ---------------------------------------------------------------------------
# The functions Gradio wraps (one per task)
# ---------------------------------------------------------------------------

def text_to_image(prompt, steps, cfg, seed):
    _ = pipe.bundle.text_encoder
    _encoders_to_dtype()
    config = GenerationConfig(num_inference_steps=int(steps), cfg_scale=float(cfg), seed=_seed(seed))
    image = pipe.generate_image((prompt,), config=config)[0]
    return to_pil_image(((image.float().cpu() + 1) / 2).clamp(0, 1))


def image_to_text(image, question, max_tokens, seed):
    if image is None:
        raise gr.Error("Please upload an image first.")
    _ = pipe.bundle.text_encoder, pipe.bundle.vision_encoder
    _encoders_to_dtype()
    tensor = preprocess_image(
        image.convert("RGB"),
        resolution=VISION.encoder_input_resolution,
        policy=getattr(VISION, "image_preprocessing", "legacy_center_crop_bicubic_v1"),
    )
    config = GenerationConfig(seed=_seed(seed))
    try:
        answer = pipe.caption((tensor,), prompt=question or "Describe this image.",
                              target_length=int(max_tokens), config=config)[0]
    except Exception as error:
        traceback.print_exc()
        raise gr.Error(f"{type(error).__name__}: {error}")
    answer = answer.strip()
    print(f"[image_to_text] question={question!r} answer={answer!r}", flush=True)
    return answer or "(the model returned an empty answer, try another seed or a longer length)"


def continue_text(prompt, max_tokens, seed):
    _ = pipe.bundle.text_encoder
    _encoders_to_dtype()
    config = GenerationConfig(seed=_seed(seed))
    return prompt + pipe.complete_text((prompt,), target_length=int(max_tokens), config=config)[0]


# ---------------------------------------------------------------------------
# The UI: one gr.Interface per task, shown as tabs
# ---------------------------------------------------------------------------

seed_box = lambda: gr.Number(value=-1, precision=0, label="Seed (-1 = random)")

t2i = gr.Interface(
    fn=text_to_image,
    inputs=[
        gr.Textbox(label="Prompt", value="A quiet observatory above the clouds."),
        gr.Slider(8, 128, value=DEFAULTS.num_inference_steps, step=1, label="Flow steps"),
        gr.Slider(0.0, 10.0, value=float(DEFAULTS.cfg_scale), step=0.1, label="CFG scale"),
        seed_box(),
    ],
    outputs=gr.Image(label="Generated image", type="pil"),
    examples=[
        ["A quiet observatory above the clouds.", 64, float(DEFAULTS.cfg_scale), 1],
        ["A red fox sitting in fresh snow, morning light.", 64, float(DEFAULTS.cfg_scale), 2],
        ["A small fishing boat on a river in Bangladesh at sunset.", 64, float(DEFAULTS.cfg_scale), 3],
    ],
    cache_examples=False,
    flagging_mode="never",
    api_name="text_to_image",
)

i2t = gr.Interface(
    fn=image_to_text,
    inputs=[
        gr.Image(label="Image", type="pil"),
        gr.Textbox(label="Question or instruction", value="Describe this image."),
        gr.Slider(16, 256, value=96, step=8, label="Max new tokens"),
        seed_box(),
    ],
    outputs=gr.Textbox(label="Answer", lines=6),
    flagging_mode="never",
    api_name="image_to_text",
)

text = gr.Interface(
    fn=continue_text,
    inputs=[
        gr.Textbox(label="Start of the text", value="The night sky over Dhaka", lines=3),
        gr.Slider(16, 256, value=96, step=8, label="Max new tokens"),
        seed_box(),
    ],
    outputs=gr.Textbox(label="Continuation", lines=8),
    flagging_mode="never",
    api_name="continue_text",
)

precision = "bf16" if DTYPE is torch.bfloat16 else "fp16"
gpu = torch.cuda.get_device_name(0)

demo = gr.TabbedInterface(
    [t2i, i2t, text],
    ["Text to Image", "Image to Text", "Text Continuation"],
    title="MF-1: Multimodal Flow demo",
)
with demo:
    gr.Markdown(
        f"One 1.6B model for image generation, image understanding and text, all as continuous flow. "
        f"[Paper](https://arxiv.org/abs/2609.40362) · [Code](https://github.com/hustvl/Multimodal-Flow) · "
        f"Running on **{gpu}** in **{precision}**."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    demo.queue(default_concurrency_limit=1).launch(share=args.share, server_port=args.port)
