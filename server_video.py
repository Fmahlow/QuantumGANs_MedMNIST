import spaces
import torch
from typing import List
from diffusers import AutoencoderKLWan, WanVACEPipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video
import gradio as gr
import tempfile
import numpy as np
from PIL import Image
import random
from briarmbg import BriaRMBG

model_id = "Wan-AI/Wan2.1-VACE-14B-diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanVACEPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16).to("cuda")

# Initialize background removal model
rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4").to("cuda", dtype=torch.float32)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=2.0)

pipe.load_lora_weights(
   "vrgamedevgirl84/Wan14BT2VFusioniX", 
   weight_name="FusionX_LoRa/Phantom_Wan_14B_FusionX_LoRA.safetensors", 
    adapter_name="phantom"
)
# pipe.load_lora_weights(
#    "vrgamedevgirl84/Wan14BT2VFusioniX", 
#    weight_name="OtherLoRa's/DetailEnhancerV1.safetensors", adapter_name="detailer"
# )
# pipe.set_adapters(["phantom","detailer"], adapter_weights=[1, .9])
# pipe.fuse_lora()



MOD_VALUE = 32
DEFAULT_H_SLIDER_VALUE = 512
DEFAULT_W_SLIDER_VALUE = 896
NEW_FORMULA_MAX_AREA = 480.0 * 832.0 

SLIDER_MIN_H, SLIDER_MAX_H = 128, 896
SLIDER_MIN_W, SLIDER_MAX_W = 128, 896
MAX_SEED = np.iinfo(np.int32).max

FIXED_FPS = 16
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 81 

# Default prompts for different modes - Updated with new mode names
MODE_PROMPTS = {
    "Reference": "the playful penguin picks up the green cat eye sunglasses and puts them on",
    "First - Last Frame": "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective.",
    "Random Transitions": "Various different characters appear and disappear in a fast transition video showcasting their unique features and personalities. The video is about showcasing different dance styles, with each character performing a distinct dance move. The background is a vibrant, colorful stage with dynamic lighting that changes with each dance style. The camera captures close-ups of the dancers' expressions and movements. Highly dynamic, fast-paced music video, with quick cuts and transitions between characters, cinematic, vibrant colors"
}

default_negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards, watermark, text, signature"


def remove_alpha_channel(image: Image.Image) -> Image.Image:
    """
    Remove alpha channel from PIL Image if it exists.
    
    Args:
        image (Image.Image): Input PIL image
        
    Returns:
        Image.Image: Image with alpha channel removed (RGB format)
    """
    if image.mode in ('RGBA', 'LA'):
        # Create a white background
        background = Image.new('RGB', image.size, (255, 255, 255))
        # Paste the image onto the white background using alpha channel as mask
        if image.mode == 'RGBA':
            background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
        else:  # LA mode
            background.paste(image.convert('RGB'), mask=image.split()[-1])
        return background
    elif image.mode == 'P':
        # Convert palette mode to RGB (some palette images have transparency)
        if 'transparency' in image.info:
            image = image.convert('RGBA')
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            return background
        else:
            return image.convert('RGB')
    elif image.mode != 'RGB':
        # Convert any other mode to RGB
        return image.convert('RGB')
    else:
        # Already RGB, return as is
        return image


# @torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h


# @torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


@spaces.GPU()
def run_rmbg(img, sigma=0.0):
    """
    Remove background from image using BriaRMBG model.
    
    Args:
        img (np.ndarray): Input image as numpy array (H, W, C)
        sigma (float): Noise parameter for blending
        
    Returns:
        tuple: (result_image, alpha_mask) where result_image is the image with background removed
    """
    H, W, C = img.shape
    assert C == 3
    k = (256.0 / float(H * W)) ** 0.5
    feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
    feed = numpy2pytorch([feed]).to(device="cuda", dtype=torch.float32)
    alpha = rmbg(feed)[0][0]
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
    alpha = alpha.movedim(1, -1)[0]
    alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
    result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
    return result.clip(0, 255).astype(np.uint8), alpha


def remove_background_from_image(image: Image.Image) -> Image.Image:
    """
    Remove background from PIL Image using RMBG model.
    
    Args:
        image (Image.Image): Input PIL image
        
    Returns:
        Image.Image: Image with background removed (transparent background)
    """
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Remove background using RMBG
    result_array, alpha_mask = run_rmbg(img_array)
    
    # Convert back to PIL with alpha channel
    result_image = Image.fromarray(result_array)
    
    # Create RGBA image with alpha mask
    if result_image.mode != 'RGBA':
        result_image = result_image.convert('RGBA')
    
    # Handle alpha mask dimensions and convert to PIL
    # The alpha_mask might have extra dimensions, so squeeze and ensure 2D
    alpha_mask_2d = np.squeeze(alpha_mask)
    if alpha_mask_2d.ndim > 2:
        # If still more than 2D, take the first channel
        alpha_mask_2d = alpha_mask_2d[:, :, 0] if alpha_mask_2d.shape[-1] == 1 else alpha_mask_2d[:, :, 0]
    
    # Convert to uint8 and create PIL Image without deprecated mode parameter
    alpha_array = (alpha_mask_2d * 255).astype(np.uint8)
    alpha_pil = Image.fromarray(alpha_array, 'L')
    result_image.putalpha(alpha_pil)
    
    return result_image


def _calculate_new_dimensions_wan(pil_image, mod_val, calculation_max_area,
                                 min_slider_h, max_slider_h,
                                 min_slider_w, max_slider_w,
                                 default_h, default_w):
    orig_w, orig_h = pil_image.size
    if orig_w <= 0 or orig_h <= 0:
        return default_h, default_w

    aspect_ratio = orig_h / orig_w
    
    calc_h = round(np.sqrt(calculation_max_area * aspect_ratio))
    calc_w = round(np.sqrt(calculation_max_area / aspect_ratio))

    calc_h = max(mod_val, (calc_h // mod_val) * mod_val)
    calc_w = max(mod_val, (calc_w // mod_val) * mod_val)
    
    new_h = int(np.clip(calc_h, min_slider_h, (max_slider_h // mod_val) * mod_val))
    new_w = int(np.clip(calc_w, min_slider_w, (max_slider_w // mod_val) * mod_val))
    
    return new_h, new_w

def handle_gallery_upload_for_dims_wan(gallery_images, current_h_val, current_w_val):
    if gallery_images is None or len(gallery_images) == 0:
        return gr.update(value=DEFAULT_H_SLIDER_VALUE), gr.update(value=DEFAULT_W_SLIDER_VALUE)
    try:
        # Use the first image to calculate dimensions
        first_image = gallery_images[0][0]
        # Remove alpha channel before calculating dimensions
        first_image = remove_alpha_channel(first_image)
        new_h, new_w = _calculate_new_dimensions_wan(
            first_image, MOD_VALUE, NEW_FORMULA_MAX_AREA,
            SLIDER_MIN_H, SLIDER_MAX_H, SLIDER_MIN_W, SLIDER_MAX_W,
            DEFAULT_H_SLIDER_VALUE, DEFAULT_W_SLIDER_VALUE
        )
        return gr.update(value=new_h), gr.update(value=new_w)
    except Exception as e:
        gr.Warning("Error attempting to calculate new dimensions")
        return gr.update(value=DEFAULT_H_SLIDER_VALUE), gr.update(value=DEFAULT_W_SLIDER_VALUE)

def update_prompt_from_mode(mode):
    """Update the prompt based on the selected mode"""
    return MODE_PROMPTS.get(mode, "")


def prepare_video_and_mask_Ref2V(height: int, width: int, num_frames: int):
    frames = []
    # Ideally, this should be 127.5 to match original code, but they perform computation on numpy arrays
    # whereas we are passing PIL images. If you choose to pass numpy arrays, you can set it to 127.5 to
    # match the original code.
    frames.extend([Image.new("RGB", (width, height), (128, 128, 128))] * (num_frames))
    mask_white = Image.new("L", (width, height), 255)
    mask = [mask_white] * (num_frames)
    return frames, mask

def prepare_video_and_mask_FLF2V(first_img: Image.Image, last_img: Image.Image, height: int, width: int, num_frames: int):
    # Remove alpha channels before processing
    first_img = remove_alpha_channel(first_img)
    last_img = remove_alpha_channel(last_img)
    
    first_img = first_img.resize((width, height))
    last_img = last_img.resize((width, height))

    # Use independent placeholder frames to avoid aliasing the first/last images
    neutral_frame = Image.new("RGB", (width, height), (128, 128, 128))
    frames = [first_img]
    frames.extend(neutral_frame.copy() for _ in range(max(0, num_frames - 2)))
    frames.append(last_img)

    # Build masks with independent Image instances to guarantee both endpoints are honoured
    mask = [Image.new("L", (width, height), 0)]
    mask.extend(Image.new("L", (width, height), 255) for _ in range(max(0, num_frames - 2)))
    mask.append(Image.new("L", (width, height), 0))
    return frames, mask

def calculate_random2v_frame_indices(num_images: int, num_frames: int) -> List[int]:
    """
    Calculate evenly spaced frame indices for Random2V mode.
    
    Args:
        num_images (int): Number of input images
        num_frames (int): Total number of frames in the video
    
    Returns:
        List[int]: Frame indices where images should be placed
    """
    if num_images <= 0:
        return []
    
    if num_images == 1:
        # Single image goes in the middle
        return [num_frames // 2]
    
    if num_images >= num_frames:
        # More images than frames, use every frame
        return list(range(num_frames))
    
    # Calculate evenly spaced indices
    # We want to distribute images across the full duration
    indices = []
    step = (num_frames - 1) / (num_images - 1)
    
    for i in range(num_images):
        frame_idx = int(round(i * step))
        # Ensure we don't exceed num_frames - 1
        frame_idx = min(frame_idx, num_frames - 1)
        indices.append(frame_idx)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_indices = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)
    
    return unique_indices

def prepare_video_and_mask_Random2V(images: List[Image.Image], frame_indices: List[int], height: int, width: int, num_frames: int):
    # Remove alpha channels from all images before processing
    images = [remove_alpha_channel(img) for img in images]
    images = [img.resize((width, height)) for img in images]
    # Ideally, this should be 127.5 to match original code, but they perform computation on numpy arrays
    # whereas we are passing PIL images. If you choose to pass numpy arrays, you can set it to 127.5 to
    # match the original code.
    frames = [Image.new("RGB", (width, height), (128, 128, 128))] * num_frames
    
    mask_black = Image.new("L", (width, height), 0)
    mask_white = Image.new("L", (width, height), 255)
    mask = [mask_white] * num_frames
    
    for img, idx in zip(images, frame_indices):
        assert idx < num_frames, f"Frame index {idx} exceeds num_frames {num_frames}"
        frames[idx] = img
        mask[idx] = mask_black
    
    return frames, mask

def get_duration(gallery_images, mode, prompt, height, width, 
                   negative_prompt, duration_seconds,
                   guidance_scale, steps,
                   seed, randomize_seed, remove_bg,
                   progress):
    # Add extra time if background removal is enabled
    base_duration = 60
    if steps > 4 and duration_seconds > 2:
        base_duration = 90
    elif steps > 4 or duration_seconds > 2:
        base_duration = 75
    
    # Add extra time for background removal processing
    if mode == "Reference" and remove_bg:  # Updated to use new mode name
        base_duration += 30
    
    return base_duration

@spaces.GPU(duration=get_duration)
def generate_video(gallery_images, mode, prompt, height, width, 
                   negative_prompt=default_negative_prompt, duration_seconds = 2,
                   guidance_scale = 1, steps = 4,
                   seed = 42, randomize_seed = False, remove_bg = False,
                   progress=gr.Progress(track_tqdm=True)):
    """
    Generate a video from gallery images using the selected mode.
    
    Args:
        gallery_images (list): List of PIL images from the gallery
        mode (str): Processing mode - "Reference", "first - last frame", or "random transitions"
        prompt (str): Text prompt describing the desired animation
        height (int): Target height for the output video
        width (int): Target width for the output video
        negative_prompt (str): Negative prompt to avoid unwanted elements
        duration_seconds (float): Duration of the generated video in seconds
        guidance_scale (float): Controls adherence to the prompt
        steps (int): Number of inference steps
        seed (int): Random seed for reproducible results
        randomize_seed (bool): Whether to use a random seed
        remove_bg (bool): Whether to remove background from images (reference mode only)
        progress (gr.Progress): Gradio progress tracker
    
    Returns:
        tuple: (video_path, current_seed)
    """
    if gallery_images is None or len(gallery_images) == 0:
        raise gr.Error("Please upload at least one image to the gallery.")
    else:
        # Process images: remove background if requested (reference mode only), then remove alpha channels
        processed_images = []
        for img in gallery_images:
            image = img[0]  # Extract PIL image from gallery format
            
            # Apply background removal only for reference mode if checkbox is checked
            if mode == "Reference" and remove_bg:  # Updated mode name
                image = remove_background_from_image(image)
            
            # Always remove alpha channels to ensure RGB format
            image = remove_alpha_channel(image)
            processed_images.append(image)
        
        gallery_images = processed_images

    if mode == "First - Last Frame" and len(gallery_images) >= 2:  # Updated mode name
        gallery_images = [gallery_images[0], gallery_images[-1]]
    elif mode == "First - Last Frame" and len(gallery_images) < 2:  # Updated mode name
        raise gr.Error("First - Last Frame mode requires at least 2 images, but only {} were supplied.".format(len(gallery_images)))

    target_h = max(MOD_VALUE, (int(height) // MOD_VALUE) * MOD_VALUE)
    target_w = max(MOD_VALUE, (int(width) // MOD_VALUE) * MOD_VALUE)
    
    num_frames = np.clip(int(round(duration_seconds * FIXED_FPS)), MIN_FRAMES_MODEL, MAX_FRAMES_MODEL)
    
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)

    # Process images based on the selected mode
    if mode == "First - Last Frame":  # Updated mode name
        frames, mask = prepare_video_and_mask_FLF2V(
            first_img=gallery_images[0], 
            last_img=gallery_images[1], 
            height=target_h, 
            width=target_w, 
            num_frames=num_frames
        )
        reference_images = None
    elif mode == "Reference":  # Updated mode name
        frames, mask = prepare_video_and_mask_Ref2V(height=target_h, width=target_w, num_frames=num_frames)
        reference_images = gallery_images
    else:  # mode == "random transitions"  # Updated mode name
        # Calculate dynamic frame indices based on number of images and frames
        frame_indices = calculate_random2v_frame_indices(len(gallery_images), num_frames)
        
        frames, mask = prepare_video_and_mask_Random2V(
            images=gallery_images, 
            frame_indices=frame_indices,
            height=target_h, 
            width=target_w, 
            num_frames=num_frames
        )
        reference_images = None

    with torch.inference_mode():
        output_frames_list = pipe(
            video=frames,
            mask=mask,
            reference_images=reference_images,
            prompt=prompt, 
            negative_prompt=negative_prompt,
            height=target_h, 
            width=target_w, 
            num_frames=num_frames,
            guidance_scale=float(guidance_scale), 
            num_inference_steps=int(steps),
            generator=torch.Generator(device="cuda").manual_seed(current_seed)
        ).frames[0]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        video_path = tmpfile.name
    export_to_video(output_frames_list, video_path, fps=FIXED_FPS)
    return video_path, current_seed

