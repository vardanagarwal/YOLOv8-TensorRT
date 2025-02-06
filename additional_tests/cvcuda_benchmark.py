from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Union
import cv2
import numpy as np
import torch
from numpy import ndarray


def generate_sample_frame(width=1920, height=1080):
    """Generate a sample frame similar to what you'd get from kafka"""
    # Create a sample image
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    # Encode it to jpg bytes like it would be in kafka
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

batch_size = 16

# Generate a batch of frames
def generate_batch(batch_size=batch_size):
    return [generate_sample_frame() for _ in range(batch_size)]

def letterbox(
    im: ndarray,
    new_shape: Union[Tuple, List] = (640, 640),
    color: Union[Tuple, List] = (114, 114, 114),
) -> Tuple[ndarray, float, Tuple[float, float]]:
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # new_shape: [width, height]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    # Compute padding [width, height]
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, r, (dw, dh)

def batch_blob(im: List[ndarray]) -> torch.Tensor:
    im = np.stack(im)
    im = im[..., ::-1].transpose((0, 3, 1, 2))
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im)
    return im

def process_single_image(image, W, H):
    """Process a single image in a separate thread"""
    bgr = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None, None

    # Perform letterbox operation
    bgr, ratio, dwdh = letterbox(bgr, (W, H))

    metadata = {
        "ratio": ratio,
    }


    return bgr, metadata, dwdh

def prepare_batch_threaded(batch_images, W, H, max_workers=16):
    """Process batch of images using multiple threads"""
    preprocessed_images = []
    metadata = []
    dwdh_list = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all image processing tasks
        future_to_image = {
            executor.submit(process_single_image, image, W, H): image
            for image in batch_images
        }

        # Collect results as they complete
        for future in future_to_image:
            bgr, meta, dwdh = future.result()
            if bgr is not None:
                preprocessed_images.append(bgr)
                metadata.append(meta)
                dwdh_list.append(dwdh)

    return preprocessed_images, metadata, dwdh_list

ts1 = [0, 0, 0, 0, 0, 0, 0]
device = torch.device('cuda')
def current_approach(batch):
    preprocessed_images = []
    metadata = []
    dwdh_list = []
    t0 = time.perf_counter()
    preprocessed_images, metadata, dwdh_list = prepare_batch_threaded(
                batch, 640, 640
            )
    t1 = time.perf_counter()

    if not preprocessed_images:
        return
    while len(preprocessed_images) < batch_size:
        preprocessed_images.append(preprocessed_images[0])
        metadata.append(metadata[0])
    t2 = time.perf_counter()

    batch_tensor = batch_blob(preprocessed_images)
    batch_tensor = batch_tensor.to(device) / 255.0
    t3 = time.perf_counter()

    dwdh_tensor = torch.tile(
        torch.tensor(dwdh_list, dtype=torch.float32, device=device), (1, 2)
    )
    t4 = time.perf_counter()
    ts1[0] += t1-t0
    ts1[1] += t2-t1
    ts1[2] += t3-t2
    ts1[3] += t4-t3
    return

from torchvision.io import decode_jpeg
import cvcuda

def cvcuda_letterbox(
    im: torch.Tensor,
    new_shape: Union[Tuple, List] = (640, 640),
    color: Union[Tuple, List] = (114, 114, 114),
) -> Tuple[ndarray, float, Tuple[float, float]]:
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape  # current shape [batch, height, width, channels]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # new_shape: [width, height]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[2], new_shape[1] / shape[1])
    # Compute padding [width, height]
    new_unpad = int(round(shape[2] * r)), int(round(shape[1] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cvcuda.resize(im, (shape[0], new_unpad[1], new_unpad[0], shape[3]), cvcuda.Interp.LINEAR)
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # im = cv2.copyMakeBorder(
    #     im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    # )  # add border
    im = cvcuda.copymakeborder(
        src=im, 
        top=top, 
        bottom=bottom, 
        left=left, 
        right=right, 
        border_mode=cvcuda.Border.CONSTANT, 
        border_value=list(color)
    )
    return im, r, (dw, dh)


def prepare_frames_threaded(batch, max_workers=16):
    frames = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all image processing tasks
        future_to_frame = {
            executor.submit(cv2.imdecode, np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR): frame
            for frame in batch
        }

        # Collect results as they complete
        for future in future_to_frame:
            frames.append(future.result())
    return np.stack(frames)

device = torch.device('cuda')
ts = [0, 0, 0, 0, 0, 0, 0]
def proposed_approach(batch):
    preprocessed_images = []
    metadata = []
    dwdh_list = []
    t0 = time.perf_counter()
    frames = prepare_frames_threaded(batch)
    # print(len(frames))
    t1 = time.perf_counter()
    images = torch.from_numpy(frames).to(device)
    t2 = time.perf_counter()
    images = images.contiguous()
    t3 = time.perf_counter()
    # print(images.shape)
    # next two lines are maybe not required in the main.py code
    # images_cpu = images.cpu().numpy()
    # preprocessed_images = images_cpu[..., ::-1]
    cvcuda_image = cvcuda.as_tensor(images, "NHWC")
    t4 = time.perf_counter()
    ims, r, dwdh = cvcuda_letterbox(cvcuda_image, (640, 640))
    t5 = time.perf_counter()
    ims = cvcuda.convertto(ims, np.float32, scale=1/255)
    ims = cvcuda.reformat(ims, "NCHW")
    t6 = time.perf_counter()
    ims_torch = torch.from_dlpack(ims.cuda().__dlpack__())
    t7 = time.perf_counter()
    ts[0] += t1-t0
    ts[1] += t2-t1
    ts[2] += t3-t2
    ts[3] += t4-t3
    ts[4] += t5-t4
    ts[5] += t6-t5
    ts[6] += t7-t6
    # dwdh_tensor = torch.tile(
    #     torch.tensor([dwdh] * batch_size, dtype=torch.float32, device=device), (1, 2)
    # )
    return


import time

# Benchmark timing
num_batches = 25
batch = generate_batch()
print(f"\nBenchmarking {num_batches} batches...")
start = time.perf_counter()
for _ in range(num_batches):
    current_approach(batch)
end = time.perf_counter()
print(f"Current approach: {end - start:.2f} seconds")
print(ts1)

start = time.perf_counter()
batch = generate_batch()
for _ in range(num_batches):
    proposed_approach(batch)
end = time.perf_counter()
print(f"Proposed approach: {end - start:.2f} seconds")
print(ts)