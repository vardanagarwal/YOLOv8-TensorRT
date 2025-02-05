from models import TRTModule  # isort:skip
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
# import torch.cuda.nvtx as nvtx
from concurrent.futures import ThreadPoolExecutor

from config import CLASSES_DET, COLORS
from models.torch_utils import det_postprocess
from models.utils import batch_blob, letterbox, path_to_list
from nvtx_profiler import profiler


def process_single_image(image, W, H, args, save_path):
    """Process a single image in a separate thread"""
    bgr = cv2.imread(str(image))
    if bgr is None:
        return None, None, None

    if args.save:
        draw = bgr.copy()
    else:
        draw = None

    # Perform letterbox operation
    bgr, ratio, dwdh = letterbox(bgr, (W, H))

    metadata = {
        "ratio": ratio,
    }

    if args.save:
        metadata["draw"] = draw
        metadata["save_path"] = save_path / image.name

    return bgr, metadata, dwdh


def prepare_batch_threaded(batch_images, W, H, args, save_path, max_workers=16):
    """Process batch of images using multiple threads"""
    preprocessed_images = []
    metadata = []
    dwdh_list = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all image processing tasks
        future_to_image = {
            executor.submit(process_single_image, image, W, H, args, save_path): image
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


def main(args: argparse.Namespace) -> None:
    with profiler.range("Initialization"):
        device = torch.device(args.device)
        Engine = TRTModule(args.engine, device)
        H, W = Engine.inp_info[0].shape[-2:]
        BATCH_SIZE = args.batch_size

        print("\nDiagnostic Information:")
        print("-" * 40)
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability()}")
        print(f"FP16 Mode: {args.half}")
        print(f"Batch Size: {BATCH_SIZE}")
        print(f"Input Resolution: {W}x{H}")
        print("-" * 40 + "\n")

        Engine.set_desired(["num_dets", "bboxes", "scores", "labels"])
        images = path_to_list(args.imgs)
        if args.save:
            save_path = Path(args.out_dir)
            if not save_path.exists():
                save_path.mkdir(parents=True, exist_ok=True)

    # Warmup
    with profiler.range("Warmup"):
        print("Starting warmup...")
        warmup_input = torch.randn(BATCH_SIZE, 3, H, W, device=device, 
                               dtype=torch.float16 if args.half else torch.float32)
        # Regular warmup
        for _ in range(3):
            with torch.cuda.stream(Engine.stream):
                output = Engine(warmup_input)
                torch.cuda.current_stream().synchronize()
        print("Warmup complete\n")

    print(f"Processing {len(images)} images in {(len(images) + BATCH_SIZE - 1) // BATCH_SIZE} batches")

    for i in range(0, len(images), BATCH_SIZE):
        with profiler.range("Batch"):
            batch_images = images[i : i + BATCH_SIZE]
            
            with profiler.range("Image Loading & Preprocessing"):
                preprocessed_images, metadata, dwdh_list = prepare_batch_threaded(
                    batch_images, W, H, args, save_path if args.save else None
                )

                # Pad batch if needed
                while len(preprocessed_images) < BATCH_SIZE:
                    preprocessed_images.append(preprocessed_images[0])
                    metadata.append(metadata[0])
                    dwdh_list.append(dwdh_list[0])

            with profiler.range("Tensor Preparation"):
                with torch.cuda.stream(Engine.stream):
                    # Prepare input tensor with proper memory alignment
                    batch_tensor = batch_blob(preprocessed_images)
                    
                    if args.half:
                        batch_tensor = batch_tensor.to(device).half() / 255.0
                    else:
                        batch_tensor = batch_tensor.to(device) / 255.0
                    
                    # Prepare dwdh tensor efficiently
                    dwdh_tensor = torch.tensor(dwdh_list, 
                                             dtype=torch.float16 if args.half else torch.float32,
                                             device=device)
                    dwdh_tensor = torch.tile(dwdh_tensor, (1, 2))

            with profiler.range("Inference"):
                outputs = Engine(batch_tensor)

            with profiler.range("Post-processing"):
                processed_results = []
                for idx in range(len(batch_images)):
                    single_data = [d[idx : idx + 1] for d in outputs]
                    bboxes, scores, labels = det_postprocess(single_data)
                    
                    if bboxes.numel() > 0:
                        bboxes -= dwdh_tensor[idx].view(1, -1)
                        bboxes /= metadata[idx]["ratio"]
                        processed_results.append({
                            "bboxes": bboxes,
                            "scores": scores,
                            "labels": labels,
                            "metadata": metadata[idx],
                        })

                # Visualization code...
                for result in processed_results:
                    if not args.save and not args.show:
                        continue
                        
                    bboxes = result["bboxes"].cpu()
                    scores = result["scores"].cpu()
                    labels = result["labels"].cpu()
                    
                    if args.save:
                        draw = result["metadata"]["draw"]
                        save_path = result["metadata"]["save_path"]
                        
                        for bbox, score, label in zip(bboxes, scores, labels):
                            bbox = bbox.round().int().tolist()
                            cls_id = int(label)
                            cls = CLASSES_DET[cls_id]
                            color = COLORS[cls]
                            
                            text = f"{cls}:{score:.3f}"
                            x1, y1, x2, y2 = bbox
                            
                            if args.show or args.save:
                                (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
                                y1_text = min(y1 + 1, draw.shape[0])
                                
                                cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
                                cv2.rectangle(draw, (x1, y1_text), (x1 + tw, y1_text + th + bl), (0, 0, 255), -1)
                                cv2.putText(draw, text, (x1, y1_text + th), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                                
                        if args.show:
                            cv2.imshow("result", draw)
                            cv2.waitKey(0)
                        if args.save:
                            cv2.imwrite(str(save_path), draw)

    profiler.print_stats()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, help="Engine file")
    parser.add_argument("--imgs", type=str, help="Images file")
    parser.add_argument(
        "--show", action="store_true", help="Show the detection results"
    )
    parser.add_argument(
        "--out-dir", type=str, default="./output", help="Path to output file"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="TensorRT infer device"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="TensorRT batch size")
    parser.add_argument(
        "--save", action="store_true", help="Save the detection result images"
    )
    parser.add_argument(
        "--multi-thread", action="store_true", help="Use multi-threading"
    )
    parser.add_argument(
        "--half", action="store_true", help="Use half precision for inference"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    from time import time

    args = parse_args()
    start = time()
    main(args)
    end = time()
    total_images = len(path_to_list(args.imgs))
    total_time = end - start

    print(f"Total time: {total_time:.2f} seconds")
    print(f"Images per second: {total_images/total_time:.2f}")
    print(
        f"Average time per batch: {total_time/(total_images/args.batch_size):.3f} seconds"
    )
