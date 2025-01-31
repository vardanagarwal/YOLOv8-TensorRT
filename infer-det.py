from models import TRTModule  # isort:skip
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.cuda.nvtx as nvtx

from config import CLASSES_DET, COLORS
from models.torch_utils import det_postprocess
from models.utils import batch_blob, letterbox, path_to_list


def main(args: argparse.Namespace) -> None:
    nvtx.range_push("Initialization")
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    H, W = Engine.inp_info[0].shape[-2:]
    BATCH_SIZE = args.batch_size

    # set desired output names order
    Engine.set_desired(["num_dets", "bboxes", "scores", "labels"])
    images = path_to_list(args.imgs)
    if args.save:
        save_path = Path(args.out_dir)

        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)

    nvtx.range_pop()  # End Initialization

    # Warmup
    nvtx.range_push("Warmup")
    dummy_input = torch.randn(BATCH_SIZE, 3, H, W, device=device)
    for _ in range(3):
        Engine(dummy_input)
    torch.cuda.synchronize()
    nvtx.range_pop()  # End Warmup

    for i in range(0, len(images), BATCH_SIZE):
        nvtx.range_push(f"Batch {i//BATCH_SIZE}")
        if args.save:
            save_path = Path(args.out_dir)
        batch_images = images[i : i + BATCH_SIZE]
        preprocessed_images = []
        metadata = []
        dwdh_list = []
        nvtx.range_push("Image Loading & Preprocessing CPU")

        # Prepare batch
        for image in batch_images:
            bgr = cv2.imread(str(image))
            if args.save:
                draw = bgr.copy()
            if bgr is None:
                continue
            bgr, ratio, dwdh = letterbox(bgr, (W, H))

            dwdh_list.append(dwdh)
            preprocessed_images.append(bgr)

            metadata.append(
                {
                    "ratio": ratio,
                    # 'dwdh': dwdh,
                    # 'draw': draw,
                }
            )
            if args.save:
                # draw = bgr.copy()
                metadata[-1]["draw"] = draw
                save_image = save_path / image.name
                metadata[-1]["save_path"] = save_image

        nvtx.range_pop()  # End Image Loading & Preprocessing

        if not preprocessed_images:
            continue

        nvtx.range_push("Batch Padding")
        # Pad the batch if necessary
        while len(preprocessed_images) < BATCH_SIZE:
            preprocessed_images.append(preprocessed_images[0])
            metadata.append(metadata[0])
        nvtx.range_pop()  # End Batch Padding

        # Stack tensors into a batch
        nvtx.range_push("Tensor Preparation and Push to GPU")
        batch_tensor = batch_blob(preprocessed_images)
        batch_tensor = batch_tensor.to(device) / 255.0

        # Handle dwdh properly
        dwdh_tensor = torch.tile(
            torch.tensor(dwdh_list, dtype=torch.float32, device=device), (1, 2)
        )
        nvtx.range_pop()  # End Tensor Preparation

        # inference
        nvtx.range_push("Inference")
        data = Engine(batch_tensor)
        nvtx.range_pop()  # End Inference

        nvtx.range_push("Post-processing")
        # Process each image in the batch
        processed_results = []
        for idx in range(len(batch_images)):
            # Extract single image data from batch
            single_data = [d[idx : idx + 1] for d in data]

            bboxes, scores, labels = det_postprocess(single_data)
            # if bboxes.numel() == 0:
            #     print(f'{batch_images[idx]}: no object!')
            #     continue
            if bboxes.numel() > 0:
                bboxes -= dwdh_tensor[idx].view(1, -1)
                bboxes /= metadata[idx]["ratio"]

                processed_results.append(
                    {
                        "bboxes": bboxes,
                        "scores": scores,
                        "labels": labels,
                        "metadata": metadata[idx],
                    }
                )

        for result in processed_results:
            bboxes = result["bboxes"].cpu()
            scores = result["scores"].cpu()
            labels = result["labels"].cpu()
            if args.save:
                draw = result["metadata"]["draw"]
                save_path = result["metadata"]["save_path"]

            # Draw detections
            for bbox, score, label in zip(bboxes, scores, labels):
                bbox = bbox.round().int().tolist()
                cls_id = int(label)
                cls = CLASSES_DET[cls_id]
                color = COLORS[cls]

                text = f"{cls}:{score:.3f}"
                x1, y1, x2, y2 = bbox

                if args.show or args.save:
                    (tw, th), bl = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1
                    )
                    y1_text = min(y1 + 1, draw.shape[0])

                    cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
                    cv2.rectangle(
                        draw,
                        (x1, y1_text),
                        (x1 + tw, y1_text + th + bl),
                        (0, 0, 255),
                        -1,
                    )
                    cv2.putText(
                        draw,
                        text,
                        (x1, y1_text + th),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (255, 255, 255),
                        2,
                    )

            if args.show:
                cv2.imshow("result", draw)
                cv2.waitKey(0)
            if args.save:
                cv2.imwrite(str(save_path), draw)
        nvtx.range_pop()  # End Post-processing
        nvtx.range_pop()  # End Batch


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
