from models import TRTModule  # isort:skip
import argparse
from pathlib import Path

import cv2
import torch

from config import CLASSES_DET, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    H, W = Engine.inp_info[0].shape[-2:]
    BATCH_SIZE = args.batch_size

    # set desired output names order
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

    images = path_to_list(args.imgs)
    save_path = Path(args.out_dir)

    if not args.show and not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    for i in range(0, len(images), BATCH_SIZE):
        batch_images = images[i:i + BATCH_SIZE]
        batch_tensors = []
        batch_ratios = []
        batch_dwdhs = []
        batch_draws = []
        batch_save_paths = []

        # Prepare batch
        for image in batch_images:
            save_image = save_path / image.name
            bgr = cv2.imread(str(image))
            if bgr is None:
                continue
            draw = bgr.copy()
            bgr, ratio, dwdh = letterbox(bgr, (W, H))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = blob(rgb, return_seg=False)
            
            # Convert to tensor once and expand dwdh to match bbox format
            tensor = torch.asarray(tensor, device=device)
            dwdh = torch.tensor(dwdh, dtype=torch.float32, device=device)
            # Expand dwdh to match bbox format [left, top, right, bottom]
            dwdh = torch.tile(dwdh, (2,))  # This makes it [dx, dy, dx, dy]
            
            batch_tensors.append(tensor)
            batch_ratios.append(ratio)
            batch_dwdhs.append(dwdh)
            batch_draws.append(draw)
            batch_save_paths.append(save_image)

        if not batch_tensors:
            continue

        # Pad the batch if necessary
        while len(batch_tensors) < BATCH_SIZE:
            batch_tensors.append(batch_tensors[0])
            batch_ratios.append(batch_ratios[0])
            batch_dwdhs.append(batch_dwdhs[0])
            batch_draws.append(batch_draws[0])
            batch_save_paths.append(batch_save_paths[0])

        # Stack tensors into a batch
        tensor = torch.stack(batch_tensors)
        dwdh = torch.stack(batch_dwdhs) * 2  # Multiply by 2 after stacking

        # inference
        data = Engine(tensor)

        # Process each image in the batch
        for idx in range(len(batch_images)):
            # Extract single image data from batch
            single_data = [d[idx:idx+1] for d in data]
            
            bboxes, scores, labels = det_postprocess(single_data)
            # if bboxes.numel() == 0:
            #     print(f'{batch_images[idx]}: no object!')
            #     continue

            # Apply scaling
            bboxes -= dwdh[idx].view(1, -1)  # Reshape dwdh to match bboxes dimension
            bboxes /= batch_ratios[idx]
            draw = batch_draws[idx]

            # Draw detections
            for (bbox, score, label) in zip(bboxes, scores, labels):
                bbox = bbox.round().int().tolist()
                cls_id = int(label)
                cls = CLASSES_DET[cls_id]
                color = COLORS[cls]

                text = f'{cls}:{score:.3f}'
                x1, y1, x2, y2 = bbox

                (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
                y1_text = min(y1 + 1, draw.shape[0])

                cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(draw, (x1, y1_text), (x1 + tw, y1_text + th + bl),
                            (0, 0, 255), -1)
                cv2.putText(draw, text, (x1, y1_text + th), cv2.FONT_HERSHEY_SIMPLEX,
                           0.75, (255, 255, 255), 2)

            if args.show:
                cv2.imshow('result', draw)
                cv2.waitKey(0)
            else:
                cv2.imwrite(str(batch_save_paths[idx]), draw)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file')
    parser.add_argument('--imgs', type=str, help='Images file')
    parser.add_argument('--show',
                        action='store_true',
                        help='Show the detection results')
    parser.add_argument('--out-dir',
                        type=str,
                        default='./output',
                        help='Path to output file')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='TensorRT infer device')
    parser.add_argument('--batch-size',
                        type=int,
                        default=1,
                        help='TensorRT batch size')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    from time import time
    args = parse_args()
    start = time()
    main(args)
    end = time()
    total_images = len(path_to_list(args.imgs))
    total_time = end - start
    

    print(f"Total time: {total_time:.2f} seconds")
    print(f"Images per second: {total_images/total_time:.2f}")
    print(f"Average time per batch: {total_time/(total_images/args.batch_size):.3f} seconds")