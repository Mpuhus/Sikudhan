import argparse
import os
from pathlib import Path
import random
import matplotlib
matplotlib.use('Qt5Agg')  # Use this backend
import matplotlib.pyplot as plt
import cv2
import torch
import time
from collections import defaultdict
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import (check_img_size, non_max_suppression, scale_coords,
                           xyxy2xywh, increment_path)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

def detect(opt, save_img=False):
    source, output, img_size = opt.source, opt.output, opt.img_size

    # Initialize
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    img_size = check_img_size(img_size, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = {name: [random.randint(0, 255) for _ in range(3)] for name in names}

    if half:
        model.half()  # to FP16

    # Split the source into separate paths for each week
    source_paths = source.split(';')
    week_names = [Path(p).stem for p in source_paths]  # Extract week names

    # Initialize counters for each class for each week
    class_counters = {name: {week: 0 for week in week_names} for name in names}

    # Run inference and process detections for each week
    for week, week_path in zip(week_names, source_paths):
        if not os.path.exists(week_path):
            raise Exception(f'ERROR: {week_path} does not exist')

        try:
            dataset = LoadImages(week_path, img_size=img_size, stride=stride)

            for path, img, im0s, vid_cap in dataset:
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Warmup
                if device.type != 'cpu':
                    for _ in range(3):
                        model(img, augment=opt.augment)[0]

                # Inference
                t1 = time_synchronized()
                pred = model(img, augment=opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                           classes=opt.classes, agnostic=opt.agnostic_nms)
                t2 = time_synchronized()

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            class_name = names[int(cls)]
                            label = f'{class_name} {conf:.2f}'  # Including confidence score
                            color = colors[class_name]
                            plot_one_box(xyxy, im0s, label=label, color=color, line_thickness=3)
                            class_counters[class_name][week] += 1

                # Save the processed image
                save_path = os.path.join(output, f'{week}_{Path(path).stem}.jpg')
                if save_img:
                    cv2.imwrite(save_path, im0s)
                    detection_summary = ', '.join(f'{class_counters[name][week]} {name}' for name in names)
                    print(f"{detection_summary}, Done. ({t2 - t1:.1f}ms) Inference, ({time_synchronized() - t2:.1f}ms) NMS")
                    print(f"The image with the result is saved in : {save_path}")

        except AssertionError as e:
            print(f"No images found for week {week}: {e}")

    # Plotting results
    plt.figure(figsize=(15, 8))
    for name in names:
        weekly_counts = [class_counters[name][week] for week in week_names]
        plt.plot(week_names, weekly_counts, marker='o', label=name)

    plt.title('Detected Items from YOLOv7 for GH1 Camera @ C00(19-24 )(FGV-Jengka GH1) -first season of cultivaion 2022 Oct- Jan 2023')
    plt.xlabel('Weeks')
    plt.ylabel('Number of Detected Items')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('detections_over_time.png')
    plt.show()

    # Print weekly totals in detailed format
    print("Weekly totals:")
    for week in week_names:
        totals = ', '.join(f"{class_counters[name][week]} {name}" for name in names if class_counters[name][week] > 0)
        print(f"{week}: {totals}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(opt, save_img=True)
