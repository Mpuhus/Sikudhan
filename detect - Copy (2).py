import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import torch
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import (check_img_size, non_max_suppression, scale_coords,
                           xyxy2xywh, set_logging, increment_path)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

def detect(opt, save_img=False):
    out, weights, view_img, save_txt, imgsz = \
        opt.output, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    # Initialize
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names

    if half:
        model.half()  # to FP16

    # Split the source into separate paths for each week
    source_paths = opt.source.split(';')
    week_names = [Path(p).stem for p in source_paths]  # Extract week names

    # Initialize counters for each class for each week
    class_counters = {week: {name: 0 for name in names} for week in week_names}

    # Run inference and process detections for each week
    for week, week_path in zip(week_names, source_paths):
        if not os.path.exists(week_path):
            raise Exception(f'ERROR: {week_path} does not exist')

        dataset = LoadImages(week_path, img_size=imgsz, stride=stride)

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

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
                        class_counters[week][class_name] += 1

    # After all detections have been processed, plot the results
    plt.figure(figsize=(15, 8))
    for class_name in names:
        weekly_counts = [class_counters[week][class_name] for week in week_names]
        plt.plot(week_names, weekly_counts, marker='o', label=class_name)

    plt.title('Detected Items from YOLOv7 Output')
    plt.xlabel('Weeks')
    plt.ylabel('Number of Detected Items')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.savefig('detections_over_time.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(opt)
