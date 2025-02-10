import argparse
import os
from pathlib import Path
import random
import matplotlib.pyplot as plt
import cv2
import torch
import time
from collections import defaultdict
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
from datetime import datetime
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def calculate_week(start_date, current_date):
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    current_date = datetime.strptime(current_date, "%Y-%m-%d")
    week_number = ((current_date - start_date).days // 7) + 1
    return week_number

def detect(opt, save_img=False):
    source, output, img_size, start_date = opt.source, opt.output, opt.img_size, opt.start_date

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
    class_counters = {name: defaultdict(int) for name in names}

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

    plt.title('Detected Items from YOLOv7 for GH1 Camera @ C00(19-24 )(FGV-Jengka GH1) -first season of cultivation 2022 Oct- Jan 2023')
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

    return class_counters, week_names

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
    parser.add_argument('--start-date', type=str, required=True, help='start date of the season (YYYY-MM-DD)')

    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        class_counters, week_names = detect(opt, save_img=True)

    print("Class counters keys:", class_counters.keys())  # Check the class names

    # Fuzzy Logic System
    # Create fuzzy control system variables
    Health_leaf = ctrl.Antecedent(np.arange(0, 170.01, 1), 'Health_leaf')
    wilt_leaf = ctrl.Antecedent(np.arange(0, 160.01, 1), 'wilt_leaf')
    flower = ctrl.Antecedent(np.arange(0, 150.01, 1), 'flower')
    fruit = ctrl.Antecedent(np.arange(0, 80.01, 1), 'fruit')
    week = ctrl.Antecedent(np.arange(0, 12.01, 1), 'week')
    Rock_melon = ctrl.Consequent(np.arange(0, 150.01, 1), 'Rock_melon')

    # Manually defining membership functions for each variable
    Health_leaf['Low'] = fuzz.trimf(Health_leaf.universe, [-70.83, 0, 60])
    Health_leaf['Medium'] = fuzz.trimf(Health_leaf.universe, [14.17, 85, 155])
    Health_leaf['High'] = fuzz.trimf(Health_leaf.universe, [99.47, 170.3, 241.1])

    wilt_leaf['small'] = fuzz.trimf(wilt_leaf.universe, [-66.39, 0, 40])
    wilt_leaf['middle'] = fuzz.trimf(wilt_leaf.universe, [14, 80, 140])
    wilt_leaf['larger'] = fuzz.trimf(wilt_leaf.universe, [80, 162, 229])

    flower['few'] = fuzz.trimf(flower.universe, [-62.81, 0, 30])
    flower['modelate'] = fuzz.trimf(flower.universe, [15, 75, 130])
    flower['too_much'] = fuzz.trimf(flower.universe, [87.51, 150, 212.4])

    fruit['litle'] = fuzz.trimf(fruit.universe, [-33.33, 0, 20])
    fruit['somehow'] = fuzz.trimf(fruit.universe, [15, 40, 70])
    fruit['many'] = fuzz.trimf(fruit.universe, [50, 80, 113.3])

    week['first_four_weeks'] = fuzz.trimf(week.universe, [-5, 0, 6])
    week['second_four_weeks'] = fuzz.trimf(week.universe, [4, 7, 10])
    week['last_four_weeks'] = fuzz.trimf(week.universe, [8, 12, 17])

    # Output membership functions
    Rock_melon['Normal_condition'] = fuzz.trimf(Rock_melon.universe, [-37.5, 0, 37.5])
    Rock_melon['Check_wilt_leaf'] = fuzz.trimf(Rock_melon.universe, [10, 37.5, 75])
    Rock_melon['check_flower'] = fuzz.trimf(Rock_melon.universe, [37.5, 75, 112.5])
    Rock_melon['check_fruit'] = fuzz.trimf(Rock_melon.universe, [75, 112.5, 150])
    Rock_melon['abnormal_condition'] = fuzz.trimf(Rock_melon.universe, [112.5, 150, 187.5])

    # Define fuzzy rules here
    rule1 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['small'] & flower['few'] & fruit['litle'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule2 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['small'] & flower['few'] & fruit['litle'] & week['second_four_weeks'], Rock_melon['abnormal_condition'])
    rule3 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['small'] & flower['few'] & fruit['litle'] & week['last_four_weeks'], Rock_melon['abnormal_condition'])
    rule4 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['small'] & flower['few'] & fruit['somehow'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule5 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['small'] & flower['few'] & fruit['somehow'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule6 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['small'] & flower['few'] & fruit['somehow'] & week['last_four_weeks'], Rock_melon['Normal_condition'])
    rule7 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['small'] & flower['few'] & fruit['many'] & week['first_four_weeks'], Rock_melon['check_fruit'])
    rule8 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['small'] & flower['few'] & fruit['many'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule9 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['small'] & flower['few'] & fruit['many'] & week['last_four_weeks'], Rock_melon['Normal_condition'])
    rule10 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['small'] & flower['modelate'] & fruit['litle'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule11 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['small'] & flower['modelate'] & fruit['litle'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule12 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['small'] & flower['modelate'] & fruit['litle'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule13 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['small'] & flower['modelate'] & fruit['somehow'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule14 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['small'] & flower['modelate'] & fruit['somehow'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule15 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['small'] & flower['modelate'] & fruit['somehow'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule16 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['small'] & flower['modelate'] & fruit['many'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule17 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['small'] & flower['modelate'] & fruit['many'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule18 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['small'] & flower['modelate'] & fruit['many'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule19 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['small'] & flower['too_much'] & fruit['litle'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule20 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['small'] & flower['too_much'] & fruit['litle'] & week['second_four_weeks'], Rock_melon['check_flower'])
    rule21 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['small'] & flower['too_much'] & fruit['litle'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule22 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['small'] & flower['too_much'] & fruit['somehow'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule23 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['small'] & flower['too_much'] & fruit['somehow'] & week['second_four_weeks'], Rock_melon['check_flower'])
    rule24 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['small'] & flower['too_much'] & fruit['somehow'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule25 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['small'] & flower['too_much'] & fruit['many'] & week['first_four_weeks'], Rock_melon['check_fruit'])
    rule26 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['small'] & flower['too_much'] & fruit['many'] & week['second_four_weeks'], Rock_melon['check_flower'])
    rule27 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['small'] & flower['too_much'] & fruit['many'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule28 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['middle'] & flower['few'] & fruit['litle'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule29 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['middle'] & flower['few'] & fruit['litle'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule30 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['middle'] & flower['few'] & fruit['litle'] & week['last_four_weeks'], Rock_melon['Normal_condition'])
    rule31 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['middle'] & flower['few'] & fruit['somehow'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule32 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['middle'] & flower['few'] & fruit['somehow'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule33 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['middle'] & flower['few'] & fruit['somehow'] & week['last_four_weeks'], Rock_melon['Normal_condition'])
    rule34 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['middle'] & flower['few'] & fruit['many'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule35 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['middle'] & flower['few'] & fruit['many'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule36 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['middle'] & flower['few'] & fruit['many'] & week['last_four_weeks'], Rock_melon['Normal_condition'])
    rule37 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['middle'] & flower['modelate'] & fruit['litle'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule38 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['middle'] & flower['modelate'] & fruit['litle'] & week['second_four_weeks'], Rock_melon['check_flower'])
    rule39 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['middle'] & flower['modelate'] & fruit['litle'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule40 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['middle'] & flower['modelate'] & fruit['somehow'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule41 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['middle'] & flower['modelate'] & fruit['somehow'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule42 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['middle'] & flower['modelate'] & fruit['somehow'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule43 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['middle'] & flower['modelate'] & fruit['many'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule44 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['middle'] & flower['modelate'] & fruit['many'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule45 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['middle'] & flower['modelate'] & fruit['many'] & week['last_four_weeks'], Rock_melon['Normal_condition'])
    rule46 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['middle'] & flower['too_much'] & fruit['litle'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule47 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['middle'] & flower['too_much'] & fruit['litle'] & week['second_four_weeks'], Rock_melon['check_flower'])
    rule48 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['middle'] & flower['too_much'] & fruit['litle'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule49 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['middle'] & flower['too_much'] & fruit['somehow'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule50 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['middle'] & flower['too_much'] & fruit['somehow'] & week['second_four_weeks'], Rock_melon['check_flower'])
    rule51 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['middle'] & flower['too_much'] & fruit['somehow'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule52 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['middle'] & flower['too_much'] & fruit['many'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule53 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['middle'] & flower['too_much'] & fruit['many'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule54 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['middle'] & flower['too_much'] & fruit['many'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule55 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['larger'] & flower['few'] & fruit['litle'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule56 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['larger'] & flower['few'] & fruit['litle'] & week['second_four_weeks'], Rock_melon['check_fruit'])
    rule57 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['larger'] & flower['few'] & fruit['litle'] & week['last_four_weeks'], Rock_melon['check_fruit'])
    rule58 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['larger'] & flower['few'] & fruit['somehow'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule59 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['larger'] & flower['few'] & fruit['somehow'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule60 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['larger'] & flower['few'] & fruit['somehow'] & week['last_four_weeks'], Rock_melon['Normal_condition'])
    rule61 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['larger'] & flower['few'] & fruit['many'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule62 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['larger'] & flower['few'] & fruit['many'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule63 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['larger'] & flower['few'] & fruit['many'] & week['last_four_weeks'], Rock_melon['Normal_condition'])
    rule64 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['larger'] & flower['modelate'] & fruit['litle'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule65 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['larger'] & flower['modelate'] & fruit['litle'] & week['second_four_weeks'], Rock_melon['check_flower'])
    rule66 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['larger'] & flower['modelate'] & fruit['litle'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule67 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['larger'] & flower['modelate'] & fruit['somehow'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule68 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['larger'] & flower['modelate'] & fruit['somehow'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule69 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['larger'] & flower['modelate'] & fruit['somehow'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule70 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['larger'] & flower['modelate'] & fruit['many'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule71 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['larger'] & flower['modelate'] & fruit['many'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule72 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['larger'] & flower['modelate'] & fruit['many'] & week['last_four_weeks'], Rock_melon['Normal_condition'])
    rule73 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['larger'] & flower['too_much'] & fruit['litle'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule74 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['larger'] & flower['too_much'] & fruit['litle'] & week['second_four_weeks'], Rock_melon['check_flower'])
    rule75 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['larger'] & flower['too_much'] & fruit['litle'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule76 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['larger'] & flower['too_much'] & fruit['somehow'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule77 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['larger'] & flower['too_much'] & fruit['somehow'] & week['second_four_weeks'], Rock_melon['check_flower'])
    rule78 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['larger'] & flower['too_much'] & fruit['somehow'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule79 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['larger'] & flower['too_much'] & fruit['many'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule80 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['larger'] & flower['too_much'] & fruit['many'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule81 = ctrl.Rule(Health_leaf['Low'] & wilt_leaf['larger'] & flower['too_much'] & fruit['many'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule82 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['small'] & flower['few'] & fruit['litle'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule83 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['small'] & flower['few'] & fruit['litle'] & week['second_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule84 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['small'] & flower['few'] & fruit['litle'] & week['last_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule85 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['small'] & flower['few'] & fruit['somehow'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule86 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['small'] & flower['few'] & fruit['somehow'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule87 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['small'] & flower['few'] & fruit['somehow'] & week['last_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule88 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['small'] & flower['few'] & fruit['many'] & week['first_four_weeks'], Rock_melon['check_fruit'])
    rule89 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['small'] & flower['few'] & fruit['many'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule90 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['small'] & flower['few'] & fruit['many'] & week['last_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule91 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['small'] & flower['modelate'] & fruit['litle'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule92 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['small'] & flower['modelate'] & fruit['litle'] & week['second_four_weeks'], Rock_melon['check_flower'])
    rule93 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['small'] & flower['modelate'] & fruit['litle'] & week['last_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule94 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['small'] & flower['modelate'] & fruit['somehow'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule95 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['small'] & flower['modelate'] & fruit['somehow'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule96 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['small'] & flower['modelate'] & fruit['somehow'] & week['last_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule97 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['small'] & flower['modelate'] & fruit['many'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule98 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['small'] & flower['modelate'] & fruit['many'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule99 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['small'] & flower['modelate'] & fruit['many'] & week['last_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule100 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['small'] & flower['too_much'] & fruit['litle'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule101 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['small'] & flower['too_much'] & fruit['litle'] & week['second_four_weeks'], Rock_melon['check_fruit'])
    rule102 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['small'] & flower['too_much'] & fruit['litle'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule103 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['small'] & flower['too_much'] & fruit['somehow'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule104 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['small'] & flower['too_much'] & fruit['somehow'] & week['second_four_weeks'], Rock_melon['check_flower'])
    rule105 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['small'] & flower['too_much'] & fruit['somehow'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule106 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['small'] & flower['too_much'] & fruit['many'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule107 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['small'] & flower['too_much'] & fruit['many'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule108 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['small'] & flower['too_much'] & fruit['many'] & week['last_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule109 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['middle'] & flower['few'] & fruit['litle'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule110 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['middle'] & flower['few'] & fruit['litle'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule111 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['middle'] & flower['few'] & fruit['litle'] & week['last_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule112 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['middle'] & flower['few'] & fruit['somehow'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule113 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['middle'] & flower['few'] & fruit['somehow'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule114 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['middle'] & flower['few'] & fruit['somehow'] & week['last_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule115 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['middle'] & flower['few'] & fruit['many'] & week['first_four_weeks'], Rock_melon['check_fruit'])
    rule116 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['middle'] & flower['few'] & fruit['many'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule117 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['middle'] & flower['few'] & fruit['many'] & week['last_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule118 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['middle'] & flower['modelate'] & fruit['litle'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule119 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['middle'] & flower['modelate'] & fruit['litle'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule120 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['middle'] & flower['modelate'] & fruit['litle'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule121 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['middle'] & flower['modelate'] & fruit['somehow'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule122 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['middle'] & flower['modelate'] & fruit['somehow'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule123 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['middle'] & flower['modelate'] & fruit['somehow'] & week['last_four_weeks'], Rock_melon['abnormal_condition'])
    rule124 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['middle'] & flower['modelate'] & fruit['many'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule125 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['middle'] & flower['modelate'] & fruit['many'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule126 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['middle'] & flower['modelate'] & fruit['many'] & week['last_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule127 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['middle'] & flower['too_much'] & fruit['litle'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule128 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['middle'] & flower['too_much'] & fruit['litle'] & week['second_four_weeks'], Rock_melon['check_flower'])
    rule129 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['middle'] & flower['too_much'] & fruit['litle'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule130 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['middle'] & flower['too_much'] & fruit['somehow'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule131 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['middle'] & flower['too_much'] & fruit['somehow'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule132 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['middle'] & flower['too_much'] & fruit['somehow'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule133 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['middle'] & flower['too_much'] & fruit['many'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule134 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['middle'] & flower['too_much'] & fruit['many'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule135 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['middle'] & flower['too_much'] & fruit['many'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule136 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['larger'] & flower['few'] & fruit['litle'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule137 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['larger'] & flower['few'] & fruit['litle'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule138 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['larger'] & flower['few'] & fruit['litle'] & week['last_four_weeks'], Rock_melon['check_fruit'])
    rule139 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['larger'] & flower['few'] & fruit['somehow'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule140 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['larger'] & flower['few'] & fruit['somehow'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule141 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['larger'] & flower['few'] & fruit['somehow'] & week['last_four_weeks'], Rock_melon['Normal_condition'])
    rule142 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['larger'] & flower['few'] & fruit['many'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule143 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['larger'] & flower['few'] & fruit['many'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule144 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['larger'] & flower['few'] & fruit['many'] & week['last_four_weeks'], Rock_melon['Normal_condition'])
    rule145 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['larger'] & flower['modelate'] & fruit['litle'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule146 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['larger'] & flower['modelate'] & fruit['litle'] & week['second_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule147 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['larger'] & flower['modelate'] & fruit['litle'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule148 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['larger'] & flower['modelate'] & fruit['somehow'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule149 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['larger'] & flower['modelate'] & fruit['somehow'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule150 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['larger'] & flower['modelate'] & fruit['somehow'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule151 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['larger'] & flower['modelate'] & fruit['many'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule152 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['larger'] & flower['modelate'] & fruit['many'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule153 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['larger'] & flower['modelate'] & fruit['many'] & week['last_four_weeks'], Rock_melon['Normal_condition'])
    rule154 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['larger'] & flower['too_much'] & fruit['litle'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule155 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['larger'] & flower['too_much'] & fruit['litle'] & week['second_four_weeks'], Rock_melon['check_flower'])
    rule156 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['larger'] & flower['too_much'] & fruit['litle'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule157 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['larger'] & flower['too_much'] & fruit['somehow'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule158 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['larger'] & flower['too_much'] & fruit['somehow'] & week['second_four_weeks'], Rock_melon['check_flower'])
    rule159 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['larger'] & flower['too_much'] & fruit['somehow'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule160 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['larger'] & flower['too_much'] & fruit['many'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule161 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['larger'] & flower['too_much'] & fruit['many'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule162 = ctrl.Rule(Health_leaf['Medium'] & wilt_leaf['larger'] & flower['too_much'] & fruit['many'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule163 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['small'] & flower['few'] & fruit['litle'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule164 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['small'] & flower['few'] & fruit['litle'] & week['second_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule165 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['small'] & flower['few'] & fruit['litle'] & week['last_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule166 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['small'] & flower['few'] & fruit['somehow'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule167 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['small'] & flower['few'] & fruit['somehow'] & week['second_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule168 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['small'] & flower['few'] & fruit['somehow'] & week['last_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule169 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['small'] & flower['few'] & fruit['many'] & week['first_four_weeks'], Rock_melon['check_fruit'])
    rule170 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['small'] & flower['few'] & fruit['many'] & week['second_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule171 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['small'] & flower['few'] & fruit['many'] & week['last_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule172 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['small'] & flower['modelate'] & fruit['litle'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule173 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['small'] & flower['modelate'] & fruit['litle'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule174 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['small'] & flower['modelate'] & fruit['litle'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule175 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['small'] & flower['modelate'] & fruit['somehow'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule176 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['small'] & flower['modelate'] & fruit['somehow'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule177 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['small'] & flower['modelate'] & fruit['somehow'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule178 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['small'] & flower['modelate'] & fruit['many'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule179 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['small'] & flower['modelate'] & fruit['many'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule180 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['small'] & flower['modelate'] & fruit['many'] & week['last_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule181 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['small'] & flower['too_much'] & fruit['litle'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule182 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['small'] & flower['too_much'] & fruit['litle'] & week['second_four_weeks'], Rock_melon['check_flower'])
    rule183 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['small'] & flower['too_much'] & fruit['litle'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule184 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['small'] & flower['too_much'] & fruit['somehow'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule185 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['small'] & flower['too_much'] & fruit['somehow'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule186 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['small'] & flower['too_much'] & fruit['somehow'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule187 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['small'] & flower['too_much'] & fruit['many'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule188 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['small'] & flower['too_much'] & fruit['many'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule189 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['small'] & flower['too_much'] & fruit['many'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule190 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['middle'] & flower['few'] & fruit['litle'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule191 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['middle'] & flower['few'] & fruit['litle'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule192 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['middle'] & flower['few'] & fruit['litle'] & week['last_four_weeks'], Rock_melon['abnormal_condition'])
    rule193 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['middle'] & flower['few'] & fruit['somehow'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule194 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['middle'] & flower['few'] & fruit['somehow'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule195 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['middle'] & flower['few'] & fruit['somehow'] & week['last_four_weeks'], Rock_melon['abnormal_condition'])
    rule196 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['middle'] & flower['few'] & fruit['many'] & week['first_four_weeks'], Rock_melon['check_fruit'])
    rule197 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['middle'] & flower['few'] & fruit['many'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule198 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['middle'] & flower['few'] & fruit['many'] & week['last_four_weeks'], Rock_melon['Normal_condition'])
    rule199 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['middle'] & flower['modelate'] & fruit['litle'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule200 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['middle'] & flower['modelate'] & fruit['litle'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule201 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['middle'] & flower['modelate'] & fruit['litle'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule202 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['middle'] & flower['modelate'] & fruit['somehow'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule203 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['middle'] & flower['modelate'] & fruit['somehow'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule204 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['middle'] & flower['modelate'] & fruit['somehow'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule205 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['middle'] & flower['modelate'] & fruit['many'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule206 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['middle'] & flower['modelate'] & fruit['many'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule207 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['middle'] & flower['modelate'] & fruit['many'] & week['last_four_weeks'], Rock_melon['Normal_condition'])
    rule208 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['middle'] & flower['too_much'] & fruit['litle'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule209 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['middle'] & flower['too_much'] & fruit['litle'] & week['second_four_weeks'], Rock_melon['check_flower'])
    rule210 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['middle'] & flower['too_much'] & fruit['litle'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule211 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['middle'] & flower['too_much'] & fruit['somehow'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule212 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['middle'] & flower['too_much'] & fruit['somehow'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule213 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['middle'] & flower['too_much'] & fruit['somehow'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule214 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['middle'] & flower['too_much'] & fruit['many'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule215 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['middle'] & flower['too_much'] & fruit['many'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule216 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['middle'] & flower['too_much'] & fruit['many'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule217 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['larger'] & flower['few'] & fruit['litle'] & week['first_four_weeks'], Rock_melon['Normal_condition'])
    rule218 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['larger'] & flower['few'] & fruit['litle'] & week['second_four_weeks'], Rock_melon['abnormal_condition'])
    rule219 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['larger'] & flower['few'] & fruit['somehow'] & week['first_four_weeks'], Rock_melon['abnormal_condition'])
    rule220 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['larger'] & flower['few'] & fruit['litle'] & week['last_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule221 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['larger'] & flower['few'] & fruit['somehow'] & week['second_four_weeks'], Rock_melon['abnormal_condition'])
    rule222 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['larger'] & flower['few'] & fruit['somehow'] & week['last_four_weeks'], Rock_melon['abnormal_condition'])
    rule223 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['larger'] & flower['few'] & fruit['many'] & week['last_four_weeks'], Rock_melon['abnormal_condition'])
    rule224 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['larger'] & flower['few'] & fruit['many'] & week['first_four_weeks'], Rock_melon['abnormal_condition'])
    rule225 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['larger'] & flower['few'] & fruit['many'] & week['second_four_weeks'], Rock_melon['abnormal_condition'])
    rule226 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['larger'] & flower['modelate'] & fruit['litle'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule227 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['larger'] & flower['modelate'] & fruit['litle'] & week['second_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule228 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['larger'] & flower['modelate'] & fruit['litle'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule229 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['larger'] & flower['modelate'] & fruit['somehow'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule230 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['larger'] & flower['modelate'] & fruit['somehow'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule231 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['larger'] & flower['modelate'] & fruit['somehow'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule232 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['larger'] & flower['modelate'] & fruit['many'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule233 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['larger'] & flower['modelate'] & fruit['many'] & week['second_four_weeks'], Rock_melon['Normal_condition'])
    rule234 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['larger'] & flower['modelate'] & fruit['many'] & week['last_four_weeks'], Rock_melon['abnormal_condition'])
    rule235 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['larger'] & flower['too_much'] & fruit['litle'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule236 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['larger'] & flower['too_much'] & fruit['litle'] & week['second_four_weeks'], Rock_melon['check_flower'])
    rule237 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['larger'] & flower['too_much'] & fruit['litle'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule238 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['larger'] & flower['too_much'] & fruit['somehow'] & week['first_four_weeks'], Rock_melon['Check_wilt_leaf'])
    rule239 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['larger'] & flower['too_much'] & fruit['somehow'] & week['second_four_weeks'], Rock_melon['check_flower'])
    rule240 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['larger'] & flower['too_much'] & fruit['somehow'] & week['last_four_weeks'], Rock_melon['check_flower'])
    rule241 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['larger'] & flower['too_much'] & fruit['many'] & week['first_four_weeks'], Rock_melon['abnormal_condition'])
    rule242 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['larger'] & flower['too_much'] & fruit['many'] & week['second_four_weeks'], Rock_melon['abnormal_condition'])
    rule243 = ctrl.Rule(Health_leaf['High'] & wilt_leaf['larger'] & flower['too_much'] & fruit['many'] & week['last_four_weeks'], Rock_melon['abnormal_condition'])

    # Control System Creation and Simulation
    rockmelon_ctrl = ctrl.ControlSystem([
        rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10,
        rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20,
        rule21, rule22, rule23, rule24, rule25, rule26, rule27, rule28, rule29, rule30,
        rule31, rule32, rule33, rule34, rule35, rule36, rule37, rule38, rule39, rule40,
        rule41, rule42, rule43, rule44, rule45, rule46, rule47, rule48, rule49, rule50,
        rule51, rule52, rule53, rule54, rule55, rule56, rule57, rule58, rule59, rule60,
        rule61, rule62, rule63, rule64, rule65, rule66, rule67, rule68, rule69, rule70,
        rule71, rule72, rule73, rule74, rule75, rule76, rule77, rule78, rule79, rule80,
        rule81, rule82, rule83, rule84, rule85, rule86, rule87, rule88, rule89, rule90,
        rule91, rule92, rule93, rule94, rule95, rule96, rule97, rule98, rule99, rule100,
        rule101, rule102, rule103, rule104, rule105, rule106, rule107, rule108, rule109, rule110,
        rule111, rule112, rule113, rule114, rule115, rule116, rule117, rule118, rule119, rule120,
        rule121, rule122, rule123, rule124, rule125, rule126, rule127, rule128, rule129, rule130,
        rule131, rule132, rule133, rule134, rule135, rule136, rule137, rule138, rule139, rule140,
        rule141, rule142, rule143, rule144, rule145, rule146, rule147, rule148, rule149, rule150,
        rule151, rule152, rule153, rule154, rule155, rule156, rule157, rule158, rule159, rule160,
        rule161, rule162, rule163, rule164, rule165, rule166, rule167, rule168, rule169, rule170,
        rule171, rule172, rule173, rule174, rule175, rule176, rule177, rule178, rule179, rule180,
        rule181, rule182, rule183, rule184, rule185, rule186, rule187, rule188, rule189, rule190,
        rule191, rule192, rule193, rule194, rule195, rule196, rule197, rule198, rule199, rule200,
        rule201, rule202, rule203, rule204, rule205, rule206, rule207, rule208, rule209, rule210,
        rule211, rule212, rule213, rule214, rule215, rule216, rule217, rule218, rule219, rule220,
        rule221, rule222, rule223, rule224, rule225, rule226, rule227, rule228, rule229, rule230,
        rule231, rule232, rule233, rule234, rule235, rule236, rule237, rule238, rule239, rule240,
        rule241, rule242, rule243
    ])

    rockmelon_simulator = ctrl.ControlSystemSimulation(rockmelon_ctrl)

    # Example date and start date
    current_date = datetime.now().strftime("%Y-%m-%d")  # Get the current date dynamically
    start_date = opt.start_date

    # Calculate the week based on the current date and start date
    week_number = calculate_week(start_date, current_date)

print("Class counters:", class_counters)

# Get detection counts for the current week
try:
    attributes = {
        'health_leaf': class_counters['healthy leaf'][f"week{week_number}"],
        'wilt_leaf': class_counters['wilt leaf'][f"week{week_number}"],
        'flower': class_counters['flower'][f"week{week_number}"],
        'fruit': class_counters['fruit'][f"week{week_number}"],
        'week': week_number
    }
except KeyError as e:
    print(f"Error: {e}. Please check the class names in class_counters.")
    print("Class counters keys:", class_counters.keys())
    raise

# Print attributes to ensure they are correct
print(f"Attributes: {attributes}")

# Feed attributes to the fuzzy logic system
rockmelon_simulator.input['Health_leaf'] = attributes['health_leaf']
rockmelon_simulator.input['wilt_leaf'] = attributes['wilt_leaf']
rockmelon_simulator.input['flower'] = attributes['flower']
rockmelon_simulator.input['fruit'] = attributes['fruit']
rockmelon_simulator.input['week'] = attributes['week']

# Run the fuzzy logic simulation
rockmelon_simulator.compute()

# Output the result
output_value = rockmelon_simulator.output['Rock_melon']
print("Output value:", output_value)

# Calculate membership values for each output condition
normal_cond = fuzz.interp_membership(Rock_melon.universe, Rock_melon['Normal_condition'].mf, output_value)
check_wilt_leaf = fuzz.interp_membership(Rock_melon.universe, Rock_melon['Check_wilt_leaf'].mf, output_value)
check_flower = fuzz.interp_membership(Rock_melon.universe, Rock_melon['check_flower'].mf, output_value)
check_fruit = fuzz.interp_membership(Rock_melon.universe, Rock_melon['check_fruit'].mf, output_value)
abnormal_cond = fuzz.interp_membership(Rock_melon.universe, Rock_melon['abnormal_condition'].mf, output_value)

# Dictionary to hold the membership values
output_conditions = {
    'Normal Condition': normal_cond,
    'Check Wilt Leaf': check_wilt_leaf,
    'Check Flower': check_flower,
    'Check Fruit': check_fruit,
    'Abnormal Condition': abnormal_cond
}

# Determine the condition with the highest membership value
max_condition = max(output_conditions, key=output_conditions.get)
max_value = output_conditions[max_condition]

# Print the most applicable condition and its confidence
print(f"The condition of the rock melon when health leaf is {attributes['health_leaf']}, wilt leaf is {attributes['wilt_leaf']}, "
      f"flower is {attributes['flower']}, fruit is {attributes['fruit']}, and week is {attributes['week']}, "
      f"is most likely '{max_condition}' with a confidence of {max_value:.2f}.")
 
