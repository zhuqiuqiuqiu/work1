from ultralytics import YOLO
from collections import Counter
import cv2
import matplotlib.pyplot as plt
import serial
import time
import numpy as np

ser = serial.Serial('COM11', 115200, timeout=1)
time.sleep(2)  # 等ESP32启动

def send_to_esp32(focus_rate, active_rate, abnormal_rate):

    try:
        msg = f"F:{focus_rate:.2f},A1:{active_rate:.2f},B2:{abnormal_rate:.2f}\n"
        ser.write(msg.encode())
    except Exception as e:
        print("串口发送失败:", e)

def generate_smart_advice(focus_list, active_list, abnormal_list):
    advice = []
    focus = focus_list[-1]
    active = active_list[-1]
    abnormal = abnormal_list[-1]
    if len(focus_list) >= 5:
        trend = np.mean(focus_list[-5:]) - np.mean(focus_list[-10:-5]) if len(focus_list) >= 10 else 0
    else:
        trend = 0
    low_focus_duration = sum(1 for x in focus_list[-5:] if x < 0.5)
    if focus < 0.5 and low_focus_duration >= 3:
        advice.append("【重点】学生持续注意力较低，建议立即引入互动或提问环节")
    elif trend < -0.1:
        advice.append("【趋势】课堂专注度呈下降趋势，建议调整讲解节奏或增加案例")
    if abnormal > 0.3:
        advice.append("【管理】异常行为较多，建议加强课堂管理或点名提醒")
    if active > 0.4:
        advice.append("【良好】课堂互动较高，可以继续保持当前教学方式")
    if len(focus_list) > 20 and np.mean(focus_list[-5:]) < np.mean(focus_list[:5]):
        advice.append("【节奏】课堂后期注意力下降，建议安排总结或短暂放松")
    if not advice:
        advice.append("课堂状态良好，建议保持当前教学节奏")
    return advice
# =====加载模型 =====
model = YOLO(r"E:\小项目\五一前\Student Behaviour Detection.v6i.yolov8\runs\detect\train\weights\best.pt")
names = model.names
# =====视频 =====
# cap = cv2.VideoCapture("classroom.mp4")
cap = cv2.VideoCapture(1)   # 0 = 默认摄像头
# =====定义类别=====
attention_classes = ["raise_head", "writing", "listen", "upright"]
active_classes = ["raise_hand","writing"]
abnormal_classes = ["sleep", "phone","turn_head","bow_head","bend","book"]
# ===== 时间序列数据 =====
focus_list = []
active_list = []
abnormal_list = []
frame_id = 0
fps = cap.get(cv2.CAP_PROP_FPS)
interval = int(fps)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, verbose=False)

    counter = Counter()

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            counter[cls_id] += 1

    total = sum(counter.values())

    attention = sum(counter[i] for i in counter if names[i] in attention_classes)
    active = sum(counter[i] for i in counter if names[i] in active_classes)
    abnormal = sum(counter[i] for i in counter if names[i] in abnormal_classes)

    focus_rate = attention / total if total > 0 else 0
    active_rate = active / total if total > 0 else 0
    abnormal_rate = abnormal / total if total > 0 else 0

    # ===== 每秒更新 =====
    if frame_id % interval == 0:
        focus_list.append(focus_rate)
        active_list.append(active_rate)
        abnormal_list.append(abnormal_rate)
        advice = generate_smart_advice(focus_list, active_list, abnormal_list)
        print(f"\n===== 第 {len(focus_list)} 秒 =====")
        print(f"专注度: {focus_rate:.2f}")
        print(f"活跃度: {active_rate:.2f}")
        print(f"异常率: {abnormal_rate:.2f}")
        send_to_esp32(focus_rate, active_rate, abnormal_rate)
        print("\n===== 智能教学建议 =====")
        for a in advice:
            print(a)
    frame_id += 1
cap.release()
cv2.destroyAllWindows()