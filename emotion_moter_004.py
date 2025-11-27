import cv2
from picamera2 import Picamera2
import time
from fer import FER
import csv
import RPi.GPIO as GPIO
from datetime import datetime
import threading

# GPIO 引脚设置
IN1 = 19
IN2 = 13
IN3 = 6
IN4 = 5
motor_pins = [IN1, IN2, IN3, IN4]

# 步进序列（8步模式）
step_sequence = [
    [1, 0, 0, 1],
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1]
]

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for pin in motor_pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, False)

# 步进电机线程控制变量
motor_running = False
motor_thread = None
step_index = 0

# 步进线程：只要 motor_running 为 True 就不停步进
def motor_loop():
    global step_index
    while True:
        if motor_running:
            for i in range(4):
                GPIO.output(motor_pins[i], step_sequence[step_index][i])
            step_index = (step_index + 1) % len(step_sequence)
            time.sleep(0.001)  # 1ms 控制速度（更小更快）
        else:
            for pin in motor_pins:
                GPIO.output(pin, 0)
            time.sleep(0.01)  # 待机时稍微睡久一点，节省资源

# 启动后台线程（守护线程）
motor_thread = threading.Thread(target=motor_loop, daemon=True)
motor_thread.start()

# 初始化摄像头
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    raw={"size": (640, 480)},
    main={"format": 'RGB888', "size": (640, 480)}
))
picam2.start()
time.sleep(2)

# 初始化变量
prev_time = time.time()
emotion_detector = FER()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

emotion_count = {
    "angry": 0, "disgust": 0, "fear": 0,
    "happy": 0, "sad": 0, "surprise": 0, "neutral": 0
}

while True:
    img = picam2.capture_array()
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    happy_detected = False

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_img = img[y:y + h, x:x + w]
        emotion, score = emotion_detector.top_emotion(face_img)

        if emotion is not None and score is not None:
            if emotion in emotion_count:
                emotion_count[emotion] += 1

            cv2.putText(img, f"{emotion}: {score:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if emotion == "happy" and score > 0.7:
                happy_detected = True

    # 控制电机开启与关闭
    if happy_detected:
        motor_running = True
    else:
        motor_running = False

    # 显示图像和 FPS
    cv2.putText(img, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Output", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 输出统计信息
print("Emotion recognition counts:")
for emotion, count in emotion_count.items():
    print(f"{emotion}: {count}")

# 保存到 CSV
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename = f"emotion_stats_{timestamp}.csv"

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Emotion', 'Count'])
    for emotion, count in emotion_count.items():
        writer.writerow([emotion, count])

# 清理资源
picam2.stop()
picam2.close()
motor_running = False
time.sleep(0.5)
GPIO.cleanup()

