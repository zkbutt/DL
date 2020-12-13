
import cv2


def init_video():
    cap = cv2.VideoCapture(0)  # capture=cv2.VideoCapture("1.mp4")
    cap.set(cv2.CAP_PROP_FPS, 20)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


