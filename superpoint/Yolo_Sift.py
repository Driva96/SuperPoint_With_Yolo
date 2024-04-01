import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO
import torch

# Initialize YOLO model
model = YOLO('yolov8n.pt')


# Initialize SIFT detector
def extract_SIFT_keypoints_and_descriptors(img):
    sift = cv2.SIFT_create()
    kp, desc = sift.detectAndCompute(img, None)
    return kp, desc


# Function to create a Kalman Filter for both bounding boxes and keypoints
def create_kalman_filter(dim_x=8, dim_z=4):
    kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
    kf.F = np.array([[1, 0, 1, 0, 0, 0, 0, 0],  # state transition matrix for bbox and keypoints
                     [0, 1, 0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 1, 0],  # state transition matrix for keypoints
                     [0, 0, 0, 0, 0, 1, 0, 1],
                     [0, 0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],  # measurement function for bbox
                     [0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0, 0],  # measurement function for keypoints
                     [0, 0, 0, 0, 0, 1, 0, 0]])
    kf.R *= 10  # Measurement noise
    kf.P *= 10  # Initial estimate uncertainty
    kf.Q *= 0.01  # Process noise
    return kf

# Function to update Kalman Filter with detections (bounding box + keypoints)
def update_kalman_filter(kf, detection):
    kf.update(np.array(detection))
    return kf.x

# Function to find keypoints within a bounding box
def find_keypoints_in_bbox(kp, bbox):
    keypoints = []
    x1, y1, x2, y2 = bbox
    for point in kp:
        if x1 <= point.pt[0] <= x2 and y1 <= point.pt[1] <= y2:
            keypoints.append(point)
    return keypoints

# Function to prepare Kalman data (bounding box + keypoints)
def prepare_kalman_data(bbox, keypoints):
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2
    if keypoints:
        kp_x, kp_y = keypoints[0].pt
    else:
        kp_x, kp_y = x_center, y_center  # Fallback to bbox center if no keypoints
    return [x_center, y_center, 0, 0, kp_x, kp_y, 0, 0]  # Assuming velocities are 0 for simplicity

# Main function to process the video
def main(video_path):
    cap = cv2.VideoCapture(video_path)
    kalman_filters_obj = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, _ = extract_SIFT_keypoints_and_descriptors(frame_gray)
        results = model(frame)

        # Check if results is a list and handle accordingly
        if isinstance(results, list) and len(results) > 0:
            result = results[0]  # Assuming the first result contains the detection data
            if hasattr(result, 'boxes'):
                boxes = result.boxes
                for i, box in enumerate(boxes.xyxy[0]):
                    bbox = box.cpu().numpy().astype(int)
                    object_kp = find_keypoints_in_bbox(kp, bbox[:4])

                    if i not in kalman_filters_obj:
                        kalman_filters_obj[i] = create_kalman_filter()

                    detection_data = prepare_kalman_data(bbox, object_kp)
                    kf_state = update_kalman_filter(kalman_filters_obj[i], detection_data[:4] + detection_data[4:6])

                    # Drawing bounding box and keypoints
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                    for point in object_kp:
                        cv2.circle(frame, (int(point.pt[0]), int(point.pt[1])), 3, (0, 255, 0), -1)

        cv2.imshow('Combined Detection and Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #video_path = "/Users/loganlambeth/PycharmProjects/JetsonCode2023/RedDetection.mp4"
    video_path = "/Users/loganlambeth/Documents/GitHub/SuperPoint_With_Yolo/F-16Video.mp4"

    main(video_path)
