import cv2
import numpy as np
from filterpy.kalman import KalmanFilter

def extract_SIFT_keypoints_and_descriptors(img):
    sift = cv2.SIFT_create()
    kp, desc = sift.detectAndCompute(img, None)
    return kp, desc


def match_keypoints(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    return matches


def create_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.P *= 10
    return kf

def track_keypoints(keypoints, kalman_filters):
    predicted_keypoints = []
    for kp, kf in zip(keypoints, kalman_filters):
        kf.predict()
        if kp is not None:  # Check if a keypoint was detected
            kf.update(kp.pt)
            predicted = kf.x[:2]
            predicted_keypoints.append(predicted)
        else:
            predicted_keypoints.append(None)  # Mark keypoints as None if not detected
    return predicted_keypoints

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_frame = cv2.resize(prev_frame, (640, 480))
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        prev_kp, prev_desc = extract_SIFT_keypoints_and_descriptors(prev_frame_gray)
        kp, desc = extract_SIFT_keypoints_and_descriptors(frame_gray)

        frame_with_keypoints = cv2.drawKeypoints(frame, prev_kp, outImage=None, color=(0, 255, 0), flags=0)

        matches = match_keypoints(prev_desc, desc)
        matched_frame = cv2.drawMatches(prev_frame, prev_kp, frame, kp, matches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Create Kalman filters for keypoints
        num_keypoints = len(prev_kp)
        kalman_filters = [create_kalman_filter() for _ in range(num_keypoints)]

        # Track keypoints using Kalman filter
        predicted_keypoints = track_keypoints(prev_kp, kalman_filters)

        kalmanFrame = frame.copy()
        for kp in predicted_keypoints:
            if kp is not None:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(kalmanFrame, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

        cv2.imshow('Frame', matched_frame)
        cv2.imshow('Frame with SuperPoint Keypoints', frame_with_keypoints)
        cv2.imshow('Kalman Keypoints', kalmanFrame)

        out.write(matched_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_frame_gray = frame_gray
        prev_frame = frame

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "/Users/loganlambeth/Documents/GitHub/SuperPoint_With_Yolo/F-16Video.mp4"
    main(video_path)
