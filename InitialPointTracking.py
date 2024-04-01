import numpy as np
import cv2


def initialize_points(gray_frame):
    # Use goodFeaturesToTrack for initialization
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    p0 = cv2.goodFeaturesToTrack(gray_frame, mask=None, **feature_params)
    return p0


def track_points(prev_gray, gray_frame, p0):
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray_frame, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    return good_new, good_old


def main(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    p0 = initialize_points(prev_gray)

    while (True):
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Track points
        good_new, good_old = track_points(prev_gray, gray_frame, p0)

        # Draw tracked points
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            frame = cv2.circle(frame, (a, b), 5, (0, 255, 0), -1)

        cv2.imshow('Frame', frame)

        # Update previous frame and points
        prev_gray = gray_frame.copy()
        p0 = good_new.reshape(-1, 1, 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "F-16Video.mp4"
    main(video_path)
