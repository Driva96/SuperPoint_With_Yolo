import cv2
import numpy as np
from filterpy.kalman import KalmanFilter

class ModifiedPointTracker:
    def __init__(self, max_length=5, nn_thresh=0.7, max_inactive_frames=5):
        self.max_length = max_length
        self.nn_thresh = nn_thresh
        self.tracks = {}  # Each track_id maps to a dict with keys 'kps', 'active', and 'last_updated'
        self.kalman_filters = {}
        self.track_count = 0
        self.max_inactive_frames = max_inactive_frames


    def update(self, matched_kp, current_frame):
        # Update existing tracks, mark them as inactive
        for track_id in self.tracks:
            self.tracks[track_id]['active'] = False

        # Go through all matched keypoints and update/add tracks
        for idx, kp in enumerate(matched_kp):
            if kp is not None:
                track_id = self.match_keypoint_to_track(kp, distance_threshold=30)
                if track_id is not None:
                    # Update the existing track
                    self.tracks[track_id]['kps'].append(kp)
                    self.tracks[track_id]['active'] = True
                    self.tracks[track_id]['last_updated'] = current_frame
                    # Ensure the track doesn't exceed the maximum length
                    if len(self.tracks[track_id]['kps']) > self.max_length:
                        self.tracks[track_id]['kps'].pop(0)
                    # Update Kalman filter
                    kf = self.kalman_filters[track_id]
                    kf.update(np.array([kp.pt[0], kp.pt[1]]).reshape(2, 1))
                else:
                    # Create a new track
                    self.track_count += 1
                    self.tracks[self.track_count] = {
                        'kps': [kp],
                        'active': True,
                        'last_updated': current_frame
                    }
                    kf = KalmanFilter(dim_x=4, dim_z=2)
                    # Initialize Kalman filter here...
                    self.kalman_filters[self.track_count] = kf

        # Clean up old tracks
        inactive_tracks = [track_id for track_id, track in self.tracks.items()
                           if not track['active'] and
                           (current_frame - track['last_updated']) > self.max_inactive_frames]
        for track_id in inactive_tracks:
            del self.tracks[track_id]
            del self.kalman_filters[track_id]


    def predict(self):
        for kf in self.kalman_filters.values():
            kf.predict()

    def correct(self, matched_kp):
        for track_id, kf in self.kalman_filters.items():
            if track_id in matched_kp:
                measurement = np.array([[matched_kp[track_id].pt[0]], [matched_kp[track_id].pt[1]]])
                kf.update(measurement)

    def match_keypoint_to_track(self, kp, distance_threshold=30):
        min_distance = float('inf')
        matched_track_id = None
        for track_id, track_info in self.tracks.items():
            if not track_info['active']:
                continue
            last_kp = track_info['kps'][-1]
            distance = np.sqrt((kp.pt[0] - last_kp.pt[0]) ** 2 + (kp.pt[1] - last_kp.pt[1]) ** 2)
            if distance < min_distance and distance < distance_threshold:
                min_distance = distance
                matched_track_id = track_id
        return matched_track_id


def extract_SIFT_keypoints_and_descriptors(img):
    sift = cv2.SIFT_create()
    kp, desc = sift.detectAndCompute(img, None)
    return kp, desc


def match_keypoints(desc1, desc2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    return good


def draw_tracks(frame, tracker):
    for track_id, track_info in tracker.tracks.items():
        kps, active, _ = track_info  # Unpack the track information
        if active and len(kps) >= 2:
            # Ensure that kps list contains cv2.KeyPoint objects
            if isinstance(kps[-1], cv2.KeyPoint) and isinstance(kps[-2], cv2.KeyPoint):
                # Draw the line for the movement direction
                cv2.line(frame,
                         (int(kps[-2].pt[0]), int(kps[-2].pt[1])),
                         (int(kps[-1].pt[0]), int(kps[-1].pt[1])),
                         (0, 255, 0), 2)
                # Draw the current keypoint position
                cv2.circle(frame,
                           (int(kps[-1].pt[0]), int(kps[-1].pt[1])),
                           3, (0, 0, 255), -1)


def main(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    tracker = ModifiedPointTracker(max_length=5, nn_thresh=0.7)

    current_frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, desc = extract_SIFT_keypoints_and_descriptors(frame_gray)

        if prev_frame_gray is not None:
            prev_kp, prev_desc = extract_SIFT_keypoints_and_descriptors(prev_frame_gray)
            matches = match_keypoints(prev_desc, desc)
            matched_kp = [None] * len(prev_kp)  # Initialize based on prev_kp's length
            for m in matches:
                # Check that the indices are within the valid range
                if m.queryIdx < len(prev_kp) and m.trainIdx < len(kp):
                    # Ensure that matched_kp is populated with cv2.KeyPoint objects
                    matched_kp[m.queryIdx] = kp[m.trainIdx]

            tracker.predict()
            tracker.correct(matched_kp)
            tracker.update(matched_kp, current_frame_index)

        draw_tracks(frame, tracker)

        cv2.imshow('Frame with SIFT Keypoints Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_frame_gray = frame_gray.copy()
        current_frame_index += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "F-16Video.mp4"
    main(video_path)