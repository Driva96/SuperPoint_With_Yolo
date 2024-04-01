import cv2
import numpy as np
from filterpy.kalman import KalmanFilter

class ModifiedPointTracker:
    def __init__(self, max_length=5, nn_thresh=0.7, max_inactive_frames=1):
        self.max_length = max_length
        self.nn_thresh = nn_thresh
        self.tracks = {}  # Tracks store tuples of (keypoints, active_flag, last_updated)
        self.kalman_filters = {}
        self.track_count = 0
        self.max_inactive_frames = max_inactive_frames

    def update(self, matched_kp):
        def match_keypoint_to_track(kp, existing_tracks, distance_threshold=30):
            """
            Matches a new keypoint to an existing track based on spatial proximity.

            Args:
            kp (cv2.KeyPoint): The new keypoint to match.
            existing_tracks (dict): A dictionary of existing tracks, where each key is a track_id,
                                    and each value is a list of cv2.KeyPoint objects in that track.
            distance_threshold (float): The maximum distance between keypoints to consider a match.

            Returns:
            int or None: The track_id of the matching track, or None if no match is found.
            """

            min_distance = float('inf')
            matched_track_id = None

            for track_id, track_kps in existing_tracks.items():
                if not track_kps:
                    continue

                # Calculate the distance between the new keypoint and the last keypoint in the track
                last_kp = track_kps[-1]
                distance = np.sqrt((kp.pt[0] - last_kp.pt[0]) ** 2 + (kp.pt[1] - last_kp.pt[1]) ** 2)

                # Check if this is the closest match so far and within the distance threshold
                if distance < min_distance and distance < distance_threshold:
                    min_distance = distance
                    matched_track_id = track_id
            import cv2
            import numpy as np
            from filterpy.kalman import KalmanFilter

            class ModifiedPointTracker:
                def __init__(self, max_length=5, nn_thresh=0.7, max_inactive_frames=1):
                    self.max_length = max_length
                    self.nn_thresh = nn_thresh
                    self.tracks = {}  # Tracks store tuples of (keypoints, active_flag, last_updated)
                    self.kalman_filters = {}
                    self.track_count = 0
                    self.max_inactive_frames = max_inactive_frames

                def update(self, matched_kp):
                    def match_keypoint_to_track(kp, existing_tracks, distance_threshold=30):
                        """
                        Matches a new keypoint to an existing track based on spatial proximity.

                        Args:
                        kp (cv2.KeyPoint): The new keypoint to match.
                        existing_tracks (dict): A dictionary of existing tracks, where each key is a track_id,
                                                and each value is a list of cv2.KeyPoint objects in that track.
                        distance_threshold (float): The maximum distance between keypoints to consider a match.

                        Returns:
                        int or None: The track_id of the matching track, or None if no match is found.
                        """

                        min_distance = float('inf')
                        matched_track_id = None

                        for track_id, track_kps in existing_tracks.items():
                            if not track_kps:
                                continue

                            # Calculate the distance between the new keypoint and the last keypoint in the track
                            last_kp = track_kps[-1]
                            distance = np.sqrt((kp.pt[0] - last_kp.pt[0]) ** 2 + (kp.pt[1] - last_kp.pt[1]) ** 2)

                            # Check if this is the closest match so far and within the distance threshold
                            if distance < min_distance and distance < distance_threshold:
                                min_distance = distance
                                matched_track_id = track_id

                        return matched_track_id

                    # Track existing tracks for updates
                    updated_tracks = set()

                    # Update tracks with matched keypoints
                    for kp in matched_kp:
                        track_id = match_keypoint_to_track(kp, self.tracks)

                        if track_id is not None:
                            # KeyPoint matched to an existing track
                            self.tracks[track_id].append(kp)
                            updated_tracks.add(track_id)

                            # Update Kalman filter for this track
                            kf = self.kalman_filters[track_id]
                            measurement = np.array([kp.pt[0], kp.pt[1]]).reshape(2, 1)
                            kf.update(measurement)
                        else:
                            # New keypoint, create a new track
                            self.track_count += 1
                            self.tracks[self.track_count] = [kp]
                            kf = KalmanFilter(dim_x=4, dim_z=2)
                            # Initialize Kalman filter here...
                            self.kalman_filters[self.track_count] = kf

                    # Optionally remove or handle tracks that were not updated
                    for track_id in list(self.tracks.keys()):
                        if track_id not in updated_tracks:
                            # This track did not get any new keypoints
                            # Decide whether to remove it or handle it differently
                            pass

                    # Limit track history to max_length
                    for track_id in self.tracks:
                        if len(self.tracks[track_id]) > self.max_length:
                            self.tracks[track_id] = self.tracks[track_id][-self.max_length:]

                def predict(self):
                    for kf in self.kalman_filters.values():
                        kf.predict()

                def correct(self, matched_kp):
                    for track_id, kf in self.kalman_filters.items():
                        if track_id in matched_kp:
                            measurement = np.array([[matched_kp[track_id].pt[0]], [matched_kp[track_id].pt[1]]])
                            kf.update(measurement)

            def extract_SIFT_keypoints_and_descriptors(img):
                sift = cv2.SIFT_create()
                kp, desc = sift.detectAndCompute(img, None)
                return kp, desc

            def match_keypoints(desc1, desc2):
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(desc1, desc2, k=2)
                good = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append(m)
                return good

            def draw_tracks(frame, tracker):
                for track_id, kps in tracker.tracks.items():
                    # Draw only the latest keypoint and a line to its immediate predecessor
                    if len(kps) >= 2:
                        cv2.line(frame,
                                 (int(kps[-2].pt[0]), int(kps[-2].pt[1])),
                                 (int(kps[-1].pt[0]), int(kps[-1].pt[1])),
                                 (0, 255, 0), 2)  # Line showing movement direction
                        cv2.circle(frame,
                                   (int(kps[-1].pt[0]), int(kps[-1].pt[1])),
                                   3, (0, 0, 255), -1)  # Current keypoint position

            def main(video_path):
                cap = cv2.VideoCapture(video_path)
                ret, prev_frame = cap.read()
                prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                tracker = ModifiedPointTracker(max_length=5, nn_thresh=0.7)
                i = 0

                while cap.isOpened():
                    ret, frame = cap.read()  # Read the new frame
                    if not ret:
                        break

                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    kp, desc = extract_SIFT_keypoints_and_descriptors(frame_gray)

                    if prev_frame_gray is not None:
                        prev_kp, prev_desc = extract_SIFT_keypoints_and_descriptors(prev_frame_gray)
                        matches = match_keypoints(prev_desc, desc)
                        matched_kp = [kp[m.trainIdx] for m in matches if m.trainIdx < len(kp)]

                        tracker.predict()
                        tracker.correct(matched_kp)
                        tracker.update(matched_kp)

                    draw_tracks(frame, tracker)

                    cv2.imshow('Frame with SIFT Keypoints Tracking', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    prev_frame_gray = frame_gray.copy()  # Update prev_frame_gray for the next iteration
                    i += 1

            if __name__ == "__main__":
                video_path = "F-16Video.mp4"
                main(video_path)

            return matched_track_id

        # Track existing tracks for updates
        updated_tracks = set()

        # Update tracks with matched keypoints
        for kp in matched_kp:
            track_id = match_keypoint_to_track(kp, self.tracks)

            if track_id is not None:
                # KeyPoint matched to an existing track
                self.tracks[track_id].append(kp)
                updated_tracks.add(track_id)

                # Update Kalman filter for this track
                kf = self.kalman_filters[track_id]
                measurement = np.array([kp.pt[0], kp.pt[1]]).reshape(2, 1)
                kf.update(measurement)
            else:
                # New keypoint, create a new track
                self.track_count += 1
                self.tracks[self.track_count] = [kp]
                kf = KalmanFilter(dim_x=4, dim_z=2)
                # Initialize Kalman filter here...
                self.kalman_filters[self.track_count] = kf

        # Optionally remove or handle tracks that were not updated
        for track_id in list(self.tracks.keys()):
            if track_id not in updated_tracks:
                # This track did not get any new keypoints
                # Decide whether to remove it or handle it differently
                pass

        # Limit track history to max_length
        for track_id in self.tracks:
            if len(self.tracks[track_id]) > self.max_length:
                self.tracks[track_id] = self.tracks[track_id][-self.max_length:]

    def predict(self):
        for kf in self.kalman_filters.values():
            kf.predict()

    def correct(self, matched_kp):
        for track_id, kf in self.kalman_filters.items():
            if track_id in matched_kp:
                measurement = np.array([[matched_kp[track_id].pt[0]], [matched_kp[track_id].pt[1]]])
                kf.update(measurement)


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
    for track_id, kps in tracker.tracks.items():
        # Draw only the latest keypoint and a line to its immediate predecessor
        if len(kps) >= 2:
            cv2.line(frame,
                     (int(kps[-2].pt[0]), int(kps[-2].pt[1])),
                     (int(kps[-1].pt[0]), int(kps[-1].pt[1])),
                     (0, 255, 0), 2)  # Line showing movement direction
            cv2.circle(frame,
                       (int(kps[-1].pt[0]), int(kps[-1].pt[1])),
                       3, (0, 0, 255), -1)  # Current keypoint position




def main(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    tracker = ModifiedPointTracker(max_length=5, nn_thresh=0.7)
    i = 0

    while cap.isOpened():
        ret, frame = cap.read()  # Read the new frame
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, desc = extract_SIFT_keypoints_and_descriptors(frame_gray)

        if prev_frame_gray is not None:
            prev_kp, prev_desc = extract_SIFT_keypoints_and_descriptors(prev_frame_gray)
            matches = match_keypoints(prev_desc, desc)
            matched_kp = [kp[m.trainIdx] for m in matches if m.trainIdx < len(kp)]

            tracker.predict()
            tracker.correct(matched_kp)
            tracker.update(matched_kp)

        draw_tracks(frame, tracker)

        cv2.imshow('Frame with SIFT Keypoints Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_frame_gray = frame_gray.copy()  # Update prev_frame_gray for the next iteration
        i+=1


if __name__ == "__main__":
    video_path = "F-16Video.mp4"
    main(video_path)
