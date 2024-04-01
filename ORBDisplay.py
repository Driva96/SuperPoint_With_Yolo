import cv2
import numpy as np


def initialize_detectors_and_descriptors():
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Initialize SIFT detector and descriptor
    sift = cv2.SIFT_create()

    # Initialize FAST detector
    fast = cv2.FastFeatureDetector_create()

    # Initialize BRIEF descriptor
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    return orb, sift, fast, brief


def process_frame_with_orb(frame, orb):
    # ORB processing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray_frame, None)
    orb_frame = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)
    return orb_frame


def process_frame_with_sift(frame, sift):
    # SIFT processing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray_frame, None)
    sift_frame = cv2.drawKeypoints(frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return sift_frame


def process_frame_with_fast_and_brief(frame, fast, brief):
    # FAST and BRIEF processing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints = fast.detect(gray_frame, None)
    keypoints, descriptors = brief.compute(gray_frame, keypoints)
    fast_brief_frame = cv2.drawKeypoints(frame, keypoints, None, color=(255, 0, 0))
    return fast_brief_frame


def main():
    #cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture("RedDetection.mp4")
    cap = cv2.VideoCapture("F-16Video.mp4")
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    orb, sift, fast, brief = initialize_detectors_and_descriptors()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frames with different algorithms
        orb_frame = process_frame_with_orb(frame, orb)
        #sift_frame = process_frame_with_sift(frame, sift)
        fast_brief_frame = process_frame_with_fast_and_brief(frame, fast, brief)

        # Display the frames
        cv2.imshow('ORB Features', orb_frame)
        #cv2.imshow('SIFT Features', sift_frame)
        cv2.imshow('FAST and BRIEF Features', fast_brief_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
