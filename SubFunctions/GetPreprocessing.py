from dataclasses import dataclass
from typing import List
import cv2
import numpy as np
import time
import peakutils
import tqdm
from SubFunctions.utils import convert_frame_to_grayscale


@dataclass
class Preprocessing(object):
    def __init__(self, filenames: List):
        self.filenames = filenames


    @staticmethod
    def get_frame_idx(filename, Thres=0.3):
        cap = cv2.VideoCapture(filename)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if (cap.isOpened() == False):
            print("Error opening video file")

        lstfrm = []
        lstdiffMag = []
        timeSpans = []
        images = []
        full_color = []
        lastFrame = None
        Start_time = time.process_time()

        # Read until video is completed
        for i in range(length):
            ret, frame = cap.read()
            grayframe, blur_gray = convert_frame_to_grayscale(frame)

            frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
            lstfrm.append(frame_number)
            images.append(grayframe)
            full_color.append(frame)
            if frame_number == 0:
                lastFrame = blur_gray

            if blur_gray is None:
                blur_gray = lastFrame


            diff = cv2.subtract(blur_gray, lastFrame)
            diffMag = cv2.countNonZero(diff)
            lstdiffMag.append(diffMag)
            stop_time = time.process_time()
            time_Span = stop_time - Start_time
            timeSpans.append(time_Span)
            lastFrame = blur_gray

        cap.release()
        y = np.array(lstdiffMag)
        base = peakutils.baseline(y, 2)
        indices = peakutils.indexes(y - base, Thres, min_dist=1)[:15]

        return {'length': length, 'indices': indices, 'filename': filename}

    @staticmethod
    def extend_frames(frames):
        # Find the maximum number of indices length
        max_length = max(len(d['indices']) for d in frames)

        # Loop through each dictionary and extend the indices

        for d in tqdm.tqdm(frames):
            indices = list(d['indices'])

            differnc = max_length - len(indices)


            padding = [indices[-1] for _ in range(differnc)]

            indices.extend(padding)

            d['indices'] = np.array(indices)

        return frames


    @staticmethod
    def roi(image: np.ndarray) -> np.ndarray:
        # Method to extract the Region of Interest (ROI) from an image (face detection)

        # Convert the input image to grayscale for better face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Load a pre-trained Haar Cascade classifier for face detection
        face_cascade = cv2.CascadeClassifier('Temp\\haarcascade_frontalface_alt2.xml')

        # Detect faces in the grayscale frame using the classifier
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # If faces are detected, loop through each face and extract the region of interest (face)
        for (x, y, w, h) in faces:
            face = image[y:y + h, x:x + w]  # Crop the face region from the image
            face = cv2.resize(face, (96, 96))
            return face  # Return the cropped face region

        # If no face is detected, the function will return None (optional: could return original image)
        return cv2.resize(image, (96, 96))

    def get_video(self, frames):
        filename = frames['filename']
        cap = cv2.VideoCapture(filename)

        selected_frames = []

        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                # frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 15)
                selected_frames.append(frame)
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
        selected_frames = np.array(selected_frames)[frames['indices']]

        preprocessed_frames = []
        for frame in selected_frames:
            preprocessed_frames.append(self.roi(frame))

        return np.array(preprocessed_frames)



    def get_preprocessing(self):

        selected_frames = []

        for filename in tqdm.tqdm(self.filenames, desc='Selecting Frames'):

            selected = self.get_frame_idx(filename)

            if len(selected['indices']) != 0:
                selected_frames.append(selected)

        selected_frames = self.extend_frames(selected_frames)

        videos = []
        filenames = []
        for frame in tqdm.tqdm(selected_frames, desc='Matching Frames'):

            videos.append(self.get_video(frame))
            filenames.append(frame['filename'])

        return videos, filenames




