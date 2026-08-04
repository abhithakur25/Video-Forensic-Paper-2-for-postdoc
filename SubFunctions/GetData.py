from dataclasses import dataclass
import glob
import pickle

import numpy as np
from termcolor import cprint
import random
import tqdm

from SubFunctions.GetFeatures import FeatureExtraction
from SubFunctions.GetPreprocessing import Preprocessing


@dataclass
class ReadDataset(object):
    def __init__(self, exec: bool):
        self.exec = exec

    def read_data(self) -> dict:

        if self.exec:
            cprint(f"[⁉️] Extracting the Extracted Features and Labels ", color='grey', on_color='on_white')

            path1 = glob.glob("DATASET\\manipulated_sequences\\FaceSwap\\c23\\videos\\*.mp4")
            path2 = glob.glob("DATASET\\original_sequences\\youtube\\c23\\videos\\*.mp4")
            path = path1 + path2
            random.shuffle(path)

            videos, filenames = Preprocessing(path).get_preprocessing()


            Features = []  # List to store the features
            Labels = []  # List to store labels

            for video, filename in tqdm.tqdm(zip(videos, filenames), desc='Reading Videos'):

                if filename.split("\\")[-5] == "original_sequences":
                    Labels.append(0)
                else:
                    Labels.append(1)


                features = FeatureExtraction(video).get_features()
                Features.append(features)

            Features = np.array(Features)
            Labels = np.array(Labels).astype(int)

            # Save the extracted features and labels to a pickle file for future use
            data = {
                'features': Features,
                'labels': Labels,
            }

            with open(f'Features\\Features.pkl', 'wb') as f:
                pickle.dump(data, f)

            # Print success message after feature extraction
            cprint(f"[✅] Feature Extraction Done !! ", color='grey', on_color='on_white')

        else:
            # Load the extracted features from the pickle file if `exec` is False
            cprint(f"[⁉️] Loading the Extracted Features and Labels ", color='grey', on_color='on_white')

            with open(f'Features\\Features.pkl', 'rb') as f:
                data = pickle.load(f)

            # Print success message after loading the data
            cprint(f"[✅] Feature Loading Done !! ", color='grey', on_color='on_white')

        return data  # Return the data (features and labels)
