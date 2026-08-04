from dataclasses import dataclass
from scipy.stats import skew, kurtosis
from termcolor import cprint
from tqdm import tqdm
from typing import List
import cv2
import numpy as np
from keras.applications.resnet import ResNet101
from keras.applications.vgg16 import VGG16
from keras.models import Model
import tensorflow as tf
from SubFunctions.GradCAM import GradCAM
resnet = ResNet101()
vgg16 = VGG16()

from skimage.feature import greycomatrix, greycoprops


@dataclass
class FeatureExtraction(object):
    def __init__(self, video: List):
        self.video = video

    @staticmethod
    def grad_cam(image: np.ndarray) -> np.ndarray:
        cprint('\n')
        cprint("[⚠️] Grand Cam based Deep Flow map ", color='grey', on_color='on_yellow')
        cprint("================================", color='blue')
        preprocess = tf.keras.Sequential([
            tf.keras.layers.Resizing(224, 224),
            tf.keras.layers.Rescaling(1 / 127.5, -1),
        ])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        device = 'cpu'
        layer_ = 'out_relu'
        if device == 'cuda' and len(tf.config.list_physical_devices('GPU')) == 0:
            raise ValueError('There is no cuda !!!')

        model = tf.keras.applications.MobileNetV2(classifier_activation=None)

        cam_obj = GradCAM(model, device, preprocess, layer_)
        # output is tf Tensor, overlay is ndarray
        _, overlay = cam_obj.get_heatmap(image)
        overlay = cv2.resize(overlay, (32, 32))
        return overlay

    @staticmethod
    def get_neighbour(image, x, y):  # comparing bit with threshold value of centre pixel
        try:
            neighbour = image[x][y]
            return neighbour
        except:
            return 0

    def getting_statistical_values(self, image, x, y):
        neighbor1 = self.get_neighbour(image, x - 1, y + 1)
        neighbor2 = self.get_neighbour(image, x, y + 1)
        neighbor3 = self.get_neighbour(image, x + 1, y + 1)
        neighbor4 = self.get_neighbour(image, x + 1, y)
        neighbor5 = self.get_neighbour(image, x + 1, y - 1)
        neighbor6 = self.get_neighbour(image, x, y - 1)
        neighbor7 = self.get_neighbour(image, x - 1, y - 1)
        neighbor8 = self.get_neighbour(image, x - 1, y - 1)

        neighbor_array = np.array([neighbor1, neighbor2, neighbor3, neighbor4,
                                   neighbor5, neighbor6, neighbor7, neighbor8])

        mean = neighbor_array.mean()
        variance = neighbor_array.var()
        std_deviation = neighbor_array.std()
        skew_value = skew(neighbor_array, axis=0, bias=True)
        kurtosis_value = kurtosis(neighbor_array, axis=0, bias=True)

        return [mean, variance, std_deviation, skew_value, kurtosis_value]

    def statistical_features(self, image):

        cprint("[⚠️] Getting Statistical Feature ", color='grey', on_color='on_yellow')
        cprint("================================", color='blue')

        if len(image.shape) != 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (32, 32))

        m, n = image.shape

        # Finding the mean value,variance value,standard deviation value,skew value ,kurtosis value
        mean_image = np.zeros((m, n))
        variance_image = np.zeros((m, n))
        std_image = np.zeros((m, n))
        skew_image = np.zeros((m, n))
        kurtosis_image = np.zeros((m, n))

        # converting image to lbp
        for i in range(0, m):
            for j in range(0, n):
                [mean, variance, std_deviation, skew_value, kurtosis_value] = self.getting_statistical_values(image, i,
                                                                                                              j)

                mean_image[i][j] = mean
                variance_image[i][j] = variance
                std_image[i][j] = std_deviation
                skew_image[i][j] = skew_value
                kurtosis_image[i][j] = kurtosis_value

        statistical_image = np.zeros(shape=(m, n, 5))
        statistical_image[:, :, 0] = mean_image
        statistical_image[:, :, 1] = variance_image
        statistical_image[:, :, 2] = std_image
        statistical_image[:, :, 3] = skew_image
        statistical_image[:, :, 4] = kurtosis_image

        return statistical_image

    def resnet_statistical(self, image) -> np.ndarray:
        cprint('\n')
        cprint("[⚠️] Hybrid Resnet 101 based statistical features ", color='grey', on_color='on_yellow')
        cprint("================================", color='blue')

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        resnet_model = Model(inputs=resnet.inputs, outputs=resnet.layers[2].output)
        outputs = np.squeeze(resnet_model.predict(image))
        outputs = cv2.resize(np.mean(outputs, axis=2), (32, 32))
        outputs = self.statistical_features(outputs)
        return outputs


    @staticmethod
    def sift_vgg(image) -> np.ndarray:
        cprint('\n')
        cprint("[⚠️] Getting Deep SIFT-VGG-16 flow map ", color='grey', on_color='on_yellow')
        cprint("================================", color='blue')

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Applying SIFT detector
        sift = cv2.SIFT_create()
        kp = sift.detect(gray, None)

        # Marking the keypoint on the image using circles
        sift_image = cv2.drawKeypoints(image,
                                kp,
                                image,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        image = cv2.cvtColor(sift_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        vgg16_model = Model(inputs=vgg16.inputs, outputs=vgg16.layers[2].output)
        outputs = np.squeeze(vgg16_model.predict(image))
        outputs = cv2.resize(np.mean(outputs, axis=2), (32, 32))
        return outputs



    def resnet_shape(self, image) -> np.ndarray:
        cprint('\n')
        cprint("[⚠️] Shape Descriptor enabled Resnet101 ", color='grey', on_color='on_yellow')
        cprint("================================", color='blue')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.Canny(image, 100, 200)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        resnet_model = Model(inputs=resnet.inputs, outputs=resnet.layers[2].output)
        outputs = np.squeeze(resnet_model.predict(image))
        outputs = cv2.resize(np.mean(outputs, axis=2), (32, 32))

        return outputs

    def get_features(self) -> np.ndarray:

        Features = []
        for frame in tqdm(self.video, desc='Getting Features from video '):
            gradcam = self.grad_cam(frame)
            resnet_features = self.resnet_statistical(frame)
            vgg_features = self.sift_vgg(frame)
            shape = self.resnet_shape(frame)

            Features.append(np.concatenate([gradcam, resnet_features, np.expand_dims(vgg_features, axis=-1), np.expand_dims(shape, axis=-1)], axis=-1))

        return np.array(Features)

