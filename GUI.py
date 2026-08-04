from tkinter import filedialog
import customtkinter
from PIL import Image
from matplotlib import image as imm
customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("Temp/themes/rose.json")  # Themes: "blue" (standard), "green", "dark-blue"
from scipy.stats import skew, kurtosis
import cv2
import numpy as np
from keras.applications.resnet import ResNet101
from keras.applications.vgg16 import VGG16
from keras.models import Model
import tensorflow as tf
from SubFunctions.GradCAM import GradCAM
resnet = ResNet101()
vgg16 = VGG16()


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("VIDEO FORGERY DETECTION")
        self.geometry(f"{930}x{500}")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # create navigation frame
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(8, weight=1)

        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text="Video Forgery Detection",
                                                             compound="left",
                                                             font=customtkinter.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        self.select_data_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                                          border_spacing=10,
                                                          text="Select Video",
                                                          fg_color="transparent", text_color=("gray10", "gray90"),
                                                          hover_color=("gray70", "gray30"),
                                                          anchor="w", command=self.select_data_event,
                                                          font=customtkinter.CTkFont(size=12, weight="bold"))
        self.select_data_button.grid(row=1, column=0, sticky="ew")


        self.preprocessing_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                                            border_spacing=10,
                                                            text="Preprocessing",
                                                            fg_color="transparent", text_color=("gray10", "gray90"),
                                                            hover_color=("gray70", "gray30"),
                                                            anchor="w", command=self.preprocessing_event,
                                                            font=customtkinter.CTkFont(size=12, weight="bold"))
        self.preprocessing_button.grid(row=2, column=0, sticky="ew")


        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text="Get Features",
                                                             compound="left",
                                                             font=customtkinter.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=3, column=0, padx=20, pady=20)

        self.gradcam_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                                                 border_spacing=10,
                                                                 text="GradCAM",
                                                                 fg_color="transparent",
                                                                 text_color=("gray10", "gray90"),
                                                                 hover_color=("gray70", "gray30"),
                                                                 anchor="w", command=self.get_gradcam,
                                                                 font=customtkinter.CTkFont(size=12, weight="bold"))
        self.gradcam_button.grid(row=4, column=0, sticky="ew")

        self.res_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                                      border_spacing=10,
                                                      text="Resnet Statistical",
                                                      fg_color="transparent",
                                                      text_color=("gray10", "gray90"),
                                                      hover_color=("gray70", "gray30"),
                                                      anchor="w", command=self.get_resnetstat,
                                                      font=customtkinter.CTkFont(size=12, weight="bold"))
        self.res_button.grid(row=5, column=0, sticky="ew")


        self.vgg_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                                      border_spacing=10,
                                                      text="SIFT-VGG16",
                                                      fg_color="transparent",
                                                      text_color=("gray10", "gray90"),
                                                      hover_color=("gray70", "gray30"),
                                                      anchor="w", command=self.get_vgg,
                                                      font=customtkinter.CTkFont(size=12, weight="bold"))
        self.vgg_button.grid(row=6, column=0, sticky="ew")

        self.opt_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                                      border_spacing=10,
                                                      text="Shape-Resnet",
                                                      fg_color="transparent",
                                                      text_color=("gray10", "gray90"),
                                                      hover_color=("gray70", "gray30"),
                                                      anchor="w", command=self.get_flow,
                                                      font=customtkinter.CTkFont(size=12, weight="bold"))
        self.opt_button.grid(row=7, column=0, sticky="ew")

        self.refresh_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                                      border_spacing=10,
                                                      text="Refresh",
                                                      fg_color="transparent", text_color=("gray10", "gray90"),
                                                      hover_color=("gray70", "gray30"),
                                                      anchor="w", command=self.refresh_event,
                                                      font=customtkinter.CTkFont(size=12, weight="bold"))
        self.refresh_button.grid(row=8, column=0, sticky="ew")

        self.exit_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                                   border_spacing=10,
                                                   text="Exit",
                                                   fg_color="transparent", text_color=("gray10", "gray90"),
                                                   hover_color=("gray70", "gray30"),
                                                   anchor="w", command=self.exit_event,
                                                   font=customtkinter.CTkFont(size=12, weight="bold"))
        self.exit_button.grid(row=9, column=0, sticky="ew")

        # create home frame
        self.home_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.home_frame.grid_columnconfigure(0, weight=1)

    def select_frame_by_name(self, name):
        # set button color for selected button
        self.select_data_button.configure(fg_color=("gray75", "gray25") if name == "home" else "transparent")
        # self.frame_4_button.configure(fg_color=("gray75", "gray25") if name == "frame_4" else "transparent")

        # show selected frame
        if name == "home":
            self.home_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.home_frame.grid_forget()



    @staticmethod
    def read_video(filename: str):
        frames = []
        cap = cv2.VideoCapture(filename)
        totalNoFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not cap.isOpened():
            print("Error opening video file")
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
        sel = int(totalNoFrames//2)
        return frames[sel]

    def select_data_event(self):
        global img
        filetypes = (
            ('MP4 files', '*.mp4'),
            ('AVI files', '*.avi')
        )
        self.filename = filedialog.askopenfilename(initialdir="DATASET", filetypes=filetypes)

        self.image = self.read_video(self.filename)
        imm.imsave("Temp\\image.jpg", cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))

        self.select_frame_by_name("home")
        image1 = Image.open("Temp\\image.jpg")

        # image1 = Image.fromarray(self.image)
        self.show = customtkinter.CTkImage(image1, size=(200, 200))
        self.Home_1_inside = customtkinter.CTkLabel(self.home_frame, text="Original Frame", compound='bottom',
                                                    image=self.show, font=customtkinter.CTkFont(size=12, weight="bold"))
        self.Home_1_inside.grid(row=4, column=0, padx=10, pady=15)



    @staticmethod
    def roi(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('Temp\\haarcascade_frontalface_alt2.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            face = image[y:y + h, x:x + w]  # Crop the face region from the image
            return face  # Return the cropped face region

        return image

    def preprocessing_event(self):
        self.Home_1_inside.grid_forget()
        self.preprocessed = self.roi(self.image)

        imm.imsave("Temp\\Preprocessed.jpg", cv2.cvtColor(self.preprocessed, cv2.COLOR_BGR2RGB))

        image1 = Image.open("Temp\\Preprocessed.jpg")

        self.show = customtkinter.CTkImage(image1, size=(200, 200))
        self.Home_2_inside = customtkinter.CTkLabel(self.home_frame, text="Preprocessed Image", compound='bottom',
                                                    image=self.show, font=customtkinter.CTkFont(size=12, weight="bold"))
        self.Home_2_inside.grid(row=0, column=0, padx=10, pady=10)


    @staticmethod
    def grad_cam(image: np.ndarray) -> np.ndarray:

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

        return overlay




    def get_gradcam(self):
        self.Home_2_inside.grid_forget()

        gradcam = self.grad_cam(self.preprocessed)
        imm.imsave("Temp\\gradcam.jpg", cv2.cvtColor(gradcam, cv2.COLOR_BGR2RGB))

        image1 = Image.open("Temp\\gradcam.jpg")

        self.show = customtkinter.CTkImage(image1, size=(200, 200))
        self.Home_3_inside = customtkinter.CTkLabel(self.home_frame, text="GradCAM", compound='bottom',
                                                    image=self.show, font=customtkinter.CTkFont(size=12, weight="bold"))
        self.Home_3_inside.grid(row=0, column=0, padx=10, pady=10)


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


        if len(image.shape) != 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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


        return [mean_image, variance_image, std_image, skew_image, kurtosis_image]

    def resnet_statistical(self, image) -> [np.ndarray]:

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        resnet_model = Model(inputs=resnet.inputs, outputs=resnet.layers[2].output)
        outputs = np.squeeze(resnet_model.predict(image))
        resoutputs_ = cv2.resize(np.mean(outputs, axis=2), (128, 128))
        [mean_image, variance_image, std_image, skew_image, kurtosis_image] = self.statistical_features(resoutputs_)
        return [resoutputs_, mean_image, variance_image, std_image, skew_image, kurtosis_image]

    def get_resnetstat(self):
        self.Home_3_inside.grid_forget()

        [resoutputs_, mean_image, variance_image, std_image, skew_image, kurtosis_image] = self.resnet_statistical(self.preprocessed)


        imm.imsave("Temp\\resoutputs_.jpg", resoutputs_)
        imm.imsave("Temp\\mean_image.jpg", mean_image)
        imm.imsave("Temp\\variance_image.jpg", variance_image)
        imm.imsave("Temp\\std_image.jpg", std_image)
        imm.imsave("Temp\\skew_image.jpg", skew_image)
        imm.imsave("Temp\\kurtosis_image.jpg", kurtosis_image)


        image1 = Image.open("Temp\\resoutputs_.jpg")
        image2 = Image.open("Temp\\mean_image.jpg")
        image3 = Image.open("Temp\\variance_image.jpg")
        image4 = Image.open("Temp\\std_image.jpg")
        image5 = Image.open("Temp\\skew_image.jpg")
        image6 = Image.open("Temp\\kurtosis_image.jpg")


        self.show = customtkinter.CTkImage(image1, size=(200, 200))
        self.Home_4_inside = customtkinter.CTkLabel(self.home_frame, text="ResNet", compound='bottom',
                                                    image=self.show, font=customtkinter.CTkFont(size=12, weight="bold"))
        self.Home_4_inside.grid(row=0, column=0, padx=10, pady=10)

        self.show = customtkinter.CTkImage(image2, size=(200, 200))
        self.Home_5_inside = customtkinter.CTkLabel(self.home_frame, text="Mean", compound='bottom',
                                                    image=self.show, font=customtkinter.CTkFont(size=12, weight="bold"))
        self.Home_5_inside.grid(row=0, column=1, padx=10, pady=10)


        self.show = customtkinter.CTkImage(image3, size=(200, 200))
        self.Home_6_inside = customtkinter.CTkLabel(self.home_frame, text="Variance", compound='bottom',
                                                    image=self.show, font=customtkinter.CTkFont(size=12, weight="bold"))
        self.Home_6_inside.grid(row=0, column=2, padx=10, pady=10)


        self.show = customtkinter.CTkImage(image4, size=(200, 200))
        self.Home_7_inside = customtkinter.CTkLabel(self.home_frame, text="STD", compound='bottom',
                                                    image=self.show, font=customtkinter.CTkFont(size=12, weight="bold"))
        self.Home_7_inside.grid(row=1, column=0, padx=10, pady=10)

        self.show = customtkinter.CTkImage(image5, size=(200, 200))
        self.Home_8_inside = customtkinter.CTkLabel(self.home_frame, text="Skew", compound='bottom',
                                                    image=self.show, font=customtkinter.CTkFont(size=12, weight="bold"))
        self.Home_8_inside.grid(row=1, column=1, padx=10, pady=10)

        self.show = customtkinter.CTkImage(image6, size=(200, 200))
        self.Home_9_inside = customtkinter.CTkLabel(self.home_frame, text="Kurtosis", compound='bottom',
                                                    image=self.show, font=customtkinter.CTkFont(size=12, weight="bold"))
        self.Home_9_inside.grid(row=1, column=2, padx=10, pady=10)


    @staticmethod
    def vgg_ldzp(image) -> [np.ndarray]:


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
        outputs = cv2.resize(np.mean(outputs, axis=2), (224, 224))
        return [sift_image, outputs]

    def get_vgg(self):
        self.Home_4_inside.grid_forget()
        self.Home_5_inside.grid_forget()
        self.Home_6_inside.grid_forget()
        self.Home_7_inside.grid_forget()
        self.Home_8_inside.grid_forget()
        self.Home_9_inside.grid_forget()

        [sift_image, vggoutputs] = self.vgg_ldzp(self.preprocessed)

        imm.imsave("Temp\\sift_image.jpg", sift_image)
        imm.imsave("Temp\\vggoutputs.jpg", vggoutputs)

        image1 = Image.open("Temp\\sift_image.jpg")
        image2 = Image.open("Temp\\vggoutputs.jpg")

        self.show = customtkinter.CTkImage(image1, size=(200, 200))
        self.Home_10_inside = customtkinter.CTkLabel(self.home_frame, text="SIFT", compound='bottom',
                                                    image=self.show, font=customtkinter.CTkFont(size=12, weight="bold"))
        self.Home_10_inside.grid(row=0, column=0, padx=10, pady=10)

        self.show = customtkinter.CTkImage(image2, size=(200, 200))
        self.Home_11_inside = customtkinter.CTkLabel(self.home_frame, text="VGG16", compound='bottom',
                                                    image=self.show, font=customtkinter.CTkFont(size=12, weight="bold"))
        self.Home_11_inside.grid(row=0, column=1, padx=10, pady=10)

    @staticmethod
    def object_flow_features(image) -> [np.ndarray]:

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        shape_image = cv2.Canny(image, 100, 200)

        image = cv2.cvtColor(shape_image, cv2.COLOR_GRAY2RGB)
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        resnet_model = Model(inputs=resnet.inputs, outputs=resnet.layers[2].output)
        outputs = np.squeeze(resnet_model.predict(image))
        resnet_outputs = cv2.resize(np.mean(outputs, axis=2), (224, 224))

        return [shape_image, resnet_outputs]

    def get_flow(self):
        self.Home_10_inside.grid_forget()
        self.Home_11_inside.grid_forget()

        [shape_image, resnet_outputs] = self.object_flow_features(self.preprocessed)

        imm.imsave("Temp\\shape_image.jpg", shape_image)
        imm.imsave("Temp\\resnet_outputs.jpg", resnet_outputs)

        image1 = Image.open("Temp\\shape_image.jpg")
        image2 = Image.open("Temp\\resnet_outputs.jpg")

        self.show = customtkinter.CTkImage(image1, size=(200, 200))
        self.Home_12_inside = customtkinter.CTkLabel(self.home_frame, text="Shape Descriptor", compound='bottom',
                                                     image=self.show,
                                                     font=customtkinter.CTkFont(size=12, weight="bold"))
        self.Home_12_inside.grid(row=0, column=0, padx=10, pady=10)



        self.show = customtkinter.CTkImage(image2, size=(200, 200))
        self.Home_13_inside = customtkinter.CTkLabel(self.home_frame, text="Resnet101", compound='bottom',
                                                     image=self.show,
                                                     font=customtkinter.CTkFont(size=12, weight="bold"))
        self.Home_13_inside.grid(row=0, column=0, padx=10, pady=10)


    def refresh_event(self):
        self.home_frame.grid_forget()
        self.Home_12_inside.grid_forget()
        self.Home_13_inside.grid_forget()



    @staticmethod
    def exit_event():
        app.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
