from django.shortcuts import render
from django.shortcuts import render, redirect
from .models import *
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import cv2
from keras.preprocessing.image import img_to_array
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import os


# Create your views here.
def index(request):
    return render(request, "home.html")


def about(request):
    return render(request, "about.html")


def upload(request):
    if request.method == 'POST':
        file = request.FILES['brain']
        filename = file.name
        file_path = os.path.join(r'app\static\upload', filename)
        with open(file_path, 'wb') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        # Now use file_path for further processing
        test_image_path = file_path

        np.random.seed(42)
        SIZE = 128

        # Function to load and preprocess a single image
        def load_and_preprocess_single_image(image_path, size):
            img = cv2.imread(image_path, 0)

            if img is not None:
                img = cv2.resize(img, (size, size))
                img_array = img_to_array(img)
                img_array = np.reshape(
                    img_array, (1, size, size, 1)).astype('float32') / 255
                return img_array
            else:
                raise ValueError(f"Error loading image: {image_path}")

        # Function to load and preprocess images
        def load_and_preprocess_images(folder_path, size):
            data = []
            files = os.listdir(folder_path)

            for file in tqdm(files, desc="Loading and preprocessing images"):
                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path, 0)

                if img is not None:
                    img = cv2.resize(img, (size, size))
                    data.append(img_to_array(img))

            if not data:
                raise ValueError("No valid images found.")

            return np.reshape(
                data, (len(data), size, size, 1)).astype('float32') / 255

        # Load and preprocess clean data
        clean_train = load_and_preprocess_images(
            r'C:\Users\0585\Desktop\Denoise\CODE\app\original', SIZE)
        noise_factor = 0.2
        x_train_noisy = clean_train + noise_factor * tf.random.normal(
            shape=clean_train.shape)
        x_test_noisy = clean_train + noise_factor * tf.random.normal(
            shape=clean_train.shape)

        x_train_noisy = tf.clip_by_value(x_train_noisy,
                                         clip_value_min=0.,
                                         clip_value_max=1.)
        x_test_noisy = tf.clip_by_value(x_test_noisy,
                                        clip_value_min=0.,
                                        clip_value_max=1.)

        # Define the denoising autoencoder model
        class Denoise(Model):

            def __init__(self):
                super(Denoise, self).__init__()
                self.encoder = tf.keras.Sequential([
                    layers.Input(shape=(SIZE, SIZE, 1)),
                    layers.Conv2D(16, (3, 3),
                                  activation='relu',
                                  padding='same',
                                  strides=2),
                    layers.Conv2D(8, (3, 3),
                                  activation='relu',
                                  padding='same',
                                  strides=2)
                ])

                self.decoder = tf.keras.Sequential([
                    layers.Conv2DTranspose(8,
                                           kernel_size=3,
                                           strides=2,
                                           activation='relu',
                                           padding='same'),
                    layers.Conv2DTranspose(16,
                                           kernel_size=3,
                                           strides=2,
                                           activation='relu',
                                           padding='same'),
                    layers.Conv2D(1,
                                  kernel_size=(3, 3),
                                  activation='sigmoid',
                                  padding='same')
                ])

            def call(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded

        # Create and compile the autoencoder
        autoencoder = Denoise()
        autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

        # Split the data into train and test sets
        x_train, x_test, _, _ = train_test_split(clean_train,
                                                 clean_train,
                                                 test_size=0.20,
                                                 random_state=0)

        # Train the autoencoder
        autoencoder.fit(x_train_noisy,
                        x_train_noisy,
                        epochs=10,
                        shuffle=True,
                        validation_data=(x_test_noisy, x_test_noisy))

        # Display encoder and decoder summaries
        autoencoder.encoder.summary()
        autoencoder.decoder.summary()

        # Load a single test image for evaluation
        # test_image_path = file
        test_image = load_and_preprocess_single_image(test_image_path, SIZE)

        # # Display the original test image
        # plt.figure(figsize=(5, 5))
        # plt.title("Original Test Image")
        # plt.imshow(test_image.reshape(SIZE, SIZE), cmap="gray")
        # plt.show()

        # Add noise to the test image
        noisy_test_image = test_image + noise_factor * tf.random.normal(
            shape=test_image.shape)
        noisy_test_image = tf.clip_by_value(noisy_test_image,
                                            clip_value_min=0.,
                                            clip_value_max=1.)

        # Denoise the noisy test image using the trained autoencoder
        decoded_test_image = autoencoder.predict(noisy_test_image)
        output_directory = r'app/static/outputs'
        os.makedirs(output_directory, exist_ok=True)
        # Save the original, noisy, and reconstructed test images
        original_image_path = os.path.join(output_directory,
                                           'original_test_image.png')
        noisy_image_path = os.path.join(output_directory,
                                        'noisy_test_image.png')
        reconstructed_image_path = os.path.join(
            output_directory, 'reconstructed_test_image.png')

        cv2.imwrite(original_image_path, test_image.reshape(SIZE, SIZE) * 255)
        # Convert noisy_test_image to NumPy array before saving
        noisy_array = np.array(noisy_test_image)
        cv2.imwrite(noisy_image_path, noisy_array.squeeze() * 255)
        # Convert decoded_test_image to NumPy array before saving
        decoded_array = np.array(decoded_test_image)
        cv2.imwrite(reconstructed_image_path, decoded_array.squeeze() * 255)

        ori = r"/static/outputs/original_test_image.png"
        noi = r"/static/outputs/noisy_test_image.png"    
        re = r"/static/outputs/reconstructed_test_image.png"

        # Get the file paths
        original_image_path, noisy_image_path, reconstructed_image_path
        return render(request, "result.html", {
            "original": ori,
            "noisy": noi,
            "recontructed": re
        })
    

    return render(request, "upload.html")


def register(request):
    if request.method == 'POST':
        uname = request.POST['uname']
        email1 = request.POST['mail']
        aage = request.POST['age']
        add = request.POST['add']
        password = request.POST['passw']
        confirmpassword = request.POST['cpassw']
        if password == confirmpassword:
            # Create an instance of the Register model
            a = Register(name=uname,
                         email=email1,
                         password=password,
                         age=aage,
                         address=add)
            a.save()
            msg = "Successfully Registered"
            return render(request, 'login.html', {"msg": msg})
        mssg = "Registration Failed, Try Again"

        return render(request, "register.html", {'msg': mssg})
    return render(request, "register.html")


def logins(request):
    if request.method == 'POST':
        email = request.POST['useremail']
        password = request.POST['psw']
        d = Register.objects.filter(email=email, password=password).exists()
        print(d)
        print(email)
        print(password)
        if d:
            return redirect(upload)
        else:
            h = "login failed"
            return render(request, "login.html", {"msg": h})
    return render(request, "login.html")
