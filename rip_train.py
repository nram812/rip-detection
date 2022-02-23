import imageio
import glob
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar
from dask import compute, delayed
import xarray as xr
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
from functools import partial
from tf_explain.core.grad_cam import GradCAM
from albumentations import (
    Compose, RandomBrightness, JpegCompression, HueSaturationValue, RandomContrast, HorizontalFlip,
    Rotate)
import random
import albumentations as A


# Load parent directories
base_dirs = r'/nesi/project/niwa00004/ML_DATA/CAMERA/rip-detection'
os.chdir(base_dirs)

# Default preprocessed image size
IMG_SIZE = (224,224)
ppr_func =tf.keras.applications.mobilenet.preprocess_input

# Datasets for rip detection
rips = xr.open_dataset(r'./preprocess_data/rip_training_dataset.nc')
norips = xr.open_dataset(r'./preprocess_data/norip_training.nc')


# loading the data
def create_dataset(rips, norips, n_training_sample_rips=1200,
                   n_training_sample_norips=500, to_categorical=True):
    """

    Parameters
    ----------
    rips: an xarray dataset containing the rip data images
    norips: an xarray dataset containing images with no rips
    n_training_sample_rips:
    n_training_sample_norips

    Returns
    -------

    """
    # creating a binary target rip/no rip variable
    rips['binary'] = (('sample'), np.ones((len(rips.sample))))
    norips['binary'] = (('sample'), np.zeros((len(norips.sample))))
    # extracting the images and concatenting them into an array
    x_rip_train = rips['rip_images'].isel(sample=slice(0, n_training_sample_rips))
    x_norip_train = norips['rip_images'].isel(sample=slice(0, n_training_sample_norips))
    X_train = np.concatenate([x_rip_train, x_norip_train], axis=0)
    y_train = np.array([1] * n_training_sample_rips + [0] * n_training_sample_norips)

    x_rip_test = rips['rip_images'].isel(sample=slice(n_training_sample_rips + 1, None))
    x_norip_test = norips['rip_images'].isel(sample=slice(n_training_sample_norips + 1, None))
    X_test = np.concatenate([x_rip_test, x_norip_test], axis=0)
    y_test = np.array([1] * len(x_rip_test) + [0] * len(x_norip_test))
    index_train = np.arange(len(X_train)).astype('int32')
    index_test = np.arange(len(X_test)).astype('int32')
    random.shuffle(index_train)
    random.shuffle(index_test)
    if to_categorical:
        return X_train[index_train], X_test[index_test], tf.keras.utils.to_categorical(y_train)[index_train], \
               tf.keras.utils.to_categorical(y_test)[index_test]
    else:
        return X_train[index_train], X_test[index_test], y_train[index_train], y_test[index_test]


def view_image(ds):
    image, label = next(iter(ds))  # extract 1 batch from the dataset
    image = image.numpy()
    label = label.numpy()

    fig = plt.figure(figsize=(22, 22))
    for i in range(20):
        ax = fig.add_subplot(4, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(image[i] / 255)
        ax.set_title(f"Label: {label[i]}")

AUTOTUNE = tf.data.experimental.AUTOTUNE
x_train, x_test, y_train, y_test = create_dataset(rips, norips, to_categorical =False)

# Using albumentations api to construct a wide variety of transforms that will be applied to an image
transforms = Compose([
            Rotate(limit=40),
            RandomBrightness(limit=0.1, always_apply = True),
           A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True, p=0.5),
           A.RGBShift(always_apply=False),
            JpegCompression(quality_lower=85, quality_upper=100, p=0.5),
            HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            RandomContrast(limit=0.2, p=0.5),
            HorizontalFlip(),
            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=1, shadow_dimension=2, shadow_roi=(0, 0.5, 1, 1), p=0.1),
        ])

# Identifying the image augumentation functions
def aug_fn(image, img_size,
           preprocess_func=ppr_func, split='test', preprocess=True):
    data = {"image": image}
    if split == 'train':
        aug_data = transforms(**data)
    else:
        aug_data = data
    aug_img = aug_data["image"]
    aug_img = tf.cast(aug_img, tf.float32)

    preprocess_input = preprocess_func
    if preprocess:
        aug_img = preprocess_input(aug_img)

    # aug_img = tf.image.resize(aug_img, size=[img_size, img_size])
    return aug_img


def process_data(image, label, img_size, split, preprocess):
    aug_img = tf.numpy_function(func=partial(aug_fn, split=split, preprocess=preprocess), inp=[image, img_size],
                                Tout=tf.float32)
    return aug_img, label


def set_shapes(img, label, img_shape=(224, 224, 3)):
    img.set_shape(img_shape)
    label.set_shape([])
    return img, label


# Identfying the training data pipeline
x_train_slices = tf.data.Dataset.from_tensor_slices((x_train, y_train))
x_test_slices = tf.data.Dataset.from_tensor_slices((x_test, y_test))
ds_alb_train =x_train_slices.map(partial(process_data, img_size=224, split ='train', preprocess = True),
                  num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
ds_alb_test =x_test_slices.map(partial(process_data, img_size=224, split ='test', preprocess =True),
                  num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
ds_alb_train = ds_alb_train.map(set_shapes, num_parallel_calls=AUTOTUNE).batch(128).prefetch(AUTOTUNE)
ds_alb_test = ds_alb_test.map(set_shapes, num_parallel_calls=AUTOTUNE).batch(128).prefetch(AUTOTUNE)

# Creating a model pipeline using transfer learning

base_model = keras.applications.MobileNet(include_top=False, input_shape=(224, 224, 3), weights="imagenet")
base_model.trainable = True
x = keras.layers.GlobalMaxPooling2D()(base_model.output)
x = keras.layers.Dropout(0.3)(x)
x =  keras.layers.Dense(2, activation='softmax')(x)


model = keras.models.Model(base_model.input, x)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=1e-6)

model.fit(ds_alb_train, validation_data = ds_alb_test,
          epochs =150, shuffle = True,
          callbacks =[reduce_lr], class_weight = {0:3,1:1})
