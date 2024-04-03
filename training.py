from AVA_Net import AVA_Net  # Import AVA_Net model
from loss import iou_loss_MC  # Import custom loss function

import os  # For interacting with the operating system
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
# Keras functions for data augmentation and model building
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import segmentation_models as sm  # For segmentation metrics and models
# skimage functions for image processing
from skimage import img_as_float, transform, exposure, io, color


# Function to load RGB images
def load_RGB_image(df, path, im_shape, column):
    images = []
    for i, item in df.iterrows():
        temp_item =  os.path.join(path , item[column])
        temp_img = io.imread(temp_item)
        temp_img = transform.resize(temp_img, [im_shape[0], im_shape[1], 3])
        images.append(temp_img)
    images = np.array(images)
    print(' --- Images loaded --- ');
    print('\t{}'.format(images.shape))
    return images

# Function to load grayscale images
def load_L_image(df, path, im_shape, column):
    images = []
    for i, item in df.iterrows():
        temp_item =  os.path.join(path , item[column])
        temp_img = io.imread(temp_item)
        temp_img = transform.resize(temp_img, [im_shape[0], im_shape[1], 3])
        images.append(temp_img[:, :, 0])
    images = np.expand_dims(np.array(images), axis=3)
    print(' --- Images loaded --- ');
    print('\t{}'.format(images.shape))
    return images

# Setting up directories and file paths
parent_dir = ''  # Main directory path
train_csv_path = (os.path.join(parent_dir, 'train_data.csv'))
image_dir = os.path.join(parent_dir, 'Dataset\\Train Input')
mask_dir = os.path.join(parent_dir, 'Dataset\\Train Output')

# Training parameters
im_shape = (320, 320)
seed = 7
max_epochs = 5000
learning_rate = 1e-4

# Load training data
df_train = pd.read_csv(train_csv_path)
num_imgs = len(df_train)

# Load training images and masks
X_train = load_RGB_image(df_train, image_dir, im_shape, 'Input')
y_train = load_L_image(df_train, mask_dir, im_shape, 'AV')

# Data augmentation parameters
data_gen_args = dict(
    rescale=1./255,
    rotation_range=70,
    shear_range=0.4,
    zoom_range=0.5,
    brightness_range=[0.1, 1.9],
    width_shift_range=0.3,
    height_shift_range=0.3,
    fill_mode="reflect",
    horizontal_flip=True,
    vertical_flip=True
)
mask_data_gen_args = data_gen_args.copy()  # Same augmentation for masks

# Create data generators
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**mask_data_gen_args)
image_datagen.fit(X_train)
mask_datagen.fit(y_train)

# Create data iterators
image_generator = image_datagen.flow_from_dataframe(
    df_train,
    directory=image_dir,
    x_col='Input',
    batch_size=len(X_train),  # Using full batch size
    class_mode=None,
    seed=seed,
)
mask_generator = mask_datagen.flow_from_dataframe(
    df_train,
    directory=mask_dir,
    x_col='AV',
    batch_size=len(X_train),  # Using full batch size
    class_mode=None,
    seed=seed,
)
train_generator = zip(image_generator, mask_generator)  # Combine generators

# Define model input
inp = Input(shape=(None, None, 3))

# Initialize base model
base_model = AVA_Net()
base_model.summary()

# Create model
out = base_model(inp)
model = Model(inp, out, name=base_model.name)
model.summary()

# Compile model
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss=iou_loss_MC,
    metrics=[sm.metrics.IOUScore(class_weights=1, threshold=0.5), sm.metrics.FScore(class_weights=1, threshold=0.5), 'acc']
)

# Train model
model.fit(
    train_generator, 
    steps_per_epoch=(X_train.shape[0] + len(X_train) - 1) // len(X_train),
    epochs=max_epochs
)
