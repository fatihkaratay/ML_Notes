RECAP:  
If you place your data into subdirectories as test - validation, 
The ImageDataGenerator will label them for you.
```python
from tensorflow.keras.preprocessing.image
import ImageDataGenerator
```

**STEP-1: Download the dataset:**  
The actual kaggle cats vs dogs data set includes 25K images. we'll be using only 2K of them for decreasing the
training time.
```commandline
!wget --no-check-certificate https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip
```
------------------------------------------------------------------------------------------------------------------------
**STEP-2: Extract the data to the current directory:**  
```python
import zipfile
local_zip = './cats_and_dogs_filtered.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall()
zip_ref.close()
```
After this step, you should have the unzipped folder contains the training and validation sub-folder.
training dataset: Data that is used to tell the NN that is 'this is what cat or dogs looks like'
validation dataset: images of cats and dogs that NN will not see as part of the training. You can use this to test
                    how well or how badly it does in evaluating if an image is cats or dogs.
------------------------------------------------------------------------------------------------------------------------

**STEP-3: (optional) Check to make sure the subdirectories are exist and contains the desired data.**  
```python
import os

base_dir = 'cats_and_dogs_filtered'

print("Contents of base directory:")
print(os.listdir(base_dir))

print("\nContents of train directory:")
print(os.listdir(f'{base_dir}/train'))

print("\nContents of validation directory:")
print(os.listdir(f'{base_dir}/validation'))
```
------------------------------------------------------------------------------------------------------------------------

**STEP-4: (optional) Assign each of these directories a variable so that we can use them later.**  
```python
import os

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with training cat/dog pictures
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

# Directory with validation cat/dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
```
------------------------------------------------------------------------------------------------------------------------

**STEP-5: Checking the file naming convention and see what's inside.**  
```python
train_cat_fnames = os.listdir( train_cats_dir )
train_dog_fnames = os.listdir( train_dogs_dir )

print(train_cat_fnames[:10])
print(train_dog_fnames[:10])
```

------------------------------------------------------------------------------------------------------------------------

***STEP-6: Find out the total number of cats and dog images in the train and validation directories***  
```python
print('total training cat images :', len(os.listdir(train_cats_dir)))
print('total training dog images :', len(os.listdir(train_dogs_dir)))

print('total validation cat images :', len(os.listdir(validation_cats_dir)))
print('total validation dog images :', len(os.listdir(validation_dogs_dir)))
```

------------------------------------------------------------------------------------------------------------------------

**STEP-7: display a few data to see what they look like.**  .
```python
%matplotlib inline

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

pic_index = 0 # Index for iterating over images

# display a batch or 8 cat and 8 dog pictures. You can rerun this code to generate different images.
# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

pic_index+=8

next_cat_pix = [os.path.join(train_cats_dir, fname)
                for fname in train_cat_fnames[ pic_index-8:pic_index]
               ]

next_dog_pix = [os.path.join(train_dogs_dir, fname)
                for fname in train_dog_fnames[ pic_index-8:pic_index]
               ]

for i, img_path in enumerate(next_cat_pix+next_dog_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()
```

Please note that the images comes with different sizes. They needed to be adjusted before feeding them to the NN
we choose 150x150 pixels.
Those can be done in the fly as seen in the next step
------------------------------------------------------------------------------------------------------------------------

**STEP-8: Building a small model from scratch to get ~72% accuracy**  
```python
import tensorflow as tf

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```
Please note that we have 2 options. Therefore, "sigmoid" activation function is used in here.

------------------------------------------------------------------------------------------------------------------------

**STEP-9: review the model**  
```python
model.summary()
```

------------------------------------------------------------------------------------------------------------------------

**STEP-10: Configure the specifications for model training.**  
```python
loss= binary_crossentropy
final actication=sigmoid
optimizer=RMSprop
learning rate=0.001
metric=accuracy
```
Here RMSprop is a good chooice for SGD because it automates learning rate tunning for us.
Other optimizers such as Adam and Adagrad are also great chooices.
```python
from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics = ['accuracy']
)
```

------------------------------------------------------------------------------------------------------------------------

**STEP-11: Set up data generators that will read pictures in the source folders, convert them to float32 tensors,
and feed them with their labels to the model.**  
You will have one generator for train and validation.
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255.
train_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))
# Flow validation images in batches of 20 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=20,
                                                         class_mode  = 'binary',
                                                         target_size = (150, 150))
```


------------------------------------------------------------------------------------------------------------------------

**STEP-12: Training the data.**  
Note the values per epochs.
Here are some important values for performance:  
Loss.............: Good indicator in the training: measures the current model prediction against the known labels.  
Accuracy.........: Good indicator in the training: portion of correct guesses  
Validation Loss  
Validation Accuracy  
```python

history = model.fit(
            train_generator,
            epochs=15,
            validation_data=validation_generator,
            verbose=2
            )
```
This part can take some time depends on the dataset sizes.

------------------------------------------------------------------------------------------------------------------------

**STEP-13: Model prediction: This step allows you to pick one or more file and test it to make sure the prediction is
correct or not.**  
```python
import numpy as np

from google.colab import files
from tensorflow.keras.utils import load_img, img_to_array

uploaded=files.upload()

for fn in uploaded.keys():

  # predicting images
  path='/content/' + fn
  img=load_img(path, target_size=(150, 150))

  x=img_to_array(img)
  x /= 255
  x=np.expand_dims(x, axis=0)
  images = np.vstack([x])

  classes = model.predict(images, batch_size=10)

  print(classes[0])

  if classes[0]>0.5:
    print(fn + " is a dog")
  else:
    print(fn + " is a cat")
```
------------------------------------------------------------------------------------------------------------------------

**STEP-14: Visualizing the intermediate representations.**  
Select a random image from training set and see the actual progress. Like visualize how an input gets transformed
as it goes through the model.
```python
import numpy as np
import random
from tensorflow.keras.utils import img_to_array, load_img

# Define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model
successive_outputs = [layer.output for layer in model.layers]
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

# Prepare a random input image from the training set.
cat_img_files = [os.path.join(train_cats_dir, f) for f in train_cat_fnames]
dog_img_files = [os.path.join(train_dogs_dir, f) for f in train_dog_fnames]
img_path = random.choice(cat_img_files + dog_img_files)
img = load_img(img_path, target_size=(150, 150))  # this is a PIL image
x   = img_to_array(img)                           # Numpy array with shape (150, 150, 3)
x   = x.reshape((1,) + x.shape)                   # Numpy array with shape (1, 150, 150, 3)

# Scale by 1/255
x /= 255.0

# Run the image through the network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so you can have them as part of our plot
layer_names = [layer.name for layer in model.layers]

# Display the representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):

  if len(feature_map.shape) == 4:

    #-------------------------------------------
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    #-------------------------------------------
    n_features = feature_map.shape[-1]  # number of features in the feature map
    size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)

    # Tile the images in this matrix
    display_grid = np.zeros((size, size * n_features))

    #-------------------------------------------------
    # Postprocess the feature to be visually palatable
    #-------------------------------------------------
    for i in range(n_features):
      x  = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std ()
      x *=  64
      x += 128
      x  = np.clip(x, 0, 255).astype('uint8')
      display_grid[:, i * size : (i + 1) * size] = x # Tile each filter into a horizontal grid

    #-----------------
    # Display the grid
    #-----------------
    scale = 20. / n_features
    plt.figure( figsize=(scale * n_features, scale) )
    plt.title ( layer_name )
    plt.grid  ( False )
    plt.imshow( display_grid, aspect='auto', cmap='viridis' )
```
------------------------------------------------------------------------------------------------------------------------

**STEP-15: Evaluating the model. Specifically accuracy and loss. You plot training/validation accuracy and loss.**  
Retrieve a list of list results on training and test data sets for each training epoch
```python
acc      = history.history[     'accuracy' ]
val_acc  = history.history[ 'val_accuracy' ]
loss     = history.history[    'loss' ]
val_loss = history.history['val_loss' ]

epochs   = range(len(acc)) # Get number of epochs

# Plot training and validation accuracy per epoch
plt.plot  ( epochs,     acc )
plt.plot  ( epochs, val_acc )
plt.title ('Training and validation accuracy')
plt.figure()

# Plot training and validation loss per epoch
plt.plot  ( epochs,     loss )
plt.plot  ( epochs, val_loss )
plt.title ('Training and validation loss'   )

```

By looking at those plots, you can determine if the model is overfitting.
You can observe this by looking at the training accuracy and validation accuracy.
Overfitting occurs when a model exposed to too few examples to learn patters that do not generalize to new data.



