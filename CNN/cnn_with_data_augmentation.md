**Data Augmentation**  
While creating the data generator object with ImageDataGenerator, add more parameters as below to perform
data augmentation:
```python
train_datagen = ImageDataGenerator(
      rotation_range=40,        # 0-180 within which to randomly rotate pitures
      width_shift_range=0.2,    # randomly translate pictures vertically or horizontally
      height_shift_range=0.2,
      shear_range=0.2,          # randomly applying shearing transformation
      zoom_range=0.2,           # randomfilling newly created pixels.
ly zooming inside pictures
      horizontal_flip=True,     # randomly flipping half of the images horizontally.
      fill_mode='nearest')      # filling newly created pixels.

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
```
