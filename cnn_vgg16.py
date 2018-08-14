from keras.models import Sequential
from keras.layers import Dense, Flatten, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.applications import vgg16
from keras.callbacks import ModelCheckpoint

model = Sequential()
# vgg16_model = vgg16.VGG16(include_top=False, input_shape=(x, y, 3)) # For train in customize input shape
vgg16_model = vgg16.VGG16()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

# model.add(Flatten(name='flatten')) # For train in customize input shape
# model.add(Dense(4096, activation='relu', name='fc1')) # For train in customize input shape
# model.add(Dense(4096, activation='relu', name='fc2')) # For train in customize input shape

for layer in model.layers:
    layer.trainable = False

model.add(Dense(13, activation="softmax"))
model.summary()

train_path = "demo_dataset/train"
valid_path = "demo_dataset/valid"

train_batch = ImageDataGenerator().flow_from_directory(train_path,
                                                       target_size=(224, 224),
                                                       classes=["nm0", "nm1", "nm2", "nm3", "nm4", "nm5", "nm6", "nm7",
                                                                "nm8", "nm9", "sp17", "sp18", "sp19"],
                                                       batch_size=40)
valid_batch = ImageDataGenerator().flow_from_directory(valid_path,
                                                       target_size=(224, 224),
                                                       classes=["nm0", "nm1", "nm2", "nm3", "nm4", "nm5", "nm6", "nm7",
                                                                "nm8", "nm9", "sp17", "sp18", "sp19"],
                                                       batch_size=40)

model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
filepath = "model-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit_generator(train_batch, steps_per_epoch=104,
                    validation_data=valid_batch, validation_steps=26, epochs=300, verbose=2, callbacks=callbacks_list)

model.save("thai_model_finished.h5")
