from keras.models import Model
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.applications import xception
from keras.callbacks import ModelCheckpoint

model = xception.Xception(include_top=False, pooling='avg')
for layer in model.layers:
    layer.trainable = False
output = Dense(13, activation='softmax')(model.output)
model = Model(model.input, output)

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
