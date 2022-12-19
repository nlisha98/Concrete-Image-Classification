#%%
#1. Import Packages
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os, datetime
import matplotlib.pyplot as plt
from tensorflow.keras import layers,optimizers,losses,callbacks,applications
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

#%%
#2. Data Loading
FILE_PATH = os.path.join(os.getcwd(),'dataset')

# %%
#3. Data Preparation
#Define batch size and image size
BATCH_SIZE=32
IMG_SIZE=(160,160)
SEED = 12345

#Load the data as tensorflow dataset using the special method
train_dataset = keras.utils.image_dataset_from_directory(FILE_PATH,batch_size=BATCH_SIZE,image_size=IMG_SIZE,seed=SEED, validation_split=0.3,subset='training')
val_dataset = keras.utils.image_dataset_from_directory(FILE_PATH,batch_size=BATCH_SIZE,image_size=IMG_SIZE,seed=SEED, validation_split=0.3,subset='validation')

# %%
#4.Display some examples
#Extract class names as a list
class_names = train_dataset.class_names

#Plot some examples
plt.figure(figsize=(10,10))
for images,labels in train_dataset.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')

# %%
#5. Performing validation-test split
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches//5)
validation_dataset = val_dataset.skip(val_batches//5)

# %%
#6. Convert train,validation and test dataset into prefetch dataset
AUTOTUNE = tf.data.AUTOTUNE

pf_train = train_dataset.prefetch(buffer_size=AUTOTUNE)
pf_val = validation_dataset.prefetch(buffer_size=AUTOTUNE)
pf_test = test_dataset.prefetch(buffer_size=AUTOTUNE)

# %%
#7. Create a 'model' for image augmentation
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))

#%%
#8. Repeatedly apply data augmentation on one image and see the result
for images,labels in pf_train.take(1):
    first_image = images[0]
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_image,axis=0))
        plt.imshow(augmented_image[0]/255.0)
        plt.axis('off')

# %%
#9. Before transfer learning
#Create the layer for data normalization
preprocess_input = applications.mobilenet_v2.preprocess_input

# %%
#10. Start the transfer learning
#(A) Instantiate the pretrained model
IMG_SHAPE = IMG_SIZE + (3,)
base_model = applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')

# %%
#(B) Set the pretrained model as non-trainable (frozen)
base_model.trainable = False
base_model.summary()
keras.utils.plot_model(base_model,show_shapes=True)

# %%
#(C) Create the classifier
#Create the global average pooling layer
global_avg = layers.GlobalAveragePooling2D()
#Create an output layer
output_layer = layers.Dense(len(class_names),activation='softmax')

# %%
#11. Link the layers together to form a pipeline
inputs = keras.Input(shape=IMG_SHAPE)

x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x,training=False)
x = global_avg(x)
x = layers.Dropout(0.3)(x)
outputs = output_layer(x)

#%%
#Instantiate the full model pipeline
model = keras.Model(inputs=inputs,outputs=outputs)
print(model.summary())

# %%
#12. Compile the model
optimizer = optimizers.Adam(learning_rate=0.0001)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

# %%
#13. Evaluate the model before training
loss0, acc0 = model.evaluate(pf_val)

print("----------------Evaluation Before Training----------------")
print("Loss = ", loss0)
print("Accuracy = ", acc0)

# %%
#Callbacks - Early Stopping and TensorBoard
LOGS_PATH = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_callback = TensorBoard(log_dir=LOGS_PATH)
es_callback = EarlyStopping(monitor='val_loss', patience=5)

# %%
#14. Model training
EPOCHS = 10
history = model.fit(pf_train, validation_data=pf_val, epochs=EPOCHS, callbacks=[tb_callback, es_callback])

# %%
#Third strategy of transfer learning: Fine tune pretrained model and frozen layers to improve the model 
#15. Apply transfer learning
base_model.trainable = True
#Use a for loop to freeze some layers
for layer in base_model.layers[:100]:
    layer.trainable= False

base_model.summary()

#%%
#Compile the model
optimizer = optimizers.RMSprop(learning_rate=0.00001)
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

#%%
#Continue model training with new configuration
fine_tune_epoch = 10
total_epoch = EPOCHS + fine_tune_epoch

#Follow up from previous model training
history_fine = model.fit(pf_train, validation_data=pf_val, epochs=total_epoch, initial_epoch=history.epoch[-1],callbacks=[tb_callback, es_callback])

#%%
#18. Model evaluation after training
test_loss,test_acc = model.evaluate(pf_test)
print("----------------Evaluation After Training----------------")
print("Loss = ",test_loss)
print("Accuracy = ",test_acc)

# %%
#19. Model Deployment
#Use the model to perform prediction
image_batch, label_batch = pf_test.as_numpy_iterator().next()
y_pred = np.argmax(model.predict(image_batch),axis=1)

#%%
#Confusion Matrix, Classification Report, Accuracy score
print("confusion matrix: \n", confusion_matrix(label_batch,y_pred))
print("classification report: \n", classification_report(label_batch,y_pred))
print("accuracy score: \n", accuracy_score(label_batch,y_pred))

#%%
#20. Model Saving
save_path = os.path.join(os.getcwd(), 'saved_model','saved_models.h5')
model.save(save_path)