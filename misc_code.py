# # https://www.tensorflow.org/datasets/overview
# # iterating over a dataset

# ds = datasetTrain.take(1)

# for image, label in tfds.as_numpy(ds):
#   print(type(image), type(label), label)
#   plt.imshow(image)
#   plt.title(datasetInfo.features['label'].int2str(label))

# print(datasetInfo)


#-------------------------------------------------------------------------------------------
# # Viewing augmented images

# image, label = next(iter(datasetTrain)) # only works before dataset is batched
# label = datasetInfo.features['label'].int2str(label)
# image = dataResizingRescaling(image)
# imageAugmented = dataAugmention(image)
# plt.figure(0)
# plt.imshow(image)
# plt.title(label)
# plt.figure(1)
# plt.imshow(imageAugmented)
# plt.title(label)

#-------------------------------------------------------------------------------------------

# # plotting history after training

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='lower right')
# plt.show()

#-------------------------------------------------------------------------------------------
# # loading just model weights (from a path) into an untrained model
# # can only be applied to models of the exact same architecture
# # checkpoints save every N epochs, so as to not have to wait until training is complete

# random_model = create_model()
# random_model.load_weights(checkpoint_path)

# # Then we can evaluate to see that the model was loaded
# loss, accuracy = random_model.evaluate(datasetTest)
# print(loss, accuracy)