# https://www.tensorflow.org/datasets/overview
# iterating over a dataset

# ds = datasetTrain.take(1)

# for image, label in tfds.as_numpy(ds):
#   print(type(image), type(label), label)
#   plt.imshow(image)
#   plt.title(datasetInfo.features['label'].int2str(label))


# print(datasetInfo)


#-------------------------------------------------------------------------------------------
# Viewing augmented images

# image, label = next(iter(datasetTrain))
# label = datasetInfo.features['label'].int2str(label)
# image = dataResizingRescaling(image)
# imageAugmented = dataAugmention(image)
# plt.figure(0)
# plt.imshow(image)
# plt.title(label)
# plt.figure(1)
# plt.imshow(imageAugmented)
# plt.title(label)