# https://www.tensorflow.org/datasets/overview
# iterating over a dataset

# ds = datasetTrain.take(1)

# for image, label in tfds.as_numpy(ds):
#   print(type(image), type(label), label)
#   plt.imshow(image)
#   plt.title(datasetInfo.features['label'].int2str(label))


# print(datasetInfo)


#-------------------------------------------------------------------------------------------
#Viewing augmented images

# image, label = next(iter(datasetTrain))
# _ = label = datasetInfo.features['label'].int2str(label)
# _ = image = dataResizingRescaling(image)
# _ = imageAugmented = dataAugmention(image)
# _ = plt.figure(0)
# _ = plt.imshow(image)
# _ = plt.title(label)
# _ = plt.figure(1)
# _ = plt.imshow(imageAugmented)
# _ = plt.title(label)