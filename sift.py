import cv2
import numpy as np
from scipy.cluster.vq import kmeans,vq

k_for_k_means = 200

#images that are aleady grayscale, but haven't been reshaped yet
#this is the case for the zalando raw data
def compute_descriptor_list_from_grayscale_arrays(images, dim=28):
    images = list(map(lambda image: image.reshape(dim, dim), images))
    return compute_descriptor_list(images)

#images that are in a proper for but are yet bgr not grayscale
#this is the case for cifar 10
def compute_descriptor_list_from_numpy_arrays(images):
    images = list(map(lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), images))
    return compute_descriptor_list(images)

#expects grayscale images in form of a numpy array
def compute_descriptor_list(images):
    n_no_descriptors = 0
    sift = cv2.xfeatures2d.SIFT_create()
    descriptor_list = []
    print("Computing descriptors...")
    for (index, image) in enumerate(images):
        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is not None and len(descriptors) != 0:
            descriptor_list.append(descriptors)
        else:
            descriptor_list.append(np.zeros((1, 128), dtype=float))
            n_no_descriptors = n_no_descriptors + 1
        if index % 2000 == 0:
            print("Progress {}%".format(int(index / len(images) * 100)))
    print("Finished computing descriptors!")
    print("{} entries received no descriptors through SIFT algorithm!".format(n_no_descriptors))
    return descriptor_list

def unwrap_descriptor_list(descriptor_list):
    print("Unwrapping descriptors...")
    descriptors = descriptor_list[0]
    for (index, descriptor) in enumerate(descriptor_list[1:]):
        descriptors = np.vstack((descriptors, descriptor))
        if index % 2000 == 0:
            print("Progress {}%".format(index / len(descriptor_list) * 100))
    print("Finished unwrapping descriptors!")
    return descriptors

def change_descriptors_to_float(descriptors):
    return descriptors.astype(float)

def compute_vocabulary(descriptor_list_float):
    print("Computing K Means groups for k={}...".format(k_for_k_means))
    voc, variance = kmeans(descriptor_list_float, k_for_k_means, 1)
    print("Finished computing K means!")
    return voc

def compute_image_features(descriptors_list, vocabulary):
    im_features = np.zeros((len(descriptors_list), k_for_k_means), "float32")
    for i in range(len(descriptors_list)):
        if(len(descriptors_list[i]) > 0):
            words, distance = vq(descriptors_list[i], vocabulary)
            for w in words:
                im_features[i][w]+=1
    return im_features