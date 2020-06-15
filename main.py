
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans, MeanShift
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score


# this algorithm takes 15 minutes to finish for k-means train and tests, and 15 minutes for mean shift train and tests
# totaling 30 minutes run time
# when run plots confusion matrix and visual vocabulary histogram


def main():

    # grid-1 [3*3] k-means train and tests
    # last 2 parameters in test method is kvals and grid size
    cluster, scale, svm_ = train(k_val=50, grid_size=3)
    test(cluster, scale, svm_,  50, 3)
    cluster, scale, svm_ = train(k_val=250, grid_size=3)
    test(cluster, scale, svm_,  250, 3)
    cluster, scale, svm_ = train(k_val=500, grid_size=3)
    test(cluster, scale, svm_,  500, 3)
    print("first train and test group is done")
    #
    # grid-2 [5*5] k-means train and tests
    cluster, scale, svm_ = train(k_val=50, grid_size=5)
    test(cluster, scale, svm_,  50, 5)
    cluster, scale, svm_ = train(k_val=250, grid_size=5)
    test(cluster, scale, svm_,  250, 5)
    cluster, scale, svm_ = train(k_val=500, grid_size=5)
    test(cluster, scale, svm_,  500, 5)
    print("second train and test group is done")

    # key point k-means train and tests
    cluster, scale, svm_ = train(k_val=50, grid_size=0,)
    test(cluster, scale, svm_,  50, 0)
    cluster, scale, svm_ = train(k_val=250, grid_size=0)
    test(cluster, scale, svm_,  250, 0)
    cluster, scale, svm_ = train(k_val=500, grid_size=0)
    test(cluster, scale, svm_,  500, 0)
    print("third train and test group is done")

    # mean shift tests with corresponding grids, k_val=0 means use mean-shift
    cluster, scale, svm_ = train(k_val=0, grid_size=3)
    test(cluster, scale, svm_,  0, 3)
    cluster, scale, svm_ = train(k_val=0, grid_size=5)
    test(cluster, scale, svm_,  0, 5)
    cluster, scale, svm_ = train(k_val=0, grid_size=0)
    test(cluster, scale, svm_,  0, 0)
    print("All done!")


# get grids
def split_image_into_grids(image, grid_size):
    M = image.shape[0]//grid_size
    N = image.shape[1]//grid_size

    tiles = [image[x:x+M, y:y+N] for x in range(0, image.shape[0], M) for y in range(0, image.shape[1], N)]

    return tiles


# test trained parameters
def test(cluster, scale, svm_, kval, grid_size):
    test_path = 'data/test'
    test_images = get_files(test_path)

    count = 0
    true = []
    descriptor_list = []

    name_dict = {"0": "airplanes", "1": "cars", "2": "faces", "3": "motorbikes"}

    sift = cv.xfeatures2d.SIFT_create()

    # for every image get sift descriptors
    for image_path in test_images:
        img = read_image(image_path)

        # grid_size == 0 => use key points
        if grid_size == 0:
            des = get_descriptors(sift, img)
            if des is not None:
                count += 1
                descriptor_list.append(des)

                if "airplanes" in image_path:
                    true.append("airplanes")
                elif "cars" in image_path:
                    true.append("cars")
                elif "faces" in image_path:
                    true.append("faces")
                else:
                    true.append("motorbikes")

        # use grids
        else:
            img = split_image_into_grids(img, grid_size)
            dests = []
            for i in img:
                des = get_descriptors(sift, i)
                if des is not None:
                    dests.extend(des)

            if dests is not None:
                count += 1
                descriptor_list.append(dests)

                if "airplanes" in image_path:
                    true.append("airplanes")
                elif "cars" in image_path:
                    true.append("cars")
                elif "faces" in image_path:
                    true.append("faces")
                else:
                    true.append("motorbikes")

    # if k_val == 0 => use mean shift
    if kval == 0:
        labels_unique = np.unique(cluster.labels_)
        n_clusters_ = len(labels_unique)
        test_features = extract_features(cluster, descriptor_list, count, n_clusters_)
    # use k means
    else:
        test_features = extract_features(cluster, descriptor_list, count, kval)

    test_features = scale.transform(test_features)
    predictions = [name_dict[str(int(i))] for i in svm_.predict(test_features)]

    # use mean shift
    if kval == 0:
        labels_unique = np.unique(cluster.labels_)
        n_clusters_ = len(labels_unique)
        plot_confusions(true, predictions, n_clusters_, grid_size)

    # use k means
    else:
        plot_confusions(true, predictions, kval, grid_size)


# get sift descriptors, calculate visual vocabulary, train svm
def train(k_val, grid_size):
    train_path = 'data/train'
    images = get_files(train_path)

    sift = cv.xfeatures2d.SIFT_create()

    descriptor_list = []
    train_labels = np.array([])
    image_count = len(images)

    # for every image get sift descriptors
    for image_path in images:
        if "airplanes" in image_path:
            class_index = 0
        elif "cars" in image_path:
            class_index = 1
        elif "faces" in image_path:
            class_index = 2
        elif "motorbikes" in image_path:
            class_index = 3
        else:
            class_index = None

        train_labels = np.append(train_labels, class_index)
        img = read_image(image_path)

        # use mean shift
        if grid_size == 0:
            des = get_descriptors(sift, img)
            descriptor_list.append(des)

        # use k means
        else:
            img = split_image_into_grids(img, grid_size)
            dests = []
            for i in img:
                des = get_descriptors(sift, i)
                if des is not None:
                    dests.extend(des)
            descriptor_list.append(dests)

    # stack descriptors
    descriptors = vstack_descriptors(descriptor_list)

    if k_val != 0:
        cluster = cluster_descriptors(descriptors, k_val)

        im_features = extract_features(cluster, descriptor_list, image_count, k_val)

        # standardize features to increase accuracy
        scale = StandardScaler().fit(im_features)
        im_features = scale.transform(im_features)

        plot_histogram(im_features, k_val)

    # use mean shift
    else:
        cluster = cluster_descriptors(descriptors, k_val)
        labels_unique = np.unique(cluster.labels_)
        n_clusters_ = len(labels_unique)

        im_features = extract_features(cluster, descriptor_list, image_count, n_clusters_)
        scale = StandardScaler().fit(im_features)
        im_features = scale.transform(im_features)

    svm_ = svm.SVC()
    svm_.fit(im_features, train_labels)

    return cluster, scale, svm_


def plot_confusions(true, predictions, kval, grid_size):
    np.set_printoptions(precision=2)

    acc = str(accuracy_score(true, predictions) * 100)

    if kval != 0 and grid_size == 0:
        title = 'Confusion matrix k-means with acc= %' + acc\
                + "\nclusters= " + str(kval) + ' using keypoints'

    elif kval == 0 and grid_size == 0:
        title = 'Confusion matrix mean-shift with acc= %' + acc \
                + "\nclusters= " + str(kval) + ' using keypoints'

    elif kval == 0 and grid_size != 0:
        title = 'Confusion matrix mean-shift with acc= %' + acc \
                + "\nclusters= " + str(kval) + ' grid size= [' + str(grid_size) + '*' + str(grid_size) + ']'
    else:
        title = 'Confusion matrix k-means with acc= %' + acc \
                + "\nclusters= " + str(kval) + ' grid size= [' + str(grid_size) + '*' + str(grid_size) + ']'

    class_names = ["airplanes", "cars", "faces", "motorbikes"]

    plot_confusionMatrix(true, predictions, classes=class_names, title=title)

    plt.show()


def plot_confusionMatrix(y_true, y_pred, classes, title, cmap=plt.cm.Blues):

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def plot_histogram(im_features, no_clusters):

    x_scalar = np.arange(no_clusters)
    y_scalar = np.array([abs(np.sum(im_features[:, h], dtype=np.int32)) for h in range(no_clusters)])

    plt.bar(x_scalar, y_scalar)
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.title("Visual vocabulary histogram")
    plt.xticks(x_scalar + 0.4, x_scalar)
    plt.show()


# feature extraction method
def extract_features(cluster, descriptor_list, image_count, no_clusters):

    im_features = np.array([np.zeros(no_clusters) for i in range(image_count)])

    for i in range(image_count):
        for j in range(len(descriptor_list[i])):
            feature = descriptor_list[i][j]
            feature = feature.reshape(1, 128)
            idx = cluster.predict(feature)
            im_features[i][idx] += 1

    return im_features


# train cluster algorithms
def cluster_descriptors(descriptors, no_clusters):
    if no_clusters == 0:
        cluster = MeanShift(bin_seeding=True).fit(descriptors)
    else:
        cluster = KMeans(n_clusters=no_clusters).fit(descriptors)

    return cluster


# stack descriptors
def vstack_descriptors(descriptor_list):
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    return descriptors


# use sift to get descriptors
def get_descriptors(sift, img):
    kp, des = sift.detectAndCompute(img, None)
    if des is not None:
        return des


# read image and resize to save time
def read_image(img_path):
    img = cv.imread(img_path, 0)
    return cv.resize(img, (150, 150))


def get_files(path):
    images = []
    for folder in os.listdir(path):
        for file in os.listdir(os.path.join(path, folder)):
            images.append(os.path.join(path, os.path.join(folder, file)))

    return images


def show_multiple_images(images):
    for image in images:
        plt.imshow(image, 'gray')
        plt.show()


if __name__ == '__main__':
    main()
