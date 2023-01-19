import pprint
import matplotlib.pyplot as plt
import numpy as np
import skimage
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from transformers import RGB2GrayTransformer, HogTransformer
from util import plot_confusion_matrix, plot_bar, hog_ex
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import svm

pp = pprint.PrettyPrinter(indent=4)

data_path = fr"C:\Users\Rares\Desktop\master an 1\sem3\AnimalFace\Image"
width = 80

def main():
    base_name = 'cow_mouse_faces'

    #needs to be called only once
        #include = {'CowHead', 'MouseHead'}
        #read all the images in the directories, resize them and save the dictionaries to a pickle file
        #the label, filename and data
        #resize_all(src=data_path, pklname=base_name, width=width, include=include)

    #load the pickle file
    data = joblib.load(f'{base_name}_{width}x{width}px.pkl')

    # lastSample = data[:len(data)-1]

    #STATS ABOUT THE DATA
    print('number of samples: ', len(data['data']))
    print('keys: ', list(data.keys()))
    print('description: ', data['description'])
    print('image shape: ', data['data'][0].shape)
    print('labels:', np.unique(data['label']))
    print(Counter(data['label']))


    #PLOT SAMPLE DATA
    # use np.unique to get all unique values in the list of labels
    labels = np.unique(data['label'])

    # set up the matplotlib figure and axes, based on the number of labels
    # prepare the plot of unique types
    fig, axes = plt.subplots(1, len(labels))
    fig.set_size_inches(15,4)
    fig.tight_layout()

    # make a plot for every label (equipment) type. The index method returns the
    # index of the first item corresponding to its search string, label in this case
    for ax, label in zip(axes, labels):
        idx = data['label'].index(label)

        ax.imshow(data['data'][idx])
        ax.axis('off')
        ax.set_title(label)

    plt.show()

    X = np.array(data['data'])
    y = np.array(data['label'])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        random_state=42,
    )

    #PLOT TRAIN VS TEST DIST
    plt.suptitle('relative amount of photos per type')
    plot_bar(y_train, loc='left')
    plot_bar(y_test, loc='right')
    plt.legend([
        'train ({0} photos)'.format(len(y_train)),
        'test ({0} photos)'.format(len(y_test))
    ])

    #HOG EXAMPLE
    hog_ex()

    #TRANSFORM DATA WITH PIPELINE
    HOG_pipeline = Pipeline([
        ('grayify', RGB2GrayTransformer()),
        ('hogify', HogTransformer(
            pixels_per_cell=(14, 14),
            cells_per_block=(2, 2),
            orientations=9,
            block_norm='L2-Hys')
         ),
        ('scalify', StandardScaler()),
        ('classify', svm.SVC(kernel='linear'))
    ])

    clf = HOG_pipeline.fit(X_train, y_train)

    #SHOW RESULTS
    best_pred = clf.predict(X_test)
    print('Percentage correct: ', 100*np.sum(best_pred == y_test)/len(y_test))
    cmx_svm = confusion_matrix(y_test, best_pred)
    print(cmx_svm)
    plot_confusion_matrix(cmx_svm, vmax1=225, vmax2=100, vmax3=12)
    plt.show()

    filename = "mouse.PNG"

    # myimage = skimage.io.imread(filename).astype(np.float32)
    # print(myimage)

    # img = data["data"][0]
    #
    # np.array(data['data'])

    random = X_test[-2:-1]

    prediction = clf.predict(random)
    print(f"THE PREDICTION IS {prediction}")

main()