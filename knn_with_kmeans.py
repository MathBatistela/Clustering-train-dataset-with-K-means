import numpy as np
import sys
from sklearn.cluster import KMeans

def main(dataTr, dataTs, n_clusters):
    
    train = np.loadtxt(dataTr)
    test = np.loadtxt(dataTs)
    
    """Your dataset structure must be the same as in the repository"""
    y_test = test[:,132]
    X_test = test[:, 1 : 132]

    y_train = train[:,132]
    X_train = train[:, 1 : 132]

    """Normalizing the data"""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    """Splitting the dataset by classes"""
    X_classes_arr = np.array_split(X_train, 10)
    y_classes_arr = np.array_split(y_train, 10)

    X_centers = []
    y_centers = []
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)

    """Clustering each class and formatting the new dataset of centes from each class"""
    for _class, _label in zip(X_classes_arr, y_classes_arr):
        kmeans.fit(_class)
        
        for center in kmeans.cluster_centers_:
            X_centers.append(center)
            y_centers.append(_label[0])
            
    from sklearn import metrics
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')

    """Getting the accuracy by using all the training data"""
    knn.fit(X_train, y_train)
    
    print(f"------------------{knn}------------------")
    print(f"\n - No clustering training dataset (default) accuracy: {metrics.accuracy_score(y_test, knn.predict(X_test))}\n") 
    print("-------------------------------------------------------------------------------------------")
    
    """Getting the accuracy by using the training data clustered by K-means"""
    knn.fit(X_centers, y_centers)
    predicted = knn.predict(X_test)
    
    print(f"\n - Clustering training dataset with K-means: {metrics.accuracy_score(y_test, predicted)}\n")
    print(metrics.classification_report(y_test, predicted))
    
    import matplotlib.pyplot as plt
    disp = metrics.plot_confusion_matrix(knn, X_test, y_test)
    print(f"Confusion matrix:\n{disp.confusion_matrix}\n")
    print("-------------------------------------------------------------------------------------------")    
    disp.figure_.suptitle("Confusion Matrix")
    plt.show()
    
if __name__ == "__main__":
        if len(sys.argv) != 4:
                sys.exit("Use: example.py <dataTrain> <dataTest> <n_clusters>")

        main(sys.argv[1], sys.argv[2], int(sys.argv[3]))