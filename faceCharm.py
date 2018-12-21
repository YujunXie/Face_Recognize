import os
import numpy as np
import json
import dlib
import cv2 as cv
import math
import pickle
import argparse
from sklearn import neighbors
from PIL import Image, ImageDraw

def bulid_DB(DirPath, detector, sp, facerec):

    data = np.zeros((1, 128))
    label = []

    # 1 - read images
    for root, dirs, files in os.walk(DirPath):
        for d in dirs:
            labelName = d
            Filepath = os.path.join(root, d)
            for root1, dirs1, files1 in os.walk(Filepath):
                #for f in files1:
                ImagePath = os.path.join(root1, files1[0])
                im = Image.open(ImagePath)
                array_im = np.array(im)

                # 2 - detect faces
                face_locations = detector(array_im, upsample_num_times = 1)

                for l in face_locations:
                    rec = dlib.rectangle(max(l.left(), 0), max(l.top(), 0), min(l.right(), array_im.shape[1]), min(l.bottom(), array_im.shape[0]))
                    landmarks = sp(array_im, rec)

                    '''
                    # 3 - align faces
                    faces = dlib.full_object_detections()
                    faces.append(landmarks)
                    im = dlib.get_face_chips(array_im, faces, 160, 0.1)
                    #Image.fromarray(im[0]).show()
                    '''

                    # 4 - computer feature vectors
                    face_descriptor = facerec.compute_face_descriptor(im, landmarks)
                    faceArray = np.array(face_descriptor).reshape((1, 128))
                    data = np.concatenate((data, faceArray))
                    label.append(labelName)
                    print(labelName)

    # 保存人脸合成的矩阵到本地
    data = data[1:, :]
    np.savetxt('faceData.txt', data, fmt='%f')

    labelFile = open('label.txt', 'w')
    json.dump(label, labelFile)
    labelFile.close()

def findLabelInDB(face_descriptor, threshod):

    #data = np.zeros((1, 128))
    #faceLabel = []
    data = np.loadtxt('faceData.txt', dtype=np.float32)

    fl = open("label.txt", 'r')
    faceLabel = json.load(fl)

    temp = face_descriptor - data
    e = np.linalg.norm(temp, axis=1, keepdims=True)
    min_distance = e.min()
    print(min_distance)

    if(min_distance > threshod):
        return 'John Doe'

    index = np.argmin(e)
    return faceLabel[index]

def drawBox(im, location, label):

    draw = ImageDraw.Draw(im)

    left, top, right, bottom = (
    max(location.left(), 0), max(location.top(), 0), min(location.right(), np.array(im).shape[1]), min(location.bottom(), np.array(im).shape[0]))
    draw.rectangle((left, top, right, bottom), outline=(0, 0, 255))
    label = label.encode("UTF-8")
    text_width, text_height = draw.textsize(label)
    draw.rectangle((left, bottom, right, bottom + text_height * 2), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + (right - left - text_width) / 2, bottom + text_height / 2), label, fill=(255, 255, 255, 255))

    im.show()

def recognitionByDB(ImgPath, detector, sp, facerec, threshold):
    im = Image.open(ImgPath)
    array_im = np.array(im)
    face_locations = detector(array_im, upsample_num_times=1)

    for l in face_locations:
        landmarks = sp(array_im, l)

        # 4 - computer feature vectors
        face_descriptor = facerec.compute_face_descriptor(array_im, landmarks)

        # 5 - compute distance
        face_descriptor = np.array(face_descriptor).reshape((1, 128))
        label = findLabelInDB(face_descriptor, threshold)
        print(label)

        drawBox(im, l, label)

def trainKnn(DirPath, n_neighbors, detector, sp, facerec):
    X_train = []
    y_train = []

    for root, dirs, files in os.walk(DirPath):
        for d in dirs:
            labelName = d
            Filepath = os.path.join(root, d)
            for root1, dirs1, files1 in os.walk(Filepath):
                for f in files1:
                    ImagePath = os.path.join(root1, f)
                    im = Image.open(ImagePath)
                    array_im = np.array(im)

                    face_locations = detector(array_im, upsample_num_times=1)

                    for l in face_locations:
                        rec = dlib.rectangle(max(l.left(), 0), max(l.top(), 0), min(l.right(), array_im.shape[1]),
                                             min(l.bottom(), array_im.shape[0]))
                        landmarks = sp(array_im, rec)

                        face_descriptor = facerec.compute_face_descriptor(im, landmarks)
                        face_encoding = np.array(face_descriptor)

                        # Add face encoding for current image to the training set
                        X_train.append(face_encoding)
                        y_train.append(labelName)
                print(labelName)

    if n_neighbors <= 0:
        n_neighbors = int(round(math.sqrt(len(X_train))))

    # Create and train the KNN classifier
    # algorithm: ball_tree, kd_tree, brute, auto
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm="ball_tree", weights='distance')
    knn_clf.fit(X_train, y_train)

    # Save the trained KNN classifier
    with open('models/trained_knn_model.clf', 'wb') as f:
        pickle.dump(knn_clf, f)

def recognitionByKnn(ImgPath, detector, sp, facerec, threshold):

    #if knn_clf is None and model_path is None:
        #raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model
    with open('models/trained_knn_model.clf', 'rb') as f:
        knn_clf = pickle.load(f)

    im = Image.open(ImgPath)
    array_im = np.array(im)
    face_locations = detector(array_im, upsample_num_times=1)

    for l in face_locations:
        # align faces
        landmarks = sp(array_im, l)

        # computer feature vectors
        face_descriptor = facerec.compute_face_descriptor(array_im, landmarks)
        face_encoding = np.array(face_descriptor).reshape((1, 128))

        # Use the KNN model to find the best matches for the test face
        closest_distances = knn_clf.kneighbors(face_encoding, n_neighbors=1)

        if closest_distances[0][0][0] > threshold:
            label_name = 'John Doe'
        else:
            label_name = knn_clf.predict(face_encoding)[0]

        print(label_name)
        drawBox(im, l, label_name)

def onlineRecognize(img, detector,sp, facerec):
    # dlib检测器
    faces = detector(img, upsample_num_times = 1)
    for face in faces:
        landmarks = sp(img, face)
        face_descriptor = facerec.compute_face_descriptor(img, landmarks)
        face_descriptor = np.array(face_descriptor).reshape((1, 128))
        label = findLabelInDB(face_descriptor, 0.6)
        print(label)

        top, right, bottom, left = max(face.top(), 0), min(face.right(), img.shape[1]), min(face.bottom(), img.shape[0]), max(face.left(), 0)
        cv.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        cv.putText(img, label, (left, bottom), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

    cv.imshow('Image', img)

    '''
    # opencv检测器
    cap = cv.CascadeClassifier("models/haarcascade_frontalface_default.xml")
    faceRects = cap.detectMultiScale(img, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50))
    if len(faceRects):
        for faceRect in faceRects:
            x, y, w, h = faceRect
            cv.rectangle(img, (x, y), (x + h, y + w), (0, 255, 0), 2)
    cv.imshow("Image", img)
    '''

def Makeup(ImagePath, detector, sp):

    im = Image.open(ImagePath)
    array_im = np.array(im)
    face_locations = detector(array_im, upsample_num_times=1)

    face_landmarks_list = []

    for l in face_locations:
        rec = dlib.rectangle(max(l.left(), 0), max(l.top(), 0), min(l.right(), array_im.shape[1]),
                             min(l.bottom(), array_im.shape[0]))
        landmarks = sp(array_im, rec)
        points = [(p.x, p.y) for p in landmarks.parts()]

        face_landmarks_list = [{
            "chin": points[0:17],
            "left_eyebrow": points[17:22],
            "right_eyebrow": points[22:27],
            "nose_bridge": points[27:31],
            "nose_tip": points[31:36],
            "left_eye": points[36:42],
            "right_eye": points[42:48],
            "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
            "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [
                points[64]]
        }]

    for face_landmarks in face_landmarks_list:
        draw = ImageDraw.Draw(im)

        # Makeup the eyebrows
        draw.polygon(face_landmarks['left_eyebrow'], fill=(0, 0, 0))
        draw.polygon(face_landmarks['right_eyebrow'], fill=(0, 0, 0))

        # Makeup the eyes
        draw.polygon(face_landmarks['left_eye'], outline=(0, 0, 255))
        draw.polygon(face_landmarks['right_eye'], outline=(0, 0, 255))

        # Makeup the lips
        draw.polygon(face_landmarks['top_lip'], fill=(255, 0, 0))
        draw.polygon(face_landmarks['bottom_lip'], fill=(255, 0, 0))

        im.show()
