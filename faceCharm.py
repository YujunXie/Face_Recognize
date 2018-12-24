#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
import json
import dlib
import cv2 as cv
import math
import pickle
import argparse
from sklearn import neighbors
from PIL import Image, ImageDraw, ImageFont

def bulid_DB(DirPath, detector, sp, facerec):

    data = np.zeros((1, 128))
    label = []
    cnt = 0

    # 1 - read images
    for root, dirs, files in os.walk(DirPath):
        for d in dirs:
            labelName = d.replace('(', '').replace(')', '')
            Filepath = os.path.join(root, d)
            for root1, dirs1, files1 in os.walk(Filepath):

                if not files1:
                    break
                #for f in files1:
                ImagePath = os.path.join(root1, files1[0])

                im = Image.open(ImagePath)
                im = im.convert('RGB')
                array_im = np.array(im)

                # 2 - detect faces
                face_locations = detector(array_im, upsample_num_times = 1)

                for l in face_locations:
                    rec = dlib.rectangle(max(l.left(), 0), max(l.top(), 0), min(l.right(), array_im.shape[1]), min(l.bottom(), array_im.shape[0]))
                    landmarks = sp(array_im, rec)

                    #drawBox(im, l, files1[0])
                    '''
                    # 3 - align faces
                    faces = dlib.full_object_detections()
                    faces.append(landmarks)
                    im = dlib.get_face_chips(array_im, faces, 160, 0.1)
                    #Image.fromarray(im[0]).show()
                    '''
                    # 4 - computer feature vectors
                    face_descriptor = facerec.compute_face_descriptor(array_im, landmarks)
                    faceArray = np.array(face_descriptor).reshape((1, 128))
                    data = np.concatenate((data, faceArray))
                    label.append(labelName)
                    print(labelName)
                    cnt += 1
                    print(cnt)

    # 保存人脸合成的矩阵到本地
    data = data[1:, :]
    np.savetxt('StarFaceData.txt', data, fmt='%f')

    labelFile = open('StarLabel.txt', 'w', encoding='utf-8')
    json.dump(label, labelFile, ensure_ascii=False)
    labelFile.close()

def findLabelInDB(face_descriptor, threshod, flag):

    if flag:
        data = np.loadtxt('data/StarFaceData.txt', dtype=np.float32)
        fl = open("data/StarLabel.txt", 'r', encoding='utf-8')
    else:
        data = np.loadtxt('data/LFWFaceData.txt', dtype=np.float32)
        fl = open("data/LFWLabel.txt", 'r', encoding='utf-8')

    faceLabel = json.load(fl)

    temp = face_descriptor - data
    e = np.linalg.norm(temp, axis=1, keepdims=True)
    min_distance = e.min()
    print(min_distance)

    if(min_distance > threshod):
        return 'John Doe'

    index = np.argmin(e)
    return faceLabel[index]

def drawBox(im, location, label, flag):

    if(flag):
        star_path = "../mx/" + label + "/" + label + ".jpg"
        if os.path.exists(star_path):
            (Image.open(star_path)).show()
        else:
            print("no star image")

    draw = ImageDraw.Draw(im)
    ft = ImageFont.truetype("C:/Windows/Fonts/STKAITI.TTF")
    left, top, right, bottom = (
        max(location.left(), 0), max(location.top(), 0), min(location.right(), np.array(im).shape[1]),
        min(location.bottom(), np.array(im).shape[0]))
    draw.rectangle((left, top, right, bottom), outline=(0, 0, 255), width=5)
    text_width, text_height = ft.getsize(label)
    draw.text((left + (right - left - text_width) / 2, bottom + text_height / 2), label, font=ft, fill=(255, 255, 255))
    im.show()

def recognitionByDB(ImgPath, detector, sp, facerec, threshold, flag):
    im = Image.open(ImgPath)
    im = im.convert('RGB')
    array_im = np.array(im)
    im.show()
    face_locations = detector(array_im, upsample_num_times=1)

    if not face_locations:
        print("no face detected")

    for l in face_locations:
        landmarks = sp(array_im, l)

        # 4 - computer feature vectors
        face_descriptor = facerec.compute_face_descriptor(array_im, landmarks)

        # 5 - compute distance
        face_descriptor = np.array(face_descriptor).reshape((1, 128))
        label = findLabelInDB(face_descriptor, threshold, flag)
        print(label)

        drawBox(im, l, label, flag)

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
    with open('models/star_trained_knn_model.clf', 'rb') as f:
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
        drawBox(im, l, label_name, 0)

def onlineRecognize(img, detector,sp, facerec):
    # dlib检测器
    faces = detector(img, upsample_num_times = 1)
    for face in faces:
        landmarks = sp(img, face)
        face_descriptor = facerec.compute_face_descriptor(img, landmarks)
        face_descriptor = np.array(face_descriptor).reshape((1, 128))
        label = findLabelInDB(face_descriptor, 0.6, 1)
        print(label)

        drawBox(Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB)), face, label, 1)

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
