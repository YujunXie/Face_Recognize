from faceCharm import *

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')
threshold = 0.6

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--bulider", type=str, choices=["DB", "knn"], help="choose the kind of classfier")
    parser.add_argument("-r", "--recognize", type=str, choices=["DB", "knn"], help="recognize faces in one image")
    parser.add_argument("--image_file", type=str, help="choose Image to recognize", default='lfw/Aaron_Guiel/Aaron_Guiel_0001.jpg')
    parser.add_argument("--n_neighbors", type=int, help="choose the number of neighbors", default=2)
    parser.add_argument("--online", help="online recognize faces through camera", action="store_true")
    parser.add_argument("--makeup", help="makeup for people in images", action="store_true")
    parser.add_argument("--star", help="find the star you are alike most", action="store_true")
    args = parser.parse_args()

    if args.bulider == "DB":
        print("Start Buliding...")
        bulid_DB(args.image_file, detector, sp, facerec)
        print("Building complete!")
    if args.bulider == "knn":
        print("Start Training...")
        classifier = trainKnn(args.image_file, args.n_neighbors, detector, sp, facerec)
        print("Training complete!")

    if args.recognize == "DB":
        recognitionByDB(args.image_file, detector, sp, facerec, threshold, 0)
    if args.recognize == "knn":
        recognitionByKnn(args.image_file, detector, sp, facerec, threshold)

    if args.online:
        cap = cv.VideoCapture(0)
        while 1:
            ret, img = cap.read()
            onlineRecognize(img, detector, sp, facerec)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()

    if args.makeup:
        Makeup(args.image_file, detector, sp)

    if args.star:
        recognitionByDB(args.image_file, detector, sp, facerec, threshold, 1)
