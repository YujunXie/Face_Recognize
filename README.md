# Face_Recognize
face recognition based on [dlib](http://dlib.net)

We provide a simple way that lets you do face recognition on a folder of images from the command line!

dataset：[Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/)

## Installation

download the project.

download dataset [lfw]((http://vis-www.cs.umass.edu/lfw/) in lfw-folder

if you need the dataset of Asia Stars, please contact me.

#### Requirements

  * Python 3.3+
  * macOS, Windows, Linux

#### Prerequisite

[cmake](https://cmake.org/download/)，[dlib](http://dlib.net)，[boost](http://www.boost.org/users/download/)

Environment Guide:

[macOS](https://blog.csdn.net/gaoyueace/article/details/79198023)，[Window](https://blog.csdn.net/qq_35044509/article/details/78882316)，[Ubuntu](https://www.cnblogs.com/darkknightzh/p/5652791.html)

```bash
pip install -r requirements.txt
```

#### Windows tips:

Download and install scipy and numpy+mkl (must be mkl version) packages from [this link](https://www.lfd.uci.edu/~gohlke/pythonlibs/). Remember to grab correct version based on your current Python version.

## Usage

### bulid DataBase for recognization

* training DB
```bash
python main.py -b DB --image_file lfw/
```
DB file: FaceData.txt，label.txt

* training Knn
```bash
python main.py -b Knn --image_file lfw/
```
Knn model：models/trained_knn_model.clf

#### Recognize people in a image

* testing DB
```bash
python main.py -r DB --image_file img_file_path
```

* testing Knn
```bash
python main.py -r Knn --image_file img_file_path
```
k = 2

threshold = 0.6

![recognize.png](https://i.loli.net/2018/12/24/5c20bb4fa2325.png)

#### Recognizing people online
```bash
python main.py --online
```

![online.png](https://i.loli.net/2018/12/24/5c20bb5026494.png)

#### Makeup people in a image
```bash
python main.py --makeup --image_file img_file_path
```

![makup.png](https://i.loli.net/2018/12/24/5c20bb4e801a4.png)

#### Which star you are alike most
```bash
python main.py --star --image_file img_file_path
```

素人：
![suren.png](https://i.loli.net/2018/12/24/5c20bb4fd6718.png)

明星：
![star.png](https://i.loli.net/2018/12/24/5c20bb4e0ddf3.png)

### testing results on LFW

testing results on lfw with pretrained resnet model on dlib.

![result.png](https://i.loli.net/2018/12/24/5c205099632f2.png)

