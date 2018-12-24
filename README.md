# Face_Recognize
face recognition based on [dlib](http://dlib.net)

We provide a simple way that lets you do face recognition on a folder of images from the command line!

dataset：[Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/)

## Installation

download the project.

#### Requirements

  * Python 3.3+
  * macOS, Windows, Linux

#### Prerequisite

[cmake](https://cmake.org/download/)，[dlib](http://dlib.net)，[boost](http://www.boost.org/users/download/)

Environment Guide:

[macOS](https://blog.csdn.net/gaoyueace/article/details/79198023)，[Window](https://blog.csdn.net/qq_35044509/article/details/78882316)，[Ubuntu](https://www.cnblogs.com/darkknightzh/p/5652791.html)

```
pip install -r requirements.txt
```

#### Windows tips:

Download and install scipy and numpy+mkl (must be mkl version) packages from [this link](https://www.lfd.uci.edu/~gohlke/pythonlibs/). Remember to grab correct version based on your current Python version.

## Usage

### bulid DataBase for recognization

* using DB
```
python main.py -b DB --image_file lfw/
```

* using Knn
```
python main.py -b Knn --image_file lfw/
```

#### Recognize people in a image

* using DB
```
python main.py -r DB --image_file img_file_path
```

* using Knn
```
python main.py -r Knn --image_file img_file_path
```

#### Detecting people online
```
python main.py --online
```

#### Makeup people in a image
```
python main.py --makeup
```

#### Which star you are alike most




