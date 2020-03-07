# Garbage-Classification-for-safety

## Motivation
The motivation started by walking down my street abd seeing a huge garbage bin at the corner where all the houses in the street would dump their daily garbage. People would not try to seperate the recyclable and organic wastes but throw every type of trash together.

Every morning, as I am about to leave for my classes, I would see a garbage pick-up truck swing by and take up all the garbage. Even the garbage man does not know how to seperate all the trash.

So, I researched and found out that most of the garbages are either cleaned by hand, dump together in garbage dump fill or burnt together. In these cases, even material like plastic mix with organic dump and burnt. This not only affects the environment but also millions of dollars of money are lost because of not recycling such garbages.

This is the motivation behind the project, an humble but effective idea to segregate trash by different categories and thus reduce manpower and effects on environment and save up money by recycling.

![alt text](https://github.com/Logeswaran123/Garbage-Classification-for-safety/blob/master/pictures/garbage.jpg "Garbage disposal zone")

## How it works

The application is integrated to a camera or you can add a captured video/image. The model detects the type of garbage in the image and outputs the message indicating the type of garbage to the user. This can be integrated into an automated machine to seperate trash based on it's type. This project focuses on the software implementation and not the hardware.

The project consists of two models with each of its own purpose.

### Model 1 (Organic/Recyclable)

The **First model** is Organic/Recyclable classification model. The objective of the model is to classify the trash in the image into organic or recyclable.

**Dataset:** The dataset for this model can be found in [Link](https://www.kaggle.com/techsash/waste-classification-data).

**Preprocessing:**
1. Take all images from train and test directory and put in a single directory.
2. Create a list for the organic/recyclable class corresponding to the image by splitting at '\_' and appending only first element which can be 'O' or 'R'.
```python
gender = [i.split('_')[0] for i in files]
classes = []
for i in gender:
    if i == 'O':
        classes.append(0)
    else:
        classes.append(1)
```
**Model description:**

The model used here is a reduced version on VGG network with height=96, width=96, depth=3 and class=2 (organic/recyclable).

Find the model in [model_O_R](https://github.com/Logeswaran123/Garbage-Classification-for-safety/tree/master/model_O_R) directory.
| Train Accuracy   | Train Loss   | Validation Accuracy| Validation Loss |
| -----------|:------:|:-----|:------ |
| 0.92     | 0.23 | 0.900 | 0.248 |

![alt text](https://github.com/Logeswaran123/Garbage-Classification-for-safety/blob/master/pictures/plot.JPG "history plot")

### Model 2 (Six type classification)

The **Second model** is a six type garbage classification model. The different classes into which the input image is classified are 1. Cardboard 2. Glass 3. Metal 4. Paper 5. Plastic 6. Trash.

**Dataset:** The dataset for this model can be found in [Link](https://www.kaggle.com/asdasdasasdas/garbage-classification).

In this, the ImageDataGenerator keras.preprocessing.image is used to create the train set and validation set.

```python
train = ImageDataGenerator(horizontal_flip = True, vertical_flip = True,
                         validation_split = 0.1, rescale = 1./255,
                         shear_range = 0.2, zoom_range = 0.2,
                         width_shift_range = 0.1, height_shift_range = 0.1,)

test = ImageDataGenerator(rescale = 1/255, validation_split = 0.1)

train_generator = train.flow_from_directory(dir_path,
                                          target_size = (300,300),
                                          batch_size = 32,
                                          class_mode = 'categorical',
                                          subset = 'training')

valid_generator = test.flow_from_directory(dir_path,
                                        target_size = (300,300),
                                        batch_size = 32,
                                        class_mode = 'categorical',
                                        subset = 'validation')
```

The train and validation split is 90:1. After that, the base model is InceptionV3 with pretrained weights. From this weights, current classification model is trained. Due to low number of train images the Accuracy of the model is low.


| Train Accuracy   | Train Loss   | Validation Accuracy| Validation Loss |
| -----------|:------:|:-----|:------ |
| 0.8226     | 0.5107 | 0.6941 | 0.6847 |

![alt text](https://github.com/Logeswaran123/Garbage-Classification-for-safety/blob/master/pictures/plot2.JPG "history plot")

## Prerequisites

### Hardware

Requirements based on [OpenVino toolkit](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_windows.html).

* 6th-10th Generation Intel® Core™ processors
* Intel® Xeon® v5 family
* Intel® Xeon® v6 family
* Intel® Pentium® processor N4200/5, N3350/5, N3450/5 with Intel® HD Graphics
* Intel® Movidius™ Neural Compute Stick
* Intel® Neural Compute Stick 2
* Intel® Vision Accelerator Design with Intel® Movidius™ VPUs

Refer the [Link](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_windows.html) for detailed installation.

### Packages/Libraries

Make sure the following list are installed for easier running of application. Latest updated version required.

```
keras
tensorflow
OpenCV
pickle
matplotlib
itertools
sklearn
random
os
time
glob
```

## Installing a WEB server, a MQTT broker and a Dashboard

**Installing Node-RED**

- ***Install the latest version of node.js***

  ``` python sudo apt-get install nodejs ```

- ***Install npm***

  ```python sudo apt-get install npm ```

- ***Install Node-RED with npm***
 
  ```python sudo npm install -g --unsafe-perm node-red```

- ***Start the Node-RED service***
  
  ```python sudo node-red ```

The server starts running on port 1880.
To enter the dashboard enter the url 127.0.0.1:1880/ or localhost:1880/ in the web browser.


**Install the Aedes broker and the Dashboard**

  - ***Using the Node Red GUI:***

  ![alt text](https://github.com/joagovi/Garbage-Classification-for-safety/blob/master/pictures/menu_node.png "menu node-red")

1. Enter to the main menu of node-red.
2. Select "Manage palette"
3. Select "Install"
4. Enter **"node-red-dashboard"** in the search field
5. Click on the **"install"** button
6. When the installation is finished, click "Done."
7. Press F5 to reload.
8. You will find the **"Dashboard"** button on the right side.


- **Similarly install the Aedes broker node: “node-red-contrib-aedes”**

***Start Node-RED with the pre-existing file***

  1. Use the **.node-red** folder: ``` cd <PATH>/.node-red/ ```
Where .node-red is the folder that contains the application files

  2. Start Node-RED with the file containing the flows:```python sudo node-red flows_ip-172-31-87-186.json```

**Configuring node-red flow**

- ***Set up the Broker***

  Select any of the subscribers and then edit the server with the ip of the localhost. Replace 52.90.89.156 by 127.0.0.1.

  ![alt text](https://github.com/joagovi/Garbage-Classification-for-safety/blob/master/pictures/edit_broker.png "select broker")

  ![alt text](https://github.com/joagovi/Garbage-Classification-for-safety/blob/master/pictures/edit_broker_2.png "edit broker")

  Finally change the source ip of the template to localhost (127.0.0.1)

  ![alt text](https://github.com/joagovi/Garbage-Classification-for-safety/blob/master/pictures/template.png "select template")

  ![alt text](https://github.com/joagovi/Garbage-Classification-for-safety/blob/master/pictures/edit_template.png "edit template")


## FFmpeg service

**Install the FFmpeg service**

``` python
sudo apt update
sudo apt install ffmpeg
```

**Starting the FFmpeg service**

The server.conf file contains the information where the video is fed: fac.ffm and also this file contain in which url the video is played: facstream.mjpeg. In addition other details such as allowed IP addresses or video size: VideoSize.

**Start the FFmpeg service with a configuration file**

```
python sudo ffserver -f ffmpeg/server.conf
```

## How to run
From your workspace environment, run the following command.

**To run trash classification application**

```python
python trashclass.py -m Six Type Classification/tf_classification_model.xml -i project_trash.mp4 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 512x384 -i - http://52.90.89.176:8082/fac2.ffm -framerate 24
```
**Example**

![alt text](https://github.com/Logeswaran123/Garbage-Classification-for-safety/blob/master/pictures/example1.png "cmd example")

**To run organic/recyclable application**

```python
python organic.py -m Organic-Recyclable/tf_model.xml -i project.mp4 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 224x224 -i - http://52.90.89.176:8082/fac2.ffm -framerate 24
```
**Example**

![alt text](https://github.com/Logeswaran123/Garbage-Classification-for-safety/blob/master/pictures/example2.png "cmd example")

Note: Specify the correct path to the files beforing running the app.

**Web Page:** [Link](http://52.90.89.176:1880/ui/#!/0?socketid=yvyjSCI4CHhhg41oAAAA)

## Demo Output

### For Organic/Recyclable

![alt text](https://github.com/Logeswaran123/Garbage-Classification-for-safety/blob/master/pictures/out1.png "organic")

![alt text](https://github.com/Logeswaran123/Garbage-Classification-for-safety/blob/master/pictures/out2.png "recyclable")

### For Classification

![alt text](https://github.com/Logeswaran123/Garbage-Classification-for-safety/blob/master/pictures/out3.png "plastic")

**Project team**
* [Logeswaran Sivakumar](https://github.com/Logeswaran123)
* [Akash Singh](https://github.com/Akash4097)
* [Joagovi](https://github.com/joagovi)

Project created as a part of the Udacity OpenVino Scholarship for the showcase [OpenVino Udacity](https://www.udacity.com/scholarships/intel-edge-ai-scholarship).

Happy Learning! :-)
