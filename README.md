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
