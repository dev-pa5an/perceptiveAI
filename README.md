
# Object depth estimation of color Images using a depth camera and a image semantic segmentation model in Computer Vision with minimum latency.

The key twist is using a specific type of neural network called a "semantic segmentation network" to improve the accuracy of the distance estimation. This network learns to categorize different parts of the image (like walls, furniture, people) and uses that information to estimate the average depth to those objects.
This method is useful for a robot to enhance its perception of the environment while it is moving, getting depth estimations for particular objects in the environment. When the environment is dynamic and complex, indoor navigation tasks can also utilize this method.



## Features

- YOLOV8 Image Semantic Segmentation Model

  This model is trained using a custom dataset of over 400 images. Ultralytics yolov8's semantic segmentation model parameters are used to train the model. The model itself reached up to 100% confidence level for the validation dataset but only 93% in practice.
  
  <img src="https://github.com/dev-pa5an/perceptiveAI/blob/main/Images/val_batch0_pred.jpg" width="500" height="300" />

  ps: You can use your own semantic segmentation model in this application.
  
- Microsoft KinectV2 with python

  Microsoft KinectV2 has both RGB and depth camera. The modified version of pykinect2 library is used to recieve both RGB and depth frames simultaneously using a python program. Please use the modified version of pykinect2 otherwise the program will not run as exspected. Modified version of pykinect2 can be found here. (https://github.com/dev-pa5an/pykinect.git)
  
- Depth Estimation Algorithm

  When mapping the pixel coordinates of the image frame to the depth frame, the frame size difference results in the inclusion of depth values from areas outside the identified object in the depth array. This error can be clearly identified in the following figure.
  
   <img src="https://github.com/dev-pa5an/perceptiveAI/blob/main/Images/Difference between the image frame and the depth frame after masking the objec.png" width="500" height="300" />


## Run Locally


```bash
  git clone https://github.com/dev-pa5an/perceptiveAI
```
```bash
  python perceptiveEye.py
```

