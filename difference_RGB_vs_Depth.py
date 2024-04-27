import cv2
import numpy as np
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
from ultralytics import YOLO

class KinectHandler(object):
    def __init__(self):
        self.kinectd = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)
        self.depth_width, self.depth_height = self.kinectd.depth_frame_desc.Width, self.kinectd.depth_frame_desc.Height
        self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
        self.color_width, self.color_height = self.kinect.color_frame_desc.Width, self.kinect.color_frame_desc.Height

    def get_depth_frame(self):
        if self.kinectd.has_new_depth_frame():
            depth_frame = self.kinectd.get_last_depth_frame()
            depth_frame = depth_frame.reshape((self.depth_height, self.depth_width)).astype(np.uint16)
            return depth_frame
        return None

    def get_color_frame(self):
        if self.kinect.has_new_color_frame():
            color_frame = self.kinect.get_last_color_frame()
            color_frame = color_frame.reshape((self.color_height, self.color_width, 4))
            # Keep only RGB channels
            color_frame = color_frame[:, :, :3]
            return color_frame
        return None

    def close(self):
        self.kinect.close()

def main():
    model_path = 'last.pt'
    model = YOLO(model_path)

    kinect = KinectHandler()

    while True:
        depth_frame = kinect.get_depth_frame()

        if depth_frame is not None:
            cv2.imshow('Depth Frame', depth_frame.astype(np.uint8))

            color_frame = kinect.get_color_frame()

            if color_frame is not None:
                H, W , _ = color_frame.shape

                results = model(color_frame, verbose=False)

                for result in results:
                    if result.masks is not None:
                        print("Identified a box with confidence {:.2f}%".format(result.conf * 100))
                        for mask in result.masks.data:
                            mask = mask.numpy().astype(np.uint8) * 255
                            mask = cv2.resize(mask, (W, H))

                            # Color mask pixels in red
                            color_frame[mask > 0] = (0, 0, 255)

                            # Resize mask to match depth frame dimensions
                            mask_resized = cv2.resize(mask, (kinect.depth_width, kinect.depth_height))

                            # Color corresponding depth frame pixels in red
                            depth_frame[mask_resized > 0] = 255

                cv2.imshow('Color Frame', color_frame)
                cv2.imshow('Depth Frame', depth_frame.astype(np.uint8))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    kinect.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()