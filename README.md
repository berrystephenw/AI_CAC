# Introduction
This project uses SSD MobileNet to do object recognition and classification for a live web camera.

The provided Makefile does the following:
1. Builds caffe ssd mobilenet graph file from the caffe/SSD_MobileNet directory in the repository.
2. Copies the built NCS graph file from the SSD_MobileNet directory to the project base directory
3. Runs the provided video_objects_threaded.py program which creates a GUI window that shows the video stream along with labels and boxes around the identified objects.
4. The resulting file with labels and boxes are stored in the current directory as outputX.mp4 where X is a number from 1-20

# Prerequisites
This program requires:
- 1 NCS device
- a webcam
- NCSDK 2.04 or greater
- OpenCV 3.3 with Video for Linux (V4L) support and associated Python bindings*.

*It may run with older versions but you may see some glitches such as the GUI Window not closing when you click the X in the title bar, and other key binding issues.


Note: The OpenCV version installed with some earlier versions of the ncsdk do <strong>not</strong> provide V4L support.  To install a compatible version of OpenCV for this application and remove older versions installed by the ncsdk you can run the following command from the app's base directory:
```
   make opencv
```   
Note: All development and testing has been done on Ubuntu 16.04 on an x86-64 machine.


This applications is similar to the apps/video_objects_threaded application in the ncappzoo, but it uses live video for processing SSD MobieNet inferences. The intent is to increase the frame rate further by interleaving the inference calls to two (or more) neural net sticks.

# Makefile
Provided Makefile has various targets that help with the above mentioned tasks.

## make help
Shows available targets.

## make all
Builds and/or gathers all the required files needed to run the application except building and installing opencv (this must be done as a separate step with 'make opencv'.)

## make opencv
Removes the version of OpenCV that was installed with the NCSDK and builds and installs a compatible version of OpenCV 3.3 for this app. This will take a while to finish. Once you have done this on your system you shouldn't need to do it again.

## make run
Runs the provided python program which shows the video stream along with the object boxes and classifications.

## make clean
Removes all the temporary files that are created by the Makefile
