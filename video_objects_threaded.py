#! /usr/bin/env python3

# Copyright(c) 2017-2018 Intel Corporation.
# License: MIT See LICENSE file in root directory.


from mvnc import mvncapi as mvnc
from video_processor import VideoProcessor
from ssd_mobilenet_processor import SsdMobileNetProcessor
import cv2
import numpy
import time
import os
import sys
from sys import argv
import os.path

# only accept classifications with 1 in the class id index.
# default is to accept all object clasifications.
# for example if object_classifications_mask[1] == 0 then
#    will ignore aeroplanes
object_classifications_mask = [1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1]

NETWORK_GRAPH_FILENAME = "./graph"

# the minimal score for a box to be shown
DEFAULT_INIT_MIN_SCORE = 60
min_score_percent = DEFAULT_INIT_MIN_SCORE

# for title bar of GUI window
cv_window_name = 'video_objects_threaded - SSD_MobileNet'

# the SsdMobileNetProcessor
obj_detector_proc = None

video_proc = None

# read video files from this directory
input_video_path = '.'

# the resize_window arg will modify these if its specified on the commandline
resize_output = False
resize_output_width = 0
resize_output_height = 0


def handle_keys(raw_key:int, obj_detector_proc:SsdMobileNetProcessor):
    """Handles key presses by adjusting global thresholds etc.
    :param raw_key: is the return value from cv2.waitkey
    :param obj_detector_proc: the object detector in use.
    :return: False if program should end, or True if should continue
    """
    global min_score_percent
    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
        return False
    elif (ascii_code == ord('B')):
        min_score_percent = obj_detector_proc.get_box_probability_threshold() * 100.0 + 5
        if (min_score_percent > 100.0): min_score_percent = 100.0
        obj_detector_proc.set_box_probability_threshold(min_score_percent/100.0)
        print('New minimum box percentage: ' + str(min_score_percent) + '%')
    elif (ascii_code == ord('b')):
        min_score_percent = obj_detector_proc.get_box_probability_threshold() * 100.0 - 5
        if (min_score_percent < 0.0): min_score_percent = 0.0
        obj_detector_proc.set_box_probability_threshold(min_score_percent/100.0)
        print('New minimum box percentage: ' + str(min_score_percent) + '%')

    return True



def overlay_on_image(display_image:numpy.ndarray, object_info_list:list):
    """Overlays the boxes and labels onto the display image.
    :param display_image: the image on which to overlay the boxes/labels
    :param object_info_list: is a list of lists which have 6 values each
           these are the 6 values:
           [0] string that is network classification ie 'cat', or 'chair' etc
           [1] float value for box upper left X
           [2] float value for box upper left Y
           [3] float value for box lower right X
           [4] float value for box lower right Y
           [5] float value that is the probability 0.0 -1.0 for the network classification.
    :return: None
    """
    source_image_width = display_image.shape[1]
    source_image_height = display_image.shape[0]

    for one_object in object_info_list:
        percentage = int(one_object[5] * 100)

        label_text = one_object[0] + " (" + str(percentage) + "%)"
        box_left =  int(one_object[1])  # int(object_info[base_index + 3] * source_image_width)
        box_top = int(one_object[2]) # int(object_info[base_index + 4] * source_image_height)
        box_right = int(one_object[3]) # int(object_info[base_index + 5] * source_image_width)
        box_bottom = int(one_object[4])# int(object_info[base_index + 6] * source_image_height)

        box_color = (255, 128, 0)  # box color
        box_thickness = 2
        cv2.rectangle(display_image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

        scale_max = (100.0 - min_score_percent)
        scaled_prob = (percentage - min_score_percent)
        scale = scaled_prob / scale_max

        # draw the classification label string just above and to the left of the rectangle
        label_background_color = (0, int(scale * 175), 75)
        label_text_color = (255, 255, 255)  # white text

        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_left = box_left
        label_top = box_top - label_size[1]
        if (label_top < 1):
            label_top = 1
        label_right = label_left + label_size[0]
        label_bottom = label_top + label_size[1]
        cv2.rectangle(display_image, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1),
                      label_background_color, -1)

        # label text above the box
        cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)


def handle_args():
    """Reads the commandline args and adjusts initial values of globals values to match

    :return: False if there was an error with the args, or True if args processed ok.
    """
    global resize_output, resize_output_width, resize_output_height, min_score_percent, object_classifications_mask

    labels = SsdMobileNetProcessor.get_classification_labels()

    for an_arg in argv:
        if (an_arg == argv[0]):
            continue

        elif (str(an_arg).lower() == 'help'):
            return False

        elif (str(an_arg).lower().startswith('exclude_classes=')):
            try:
                arg, val = str(an_arg).split('=', 1)
                exclude_list = str(val).split(',')
                for exclude_id_str in exclude_list:
                    exclude_id = int(exclude_id_str)
                    if (exclude_id < 0 or exclude_id>len(labels)):
                        print("invalid exclude_classes= parameter")
                        return False
                    print("Excluding class ID " + str(exclude_id) + " : " + labels[exclude_id])
                    object_classifications_mask[int(exclude_id)] = 0
            except:
                print('Error with exclude_classes argument. ')
                return False;

        elif (str(an_arg).lower().startswith('init_min_score=')):
            try:
                arg, val = str(an_arg).split('=', 1)
                init_min_score_str = val
                init_min_score = int(init_min_score_str)
                if (init_min_score < 0 or init_min_score > 100):
                    print('Error with init_min_score argument.  It must be between 0-100')
                    return False
                min_score_percent = init_min_score
                print ('Initial Minimum Score: ' + str(min_score_percent) + ' %')
            except:
                print('Error with init_min_score argument.  It must be between 0-100')
                return False;

        elif (str(an_arg).lower().startswith('resize_window=')):
            try:
                arg, val = str(an_arg).split('=', 1)
                width_height = str(val).split('x', 1)
                resize_output_width = int(width_height[0])
                resize_output_height = int(width_height[1])
                resize_output = True
                print ('GUI window resize now on: \n  width = ' +
                       str(resize_output_width) +
                       '\n  height = ' + str(resize_output_height))
            except:
                print('Error with resize_window argument: "' + an_arg + '"')
                return False
        else:
            return False

    return True


def print_usage():
    """Prints usage information for the program.

    :return: None
    """
    labels = SsdMobileNetProcessor.get_classification_labels()

    print('\nusage: ')
    print('python3 run_video.py [help][resize_window=<width>x<height>]')
    print('')
    print('options:')
    print('  help - prints this message')
    print('  resize_window - resizes the GUI window to specified dimensions')
    print('                  must be formated similar to resize_window=1280x720')
    print('                  Default isto not resize, use size of video frames.')
    print('  init_min_score - set the minimum score for a box to be recognized')
    print('                  must be a number between 0 and 100 inclusive.')
    print('                  Default is: ' + str(DEFAULT_INIT_MIN_SCORE))

    print('  exclude_classes - comma separated list of object class IDs to exclude from following:')
    index = 0
    for oneLabel in labels:
        print("                 class ID " + str(index) + ": " + oneLabel)
        index += 1
    print('            must be a number between 0 and ' + str(len(labels)-1) + ' inclusive.')
    print('            Default is to exclude none.')

    print('')
    print('Example: ')
    print('python3 run_video.py resize_window=1920x1080 init_min_score=50 exclude_classes=5,11')


def main():
    """Main function for the program.  Everything starts here.

    :return: None
    """
    global resize_output, resize_output_width, resize_output_height, \
           obj_detector_proc, resize_output, resize_output_width, resize_output_height, video_proc

    if (not handle_args()):
        print_usage()
        return 1

    # get list of all the .mp4 files in the image directory
#    input_video_filename_list = os.listdir(input_video_path)
#    input_video_filename_list = [i for i in input_video_filename_list if i.endswith('.mp4')]
#    if (len(input_video_filename_list) < 1):
#        # no images to show
#        print('No video (.mp4) files found')
#        return 1

    # Set logging level to only log errors
    mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, 3)

    devices = mvnc.enumerate_devices()
    if len(devices) < 1:
        print('No NCS device detected.')
        print('Insert device and try again!')
        return 1

    # Pick the first stick to run the network
    # use the first NCS device that opens for the object detection.
    # open as many devices as we find
    dev_count = 0
    ncs_devices = []
    obj_detectors = []
    # open as many devices as detected
    for one_device in devices:
        print('one device ', one_device, 'dev_count ', dev_count, 'devices ', devices) 
        obj_detect_dev = mvnc.Device(one_device)
        status = obj_detect_dev.open()
        ncs_devices.append(obj_detect_dev)
        obj_detector_proc = SsdMobileNetProcessor(NETWORK_GRAPH_FILENAME, ncs_devices[dev_count],                            inital_box_prob_thresh=min_score_percent/100.0,
                            classification_mask=object_classifications_mask)
        obj_detectors.append(obj_detector_proc)
        print("opened device " + str(dev_count), 'status ', status)
        dev_count += 1

    print('ncs_devices', ncs_devices)

    cv2.namedWindow(cv_window_name)
    cv2.moveWindow(cv_window_name, 10,  10)
    cv2.waitKey(1)


    exit_app = False

    # output file 
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    #fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    #fourcc = cv2.VideoWriter_fourcc()
    filenum = 1
    # keep the number of video files to a reasonable number in the current directory
    while (filenum < 20):
        doesexist = os.path.isfile("output" + str(filenum) + ".avi")
        if (doesexist == False):
            out_filename = "output" + str(filenum) + ".avi"
            break
        filenum += 1

    print("Using output file name " + out_filename)

    #outfile = cv2.VideoWriter(out_filename, fourcc, 11.0, (640,480))
    outfile = cv2.VideoWriter(out_filename, fourcc, 11.0, (640,480))

    video_file = 0
    # create the video device
    video_device = cv2.VideoCapture(video_file)

    if ((video_device == None) or (not video_device.isOpened())):
        print('\n\n')
        print('Error - could not open video device.')
        print('If you installed python opencv via pip or pip3 you')
        print('need to uninstall it and install from source with -D WITH_V4L=ON')
        print('Use the provided script: install-opencv-from_source.sh')
        print('\n\n')
        return

    # Request the dimensions
    request_video_width = 640
    request_video_height = 480
    video_device.set(cv2.CAP_PROP_FRAME_WIDTH, request_video_width)
    video_device.set(cv2.CAP_PROP_FRAME_HEIGHT, request_video_height)

    # save the actual dimensions
    actual_video_width = video_device.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_video_height = video_device.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print('actual video resolution: ' + str(actual_video_width) + ' x ' + str(actual_video_height))


    while (True):
        # video processor that will put video frames images on the object detector's input FIFO queue

        video_proc1 = VideoProcessor(network_processor = obj_detectors[0]) # obj_detector_proc)
        # video_proc.start_processing()
        if dev_count > 1:
            video_proc2 = VideoProcessor(network_processor = obj_detectors[1]) # obj_detector_proc)

        frame_count = 0
        start_time = time.time()
        end_time = start_time

        while(True):
            # Read from the video file
            ret_val, input_image = video_device.read()
            obj_detectors[0].start_aysnc_inference(input_image)
            
            try:
                (filtered_objs, display_image) = obj_detectors[0].get_async_inference_result()
            except :
                print("exception caught in main")
                raise

            # check if the window is visible, this means the user hasn't closed
            # the window via the X button
            prop_val = cv2.getWindowProperty(cv_window_name, cv2.WND_PROP_ASPECT_RATIO)
            if (prop_val < 0.0):
                end_time = time.time()
                video_proc.stop_processing()
                exit_app = True
                break

            overlay_on_image(display_image, filtered_objs)

            if (resize_output):
                display_image = cv2.resize(display_image,
                                           (resize_output_width, resize_output_height),
                                           cv2.INTER_LINEAR)
            cv2.imshow(cv_window_name, display_image)

            outfile.write(display_image)

            raw_key = cv2.waitKey(1)
            if (raw_key != -1):
                if (handle_keys(raw_key, obj_detector_proc) == False):
                    end_time = time.time()
                    exit_app = True
#                    video_proc.stop_processing()
                    break

            frame_count += 1

#            if (obj_detectors[0].is_input_queue_empty()):
#                end_time = time.time()
#                print('Neural Network Processor has nothing to process, assuming video is finished.')
#                break

        frames_per_second = frame_count / (end_time - start_time)
        print('Frames per Second: ' + str(frames_per_second))

        throttling = ncs_devices[0].get_option(mvnc.DeviceOption.RO_THERMAL_THROTTLING_LEVEL)
        if (throttling > 0):
            print("\nDevice is throttling, level is: " + str(throttling))
            print("Sleeping for a few seconds....")
            cv2.waitKey(2000)

        #video_proc.stop_processing()
        cv2.waitKey(1)

#        video_proc.cleanup()

        if (exit_app):
            break
    #if (exit_app):
    #    break

    # Clean up the graph and the device
    obj_detectors[0].cleanup()
    ncs_devices[0].close()
    ncs_devices[0].destroy()

    cv2.destroyAllWindows()
    outfile.release()


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())
