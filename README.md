# Cat Detecting Detergent

This project uses a raspberry pi to detect when a cat is in a restricted area and deters them from entering this area.

It is based off of the [Raspberry Pi TensorFlow Lite Object Detection Example](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/raspberry_pi) and the [Video Streaming with Raspberry Pi Camera Example](https://randomnerdtutorials.com/video-streaming-with-raspberry-pi-camera/)

## Set up your hardware

Before you begin, you need to [set up your Raspberry Pi](
https://projects.raspberrypi.org/en/projects/raspberry-pi-setting-up) with
Raspberry Pi OS (preferably updated to Buster).

You also need to [connect and configure the Pi Camera](
https://www.raspberrypi.org/documentation/configuration/camera.md).


## Install the TensorFlow Lite runtime

In this project, all you need from the TensorFlow Lite API is the `Interpreter`
class. So instead of installing the large `tensorflow` package, we're using the
much smaller `tflite_runtime` package.

To install this on your Raspberry Pi, follow the instructions in the
[Python quickstart](https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python).
Return here after you perform the `apt-get install` command.


## Download the example files

First, clone this Git repo onto your Raspberry Pi like this:

```
git clone https://github.com/aproulx232/cat-cam.git
```

Then use the script to install a couple Python packages, and
download the MobileNet model and labels file:

```
# The script takes an argument specifying where you want to save the model files
bash download.sh /tmp
```


## Run the example

```
python3 stream_object_detection.py \
  --model /tmp/detect.tflite \
  --labels /tmp/coco_labels.txt
```

The camera feed with object detection bounding boxes should appear at the address 
```
http://<your local network ip address>:8000/index.html
```

