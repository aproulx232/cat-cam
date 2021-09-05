# Web streaming example
# Source code from the official PiCamera package
# http://picamera.readthedocs.io/en/latest/recipes2.html#web-streaming
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import re
import time

import numpy as np
import picamera

from PIL import Image, ImageDraw
from tflite_runtime.interpreter import Interpreter

import io
import picamera
import logging
import socketserver
from threading import Condition
from http import server

from pygame import mixer

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


def load_labels(path):
    """Loads the labels file. Supports files with or without index numbers."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r"[:\s]+", content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
    return labels


def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]["index"]
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details["index"]))
    return tensor


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    # Get all output details
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                "bounding_box": boxes[i],
                "class_id": classes[i],
                "score": scores[i],
            }
            results.append(result)
    return results


def annotate_objects(image, results, labels):
    """Draws the bounding box and label for each object in the results."""
    for obj in results:
        # Convert the bounding box figures from relative coordinates
        # to absolute coordinates based on the original resolution
        ymin, xmin, ymax, xmax = obj["bounding_box"]
        xmin = int(xmin * CAMERA_WIDTH)
        xmax = int(xmax * CAMERA_WIDTH)
        ymin = int(ymin * CAMERA_HEIGHT)
        ymax = int(ymax * CAMERA_HEIGHT)

        draw = ImageDraw.Draw(image)
        draw.rectangle([xmin, ymin, xmax, ymax], outline="Red")
        draw.text(
            [xmin, ymin],
            "%s\n%.2f" % (labels[obj["class_id"]], obj["score"]),
            fill="Red",
        )

def play_sound():
    sound = mixer.Sound('applause-1.wav')
    sound.play()

def init():
    mixer.init()


PAGE = """\
<html>
<head>
<title>Raspberry Pi - Surveillance Camera</title>
</head>
<body>
<center><h1>Raspberry Pi - Surveillance Camera</h1></center>
<center><img src="stream.mjpg" width="640" height="480"></center>
</body>
</html>
"""


class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()

    def write(self, buf):
        if buf.startswith(b"\xff\xd8"):
            # New frame, copy the existing buffer's content and notify all
            # clients it's available
            # self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.condition.notify_all()
            self.buffer.seek(0)
        return self.buffer.write(buf)


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(301)
            self.send_header("Location", "/index.html")
            self.end_headers()
        elif self.path == "/index.html":
            content = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", 0)
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=FRAME"
            )
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame

                        image = (
                            Image.open(output.buffer)
                            .convert("RGB")
                            .resize((input_width, input_height), Image.ANTIALIAS)
                        )

                        results = detect_objects(interpreter, image, args.threshold)
                        if next((item for item in results if labels[item["class_id"]] == "person"), False):
                            play_sound()
                        annotate_objects(image, results, labels)
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format="jpeg")

                        output.buffer.truncate()
                        frame = img_byte_arr.getvalue()
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except Exception as e:
                logging.warning(
                    "Removed streaming client %s: %s", self.client_address, str(e)
                )
        else:
            self.send_error(404)
            self.end_headers()


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


# main
init()
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", help="File path of .tflite file.", required=True)
parser.add_argument("--labels", help="File path of labels file.", required=True)
parser.add_argument(
    "--threshold",
    help="Score threshold for detected objects.",
    required=False,
    type=float,
    default=0.6,
)
args = parser.parse_args()

labels = load_labels(args.labels)
interpreter = Interpreter(args.model)
interpreter.allocate_tensors()
_, input_height, input_width, _ = interpreter.get_input_details()[0]["shape"]

with picamera.PiCamera(resolution=(CAMERA_WIDTH, CAMERA_HEIGHT), framerate=5) as camera:
    output = StreamingOutput()
    
    camera.start_recording(output, format="mjpeg")
    try:
        address = ("", 8000)
        server = StreamingServer(address, StreamingHandler)
        server.serve_forever()
    finally:
        camera.stop_recording()
