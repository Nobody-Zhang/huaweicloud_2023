#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# edited by ZGB
################################################################################

""" Example of deepstream using SSD neural network and parsing SSD's outputs. """
import time
import ctypes
import numpy as np
import os
import pathlib
import sys
import io

sys.path.append("../")
import gi

gi.require_version("Gst", "1.0")
from gi.repository import GObject, Gst, GLib
import platform
import sys

queue1 = None
queue2 = None
fakesink1 = None
fakesink2 = None
blockpad = None
cur_effect = None
next_effect = None
pipeline = None

def is_aarch64():
    return platform.uname()[4] == 'aarch64'


import gi
import sys

gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst


def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream\n")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write("Warning: %s: %s\n" % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write("Error: %s: %s\n" % (err, debug))
        loop.quit()
    return True


sys.path.append('/opt/nvidia/deepstream/deepstream/lib')

import pyds

CLASS_NB = 7
ACCURACY_ALL_CLASS = 0.5
UNTRACKED_OBJECT_ID = 0xffffffffffffffff
IMAGE_HEIGHT = 1080
IMAGE_WIDTH = 1920
OUTPUT_VIDEO_NAME = "./out.mp4"

np.set_printoptions(threshold=np.inf)

INPUT_HEIGHT = 640
INPUT_WIDTH = 640


class BoundingBox:
    def __init__(self, classID, confidence, x1, x2, y1, y2, image_width, image_height):
        self.classID = classID
        self.confidence = confidence
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.u1 = x1 / image_width
        self.u2 = x2 / image_width
        self.v1 = y1 / image_height
        self.v2 = y2 / image_height

    def box(self):
        return self.x1, self.y1, self.x2, self.y2

    def width(self):
        return self.x2 - self.x1

    def height(self):
        return self.y2 - self.y1

    def center_absolute(self):
        return 0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2)

    def center_normalized(self):
        return 0.5 * (self.u1 + self.u2), 0.5 * (self.v1 + self.v2)

    def size_absolute(self):
        return self.x2 - self.x1, self.y2 - self.y1

    def size_normalized(self):
        return self.u2 - self.u1, self.v2 - self.v1


def nms(boxes, box_confidences, nms_threshold=0.5):
    x_coord = boxes[:, 0]
    y_coord = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]

    areas = width * height
    ordered = box_confidences.argsort()[::-1]

    keep = list()
    while ordered.size > 0:
        i = ordered[0]
        keep.append(i)
        xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
        yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
        xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
        yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

        width1 = np.maximum(0.0, xx2 - xx1 + 1)
        height1 = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width1 * height1
        union = (areas[i] + areas[ordered[1:]] - intersection)

        iou = intersection / union

        indexes = np.where(iou <= nms_threshold)[0]
        ordered = ordered[indexes + 1]
    keep = np.array(keep).astype(int)
    return keep


def postprocess(buffer, image_width, image_height, conf_threshold=0.4, nms_threshold=0.5):
    detected_objects = []
    img_scale = [image_width / INPUT_WIDTH, image_height / INPUT_HEIGHT, image_width / INPUT_WIDTH,
                 image_height / INPUT_HEIGHT]
    num_bboxes = int(buffer[0, 0, 0, 0])

    if num_bboxes:
        bboxes = buffer[0, 1: (num_bboxes * 6 + 1), 0, 0].reshape(-1, 6)
        labels = set(bboxes[:, 5].astype(int))

        for label in labels:
            selected_bboxes = bboxes[np.where((bboxes[:, 5] == label) & (bboxes[:, 4] >= conf_threshold))]
            selected_bboxes_keep = selected_bboxes[nms(selected_bboxes[:, :4], selected_bboxes[:, 4], nms_threshold)]
            for idx in range(selected_bboxes_keep.shape[0]):
                box_xy = selected_bboxes_keep[idx, :2]
                box_wh = selected_bboxes_keep[idx, 2:4]
                score = selected_bboxes_keep[idx, 4]

                box_x1y1 = box_xy - (box_wh / 2)
                box_x2y2 = np.minimum(box_xy + (box_wh / 2), [INPUT_WIDTH, INPUT_HEIGHT])
                box = np.concatenate([box_x1y1, box_x2y2])
                box *= img_scale

                if box[0] == box[2]:
                    continue
                if box[1] == box[3]:
                    continue
                detected_objects.append(
                    BoundingBox(label, float(score), box[0], box[2], box[1], box[3], image_height, image_width))

    return detected_objects


def get_label_names_from_file(filepath="./labels.txt"):
    """
        从文件中读取标签名称。
    """
    f = io.open(filepath, "r")
    labels = f.readlines()
    labels = [elm[:-1] for elm in labels]
    f.close()
    return labels


def make_elm_or_print_err(factoryname, name, printedname, detail=""):
    """
        利用一种流媒体处理框架的对象工厂创建对象。
        factoryname: 对象工厂名称
        name: 对象名称 （在管道中唯一标识的名称）
        Creates an element with Gst Element Factory make.
        Return the element  if successfully created, otherwise print
        to stderr and return None.
    """
    print("Creating", printedname)
    elm = Gst.ElementFactory.make(factoryname, name)
    if not elm:
        sys.stderr.write("Unable to create " + printedname + " \n")
        if detail:
            sys.stderr.write(detail)
    return elm


def add_obj_meta_to_frame(bbox, score, label, batch_meta, frame_meta):
    """
    Inserts an object into the metadata
    """
    untracked_obj_id = 0xffffffffffffffff
    # this is a good place to insert objects into the metadata.
    # Here's an example of inserting a single object.
    obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
    # Set bbox properties. These are in input resolution.
    rect_params = obj_meta.rect_params
    rect_params.left = int(bbox[0])
    rect_params.top = int(bbox[1])
    rect_params.width = int((bbox[2] - bbox[0]))
    rect_params.height = int(bbox[3] - bbox[1])
    # logger.info(f"rect_params.left: {rect_params.height}")

    # Semi-transparent yellow backgroud
    rect_params.has_bg_color = 0
    rect_params.bg_color.set(1, 1, 0, 0.4)

    # Red border of width 3
    rect_params.border_width = 3
    rect_params.border_color.set(1, 0, 0, 1)

    # Set object info including class, detection confidence, etc.
    obj_meta.confidence = score
    # obj_meta.class_id = int(label)

    # There is no tracking ID upon detection. The tracker will
    # assign an ID.
    obj_meta.object_id = untracked_obj_id

    # lbl_id = int(label)
    # if lbl_id >= len(label_names):
    #    lbl_id = 0

    # Set the object classification label.
    obj_meta.obj_label = label

    # Set display text for the object.
    txt_params = obj_meta.text_params
    if txt_params.display_text:
        pyds.free_buffer(txt_params.display_text)

    txt_params.x_offset = int(rect_params.left)
    txt_params.y_offset = max(0, int(rect_params.top) - 10)
    txt_params.display_text = (
            label + " " + "{:04.3f}".format(score)
    )
    # Font , font-color and font-size
    txt_params.font_params.font_name = "Serif"
    txt_params.font_params.font_size = 10
    # set(red, green, blue, alpha); set to White
    txt_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

    # Text background color
    txt_params.set_bg_clr = 1
    # set(red, green, blue, alpha); set to Black
    txt_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)

    # Inser the object into current frame meta
    # This object has no parent
    pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)


ct = 0
def queue1_src_pad_buffer_probe(pad, info, u_data):
    global ct
    ct += 1
    if ct % 5 == 0:
        time.sleep(1)
    print("now in queue1_src_pad_buffer_probe", ct)
    return Gst.PadProbeReturn.OK

def queue2_src_pad_buffer_probe(pad, info, u_data):
    # print("now in queue2_src_pad_buffer_probe")
    return Gst.PadProbeReturn.OK


def pgie_src_pad_buffer_probe(pad, info, u_data):
    """
    Get tensor_metadata from triton inference output
    Convert tensor metadata to numpy array
    Postprocessing
    """
    image_width = 1920
    image_height = 1080
    conf_thresh = 0.5
    nms_thresh = 0.5
    label = get_label_names_from_file()
    is_save_output = False
    # get the buffer of info argument
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list

    print(time.time())
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            # logger.info("frame_meta: {}".format(frame_meta))
        except StopIteration:
            break
        iter_obj = frame_meta.obj_meta_list
        print("number_of_counts:", frame_meta.num_obj_meta)
        while iter_obj is not None:
            # 从GList对象中获取当前节点的数据
            obj_meta = pyds.NvDsObjectMeta.cast(iter_obj.data)

            # 读取对象元数据的属性
            object_id = obj_meta.object_id
            class_id = obj_meta.class_id
            left = obj_meta.rect_params.left
            top = obj_meta.rect_params.top
            width = obj_meta.rect_params.width
            height = obj_meta.rect_params.height
            confidence = obj_meta.confidence
            # print("class_id:", class_id)
            # print("left:", left)
            # print("top", top)
            # print("width", width)
            # print("height:", height)
            # print("confidence:", confidence)
            # print(object_id, left, top, width, height, confidence)
            # print(obj_meta)
            # print(class_id)
            try:
                iter_obj = iter_obj.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK

def pad_probe_cb(pad, info, pipeline):
    """
    pgie = pipeline.get_by_name("primary-inference")
    fakesink = pipeline.get_by_name("fakesink")
    nvvidconv = pipeline.get_by_name("convertor")
    # nvosd = pipeline.get_by_name("onscreendisplay")
    # queue = pipeline.get_by_name("queue")
    # nvvidconv2 = pipeline.get_by_name("convertor2")
    # capsfilter = pipeline.get_by_name("capsfilter")
    # encoder = pipeline.get_by_name("encoder")
    # codeparser = pipeline.get_by_name("codeparser")
    # container = pipeline.get_by_name("container")
    # sink = pipeline.get_by_name("sink")

    # 获取pgie到fakesink的帧数
    fakesink_frames = fakesink.get_property("num-buffers")

    # 检查网络连接状态
    is_network_connected = check_network_status()

    global on_line_status
    global countt
    # on_off_line_status = False

    # 根据条件选择分流方案
    if is_network_connected and fakesink is not None:
        # 切换到带网络连接的流，当前流的方案为：pgie -> fakesink
        print("切换到带网络连接的流", countt)
        # pgie.unlink(fakesink)
        pgie.link(nvvidconv)

        # pipeline.add(nvvidconv, nvosd, queue, nvvidconv2, capsfilter, encoder, sink)

    elif not is_network_connected and nvvidconv is not None:
        # 当前没有网络连接，且pipeline在带网络连接的流上，切换到不带网络连接的流
        print("切换到不带网络连接的流", countt)
        # pgie.unlink(nvvidconv)
        pgie.link(fakesink)
        # pipeline.add(fakesink)
    else:
        print("当前流不需要切换", countt)

    # 启动新的流
    pipeline.set_state(Gst.State.PLAYING)
    """
    return Gst.PadProbeReturn.OK
"""
def event_probe_cb(pad, info, loop):
    global cur_effect, pipeline, next_effect
    pad.remove_probe(info.id)
    print('event probe cb')
    print("before changing")
    conv_before = pipeline.get_by_name('buffer')
    conv_after = pipeline.get_by_name('fakesink')

    cur_effect.set_state(Gst.State.NULL)
    pipeline.remove(cur_effect)
    pipeline.add(next_effect)
    conv_before.link(next_effect)
    next_effect.link(conv_after)

    tmp = cur_effect
    cur_effect = next_effect
    next_effect = tmp

    next_effect.set_state(Gst.State.PLAYING)
    print('Changed')

    return Gst.PadProbeReturn.DROP

i = 0

def pad_probe_cb(pad, info, user_data):

    # remove the probe first
    pad.remove_probe(info.id)

    # install new probe for EOS
    srcpad = cur_effect.get_static_pad("src")
    srcpad.add_probe(Gst.PadProbeType.BLOCK | Gst.PadProbeType.EVENT_DOWNSTREAM, event_probe_cb, user_data)
    srcpad.unref()

    # push EOS into the element, the probe will be fired when the
    # EOS leaves the effect and it has thus drained all of its data
    sinkpad = cur_effect.get_static_pad("sink")
    sinkpad.send_event(Gst.Event.new_eos())
    sinkpad.unref()

    return Gst.PadProbeReturn.OK


def timeout_cb(loop):
    global blockpad
    blockpad.add_probe(Gst.PadProbeType.BLOCK_DOWNSTREAM, pad_probe_cb, loop)
    return True
"""

def check_network_status():
    # 检查网络连接状态
    # 返回True或False
    return True

flag = True

def check_network():
    global pipeline
    global flag
    buffer = pipeline.get_by_name("buffer")
    fakesink = pipeline.get_by_name("fakesink")
    if flag:
        # 如果网络已连接，更改pipeline
        buffer.unlink(queue1)
        queue1.unlink(fakesink)
        buffer.link(queue2)
        queue2.link(fakesink)
        flag = False
    else:
        # 如果网络未连接，恢复原始pipeline
        buffer.unlink(queue2)
        queue2.unlink(fakesink)
        buffer.link(queue1)
        queue1.link(fakesink)
        flag = True

    # 返回True以使定时器继续工作
    return True

def main(args):

    global queue1
    global queue2
    global fakesink1
    global fakesink2
    global blockpad
    global cur_effect
    global next_effect
    global pipeline
    # Check input arguments
    # 第二个参数是媒体文件或者uri
    if len(args) != 2:
        sys.stderr.write("usage: %s <media file or uri>\n" % args[0])
        sys.exit(1)

    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    # Source element for reading from the file
    source = make_elm_or_print_err("filesrc", "file-source", "Source")

    # Since the data format in the input file is elementary h264 stream,
    # we need a h264parser
    h264parser = make_elm_or_print_err("h264parse", "h264-parser", "H264Parser")

    # Use nvdec_h264 for hardware accelerated decode on GPU
    decoder = make_elm_or_print_err("nvv4l2decoder", "nvv4l2-decoder", "Decoder")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = make_elm_or_print_err("nvstreammux", "Stream-muxer", "NvStreamMux")

    # Use nvinferserver to run inferencing on decoder's output,
    # behaviour of inferencing is set through config file
    pgie = make_elm_or_print_err("nvinfer", "primary-inference", "Nvinferserver")

    buffer = make_elm_or_print_err("queue", "buffer", "Buffer")


    queue1 = make_elm_or_print_err("queue", "queue1", "Queue1")
    queue2 = make_elm_or_print_err("queue", "queue2", "Queue2")
    cur_effect = queue1
    next_effect = queue2
    # fakesink 用来接收前面的数据，作为最后的link，不做任何处理，不能删掉
    fakesink = make_elm_or_print_err("fakesink", "fakesink", "fakesink")

    print("Playing file %s " % args[1])
    source.set_property("location", args[1])
    streammux.set_property("width", IMAGE_WIDTH)
    streammux.set_property("height", IMAGE_HEIGHT)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", 4000000)

    # .txt文件中配置了模型的路径，以及模型的参数
    pgie.set_property("config-file-path", "config_infer_primary_yoloV5.txt")

    print("Adding elements to Pipeline \n")
    pipeline.add(source)
    pipeline.add(h264parser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(buffer)
    pipeline.add(queue1)
    pipeline.add(queue2)
    pipeline.add(fakesink)

    print("Linking elements in the Pipeline \n")
    source.link(h264parser)
    h264parser.link(decoder)

    sinkpad = streammux.get_request_pad("sink_0")

    srcpad = decoder.get_static_pad("src")

    srcpad.link(sinkpad)
    streammux.link(pgie)
    pgie.link(buffer)
    buffer.link(queue1)
    queue1.link(fakesink)

    loop = GObject.MainLoop()  # 创建主循环
    bus = pipeline.get_bus()  # 获取总线
    bus.add_signal_watch()  # 添加信号监视器
    bus.connect("message", bus_call, loop)  # 连接信号处理函数

    queue1srcpad = queue1.get_static_pad("src")
    queue1srcpad.add_probe(Gst.PadProbeType.BUFFER, queue1_src_pad_buffer_probe, 0)

    queue2srcpad = queue2.get_static_pad("src")
    queue2srcpad.add_probe(Gst.PadProbeType.BUFFER, queue2_src_pad_buffer_probe, 0)

    pgiesrcpad = pgie.get_static_pad("src")
    if not pgiesrcpad:
        sys.stderr.write(" Unable to get src pad of primary infer \n")

    pgiesrcpad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, 0)

    blockpad = buffer.get_static_pad("src")

    GObject.timeout_add(1000, check_network)
    # GLib.timeout_add_seconds(0.1, timeout_cb, loop)

    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
