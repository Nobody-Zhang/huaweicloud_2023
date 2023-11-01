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
import multiprocessing

sys.path.append("../")
import gi

gi.require_version("Gst", "1.0")
from gi.repository import GObject, Gst
import platform
import sys


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
OUTPUT_VIDEO_NAME1 = "./out1.mp4"
OUTPUT_VIDEO_NAME2 = "./out2.mp4"

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
        # print("number_of_counts:", frame_meta.num_obj_meta)
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

def switch_sink(pipeline):

    # 创建两个sink
    sink1 = pipeline.get_by_name("filesink1")
    sink2 = pipeline.get_by_name("filesink2")
    container = pipeline.get_by_name("qtmux")
    print("now switch sink")

    current_sink = sink1
    while True:
        # 等待1秒
        time.sleep(0.1)

        # 切换sink
        if current_sink == sink1:
            sink1.send_event(Gst.Event.new_eos())
            container.unlink(sink1)
            print("switch to sink1")
            time.sleep(2)
            os.remove("out1.mp4")
            container.link(sink1)
            current_sink = sink1
            print("switch to sink1")
        else:
            container.unlink(sink2)
            container.link(sink1)
            current_sink = sink1
            print("switch to sink1")


def main(args):
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

    # fakesink 用来接收前面的数据，作为最后的link，不做任何处理，不能删掉
    # fakesink = Gst.ElementFactory.make("fakesink")

    # Use convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = make_elm_or_print_err("nvvideoconvert", "convertor", "Nvvidconv")

    # Create OSD to draw on the converted RGBA buffer
    nvosd = make_elm_or_print_err("nvdsosd", "onscreendisplay", "OSD (nvosd)")

    # Finally encode and save the osd output
    queue = make_elm_or_print_err("queue", "queue", "Queue")

    nvvidconv2 = make_elm_or_print_err("nvvideoconvert", "convertor2", "Converter 2 (nvvidconv2)")

    capsfilter = make_elm_or_print_err("capsfilter", "capsfilter", "capsfilter")

    caps = Gst.Caps.from_string("video/x-raw, format=I420")
    capsfilter.set_property("caps", caps)

    # On Jetson, there is a problem with the encoder failing to initialize
    # due to limitation on TLS usage. To work around this, preload libgomp.
    # Add a reminder here in case the user forgets.
    preload_reminder = "If the following error is encountered:\n" + \
                       "/usr/lib/aarch64-linux-gnu/libgomp.so.1: cannot allocate memory in static TLS block\n" + \
                       "Preload the offending library:\n" + \
                       "export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1\n"
    encoder = make_elm_or_print_err("avenc_mpeg4", "encoder", "Encoder", preload_reminder)

    encoder.set_property("bitrate", 2000000)

    codeparser = make_elm_or_print_err("mpeg4videoparse", "mpeg4-parser", 'Code Parser')

    container = make_elm_or_print_err("qtmux", "qtmux", "Container")

    sink1 = make_elm_or_print_err("filesink", "filesink1", "Sink1")

    sink1.set_property("location", OUTPUT_VIDEO_NAME1)
    sink1.set_property("sync", 0)
    sink1.set_property("async", 0)

    sink2 = make_elm_or_print_err("filesink", "filesink2", "Sink2")

    sink2.set_property("location", OUTPUT_VIDEO_NAME2)
    sink2.set_property("sync", 0)
    sink2.set_property("async", 0)

    print("Playing file %s " % args[1])
    source.set_property("location", args[1])
    streammux.set_property("width", IMAGE_WIDTH)
    streammux.set_property("height", IMAGE_HEIGHT)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", 4000000)

    # .txt文件中配置了模型的路径，以及模型的参数
    pgie.set_property("config-file-path", "config_infer_primary_yoloV5.txt")

    print("Adding elements to Pipeline \n")
    # 从文件中读取h264流
    pipeline.add(source)
    # 从h264流中解析出h264数据（解码器）
    pipeline.add(h264parser)
    # 用nvdec_h264 在GPU上加速解码
    pipeline.add(decoder)
    # 将多个batch合并成一个batch
    pipeline.add(streammux)
    # 使用nvinfer对解码器的输出运行推理
    # 从文件中读取配置文件，配置文件中包含了模型的路径，以及模型的参数
    pipeline.add(pgie)
    # pipeline.add(fakesink)
    # 从NV12转换为RGBA，因为nvosd需要RGBA格式的数据
    pipeline.add(nvvidconv)
    # 绘制图像
    pipeline.add(nvosd)

    # 接下来都是转化为视频文件的操作
    pipeline.add(queue)
    pipeline.add(nvvidconv2)
    pipeline.add(capsfilter)
    pipeline.add(encoder)
    pipeline.add(codeparser)
    pipeline.add(container)
    pipeline.add(sink1)
    pipeline.add(sink2)

    # we link the elements together
    # file-source -> h264-parser -> nvh264-decoder ->
    # nvinfer -> nvvidconv -> nvosd -> video-renderer
    print("Linking elements in the Pipeline \n")
    source.link(h264parser)
    h264parser.link(decoder)

    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    srcpad = decoder.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of decoder \n")
    srcpad.link(sinkpad)
    streammux.link(pgie)
    # pgie.link(fakesink)
    pgie.link(nvvidconv)
    nvvidconv.link(nvosd)

    # 以下是转化为视频文件的操作
    nvosd.link(queue)
    queue.link(nvvidconv2)
    nvvidconv2.link(capsfilter)
    capsfilter.link(encoder)
    encoder.link(codeparser)
    codeparser.link(container)
    container.link(sink1)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()  # 创建主循环
    bus = pipeline.get_bus()  # 获取总线
    bus.add_signal_watch()  # 添加信号监视器
    bus.connect("message", bus_call, loop)  # 连接信号处理函数

    # Add a probe on the primary-infer source pad to get inference output tensors
    # pgiesrcpad（探针）用来获取推理输出的张量
    pgiesrcpad = pgie.get_static_pad("src")
    if not pgiesrcpad:
        sys.stderr.write(" Unable to get src pad of primary infer \n")

    pgiesrcpad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, 0)

    # 我觉得下面的不需要了----------------------
    # Lets add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.

    # osdsinkpad = nvosd.get_static_pad("sink")
    # if not osdsinkpad:
    #     sys.stderr.write(" Unable to get sink pad of nvosd \n")
    #
    # osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # --------------------------------------------
    # start play back and listen to events
    # 开启pipeline

    print("Starting pipeline \n")
    process = multiprocessing.Process(target=switch_sink, args=(pipeline,))

    process.start()
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
