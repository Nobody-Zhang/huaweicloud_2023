import time
import ctypes
import numpy as np
# import cv2
import os
import pyds
import pathlib
import sys
import platform
import io
from queue import Queue
import threading
from loguru import logger
from cloudinfer import cloud_infer
sys.path.append("../")
import gi

gi.require_version("Gst", "1.0")
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GObject, Gst, GstRtspServer
video_writer = None

def is_aarch64():
    return platform.uname()[4] == 'aarch64'

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

CLASS_NB = 7
ACCURACY_ALL_CLASS = 0.5
UNTRACKED_OBJECT_ID = 0xffffffffffffffff
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280
OUTPUT_VIDEO_NAME = "./out.mp4"
np.set_printoptions(threshold=np.inf)
INPUT_HEIGHT = 640
INPUT_WIDTH = 640


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
    logger.info("Creating " + printedname)
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



def osd_sink_pad_buffer_probe(pad, info, u_data):
    """
    Get tensor_metadata from triton inference output
    Convert tensor metadata to numpy array
    Postprocessing
    """
    global frame_queue
    global video_writer
    image_width = 1280
    image_height = 720
    conf_thresh = 0.5
    nms_thresh = 0.5
    label = get_label_names_from_file()
    print(label)
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
        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        # convert python array into numpy array format in the copy mode.
        frame_copy = np.array(n_frame, copy=True, order='C')
        # convert the array into cv2 default color format
        # pic = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGRA)
        # pic = cv2.cvtColor(pic, cv2.COLOR_BGRA2BGR)
        # cv2.imwrite(f"test_{time.time()}.jpg", pic)
        # cv2.imshow("pic", pic)
        # filename = 'tmp.npy'

        # np.save(filename, pic)
        # video_writer.write(pic)
        # frame_queue.append(pic)
        # frame_queue.put(pic)
        # cv2.imwrite(f"test_{time.time()}.jpg", pic)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()
        """
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
            # 用cropped_image = pic[int(top):int(top + height), int(left):int(left + width)] 裁剪
            if class_id == 2 or class_id == 6:
                # 2: face, 6: side_face
                cropped_image = pic[int(top):int(top + height), int(left):int(left + width)]
                tmp_name = f"{label[class_id]}_{confidence}.jpg"
                cv2.imwrite(tmp_name, cropped_image)
                # res = cloud_infer(tmp_name) # 云端推理这里可以用confidence联合推理
                # if res == "OK":
                #     pass
#---------------------!!!!!!!!!!!!!!!!!补充端云协同!!!!!!!!!!!!!!!!!!!!!!---------------------------------
            # print("class_id: ", class_id)
            # print("top: ", top)
            # print("left: ", left)
            # print("width: ", width)
            # print("height: ", height)
            # cropped_image = pic[int(top):int(top + height), int(left):int(left + width)]
            # cv2.imwrite(f"{class_id}_{time.time()}_{confidence}.jpg", cropped_image)
            try:
                iter_obj = iter_obj.next
            except StopIteration:
                break
        """

        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK


def cb_newpad(decodebin, decoder_src_pad, data):
    logger.info("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    logger.info("gstname= %s" % gstname)
    if gstname.find("video") != -1:
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        logger.info("features= %s" % features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                logger.opt(colors=True).warning("WARNING: Failed to link decoder src pad to source bin ghost pad\n")
        else:
            logger.error("ERROR: Decodebin did not pick nvidia decoder plugin.\n")

def decodebin_child_added(child_proxy, Object, name, user_data):
    logger.info(f"Decodebin child added: {name} \n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)


def create_source_bin(idx, uri):
    """
    Decode file video local and URI video input
    """
    logger.info("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % idx
    #logger.info(f"source_name: {bin_name}")
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        logger.opt(colors=True).warning("WARNING: Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        logger.opt(colors=True).warning("WARNING: Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        logger.opt(colors=True).warning("WARNING: Failed to add ghost pad in source bin \n")
        return None
    return nbin

def main(args):
    is_save_output = False
    codec = "H264"
    bitrate = 4000000
    GObject.threads_init()
    Gst.init(None)
    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    logger.info("Creating source bin...")
    source = make_elm_or_print_err("nvarguscamerasrc", "src-elem", "src-elem")
    nvvidconv_src = make_elm_or_print_err("nvvideoconvert", "nvvidconv-src", "nvvidconv-src")
    caps_nvvidconv_src = make_elm_or_print_err("capsfilter", "caps-nvvidconv-src", "caps-nvvidconv-src")
    caps_nvvidconv_src.set_property('caps', Gst.Caps.from_string('video/x-raw(memory:NVMM), width=1280, height=720'))

    streammux = make_elm_or_print_err("nvstreammux", "Stream-muxer", "NvStreamMux")

    # Use nvinferserver to run inferencing on decoder's output,
    # behaviour of inferencing is set through config file
    pgie = make_elm_or_print_err("nvinfer", "primary-inference", "Nvinferserver")

    # Use convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = make_elm_or_print_err("nvvideoconvert", "convertor", "Nvvidconv")

    # Create OSD to draw on the converted RGBA buffer
    nvosd = make_elm_or_print_err("nvdsosd", "onscreendisplay", "OSD (nvosd)")

    nvvidconv_postosd = make_elm_or_print_err("nvvideoconvert", "convertor_postosd", "Nvvidconv_postosd")

    # Create a caps filter
    caps = make_elm_or_print_err("capsfilter", "filter", "filter")
    caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))

    # Make the encoder
    encoder = make_elm_or_print_err("nvv4l2h264enc", "encoder", "encoder")
    encoder.set_property('maxperf-enable',1)
    encoder.set_property('bitrate', bitrate)

    if is_aarch64():
        encoder.set_property('insert-sps-pps', 1)

    # Make the payload-encode video into RTP packets
    rtppay = make_elm_or_print_err("rtph264pay", "rtppay", "rtppay")


    # Make the UDP sink
    updsink_port_num = 5400
    udpsink = make_elm_or_print_err("udpsink", "udpsink", "udpsink")
    if not udpsink:
        sys.stderr.write(" Unable to create udpsink")

    udpsink.set_property('host', '224.224.255.255')
    udpsink.set_property('port', updsink_port_num)
    udpsink.set_property('async', False)
    udpsink.set_property('sync', 0)


    source.set_property('bufapi-version', True)

    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 1)

    streammux.set_property('live-source', 1)

    streammux.set_property('batched-push-timeout', 4000000)

    logger.info("DeepStream Triton yolov5 tensorRT inference")
    pgie.set_property("config-file-path", "config_infer_primary_yoloV5.txt")

    # store the cache
    queue1 = Gst.ElementFactory.make("queue", "queue1")
    queue2 = Gst.ElementFactory.make("queue", "queue2")
    queue3 = Gst.ElementFactory.make("queue", "queue3")
    queue4 = Gst.ElementFactory.make("queue", "queue4")
    queue5 = Gst.ElementFactory.make("queue", "queue5")
    pipeline.add(queue1)
    pipeline.add(queue2)
    pipeline.add(queue3)
    pipeline.add(queue4)
    pipeline.add(queue5)

    pipeline.add(source)
    pipeline.add(nvvidconv_src)
    pipeline.add(caps_nvvidconv_src)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv_postosd)
    pipeline.add(caps)
    pipeline.add(encoder)
    pipeline.add(rtppay)
    pipeline.add(udpsink)

    logger.info("Linking elements in the source bin...")

    source.link(nvvidconv_src)
    nvvidconv_src.link(caps_nvvidconv_src)

    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    srcpad = caps_nvvidconv_src.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of caps_nvvidconv_src \n")
    srcpad.link(sinkpad)

    """
    streammux.link(pgie)
    # streammux.link(queue1)
    pgie.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(nvvidconv_postosd)
    nvvidconv_postosd.link(caps)
    caps.link(encoder)
    encoder.link(rtppay)
    rtppay.link(udpsink)
    """
    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(queue2)
    queue2.link(nvvidconv)
    nvvidconv.link(queue3)
    queue3.link(nvosd)
    nvosd.link(queue4)
    queue4.link(nvvidconv_postosd)
    nvvidconv_postosd.link(caps)
    caps.link(encoder)
    encoder.link(rtppay)
    rtppay.link(udpsink)

    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Start streaming
    rtsp_port_num = 8554

    server = GstRtspServer.RTSPServer.new()
    server.props.service = "%d" % rtsp_port_num
    server.attach(None)

    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch(
        "( udpsrc name=pay0 port=%d buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 \" )" % (
        updsink_port_num, "H264"))
    factory.set_shared(True)
    server.get_mount_points().add_factory("/test", factory)

    logger.info(" *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d/test ***" % rtsp_port_num)

    # start play back and listen to events
    logger.info("Starting pipeline")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
