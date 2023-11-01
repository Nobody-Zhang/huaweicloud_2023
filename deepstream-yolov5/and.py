
import numpy as np
import time
import cv2
import argparse
import sys
from loguru import logger
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
import platform
from gi.repository import GObject, Gst, GstRtspServer
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

def is_aarch64():
    return platform.uname()[4] == 'aarch64'

import pyds

bitrate=10000000

def main(args):

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
    print("Creating Source \n ")
    source = Gst.ElementFactory.make("nvarguscamerasrc", "src-elem")
    if not source:
        sys.stderr.write(" Unable to create Source \n")

    nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd")
    if not nvvidconv_postosd:
        sys.stderr.write(" Unable to create nvvidconv_postosd \n")

    # Create a caps filter
    caps = Gst.ElementFactory.make("capsfilter", "filter")
    caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"))

    # Make the encoder
    encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
    print("Creating H264 Encoder")
    if not encoder:
        sys.stderr.write(" Unable to create encoder")
    encoder.set_property('maxperf-enable',1)
    encoder.set_property('bitrate', bitrate)
    if is_aarch64():
        encoder.set_property('insert-sps-pps', 1)
    # Make the payload-encode video into RTP packets
    rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
    print("Creating H264 rtppay")
    if not rtppay:
        sys.stderr.write(" Unable to create rtppay")
    # Make the UDP sink
    updsink_port_num = 5400
    sink = Gst.ElementFactory.make("udpsink", "udpsink")
    if not sink:
        sys.stderr.write(" Unable to create udpsink")

    sink.set_property('host', '224.224.255.255')
    sink.set_property('port', updsink_port_num)
    sink.set_property('async', False)
    sink.set_property('sync', 1)



    print("Creating EGLSink \n")
    source.set_property('bufapi-version', True)

    #sink.set_property('sync', False)

    print("Adding elements to Pipeline \n")

    pipeline.add(source)
    pipeline.add(nvvidconv_postosd)
    pipeline.add(caps)
    pipeline.add(encoder)
    pipeline.add(rtppay)
    pipeline.add(sink)


    # we link the elements together
    print("Linking elements in the Pipeline \n")
    source.link(nvvidconv_postosd)
    nvvidconv_postosd.link(caps)
    caps.link(encoder)
    encoder.link(rtppay)
    rtppay.link(sink)


    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    # Start streaming
    rtsp_port_num = 8554

    server = GstRtspServer.RTSPServer.new()
    server.props.service = "%d" % rtsp_port_num
    server.attach(None)

    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch( "( udpsrc name=pay0 port=%d buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 \" )" % (updsink_port_num, "H264"))
    factory.set_shared(True)
    server.get_mount_points().add_factory("/test", factory)
    print("\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d/test ***\n\n" % rtsp_port_num)


    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)
if __name__ == '__main__':
    sys.exit(main(sys.argv))