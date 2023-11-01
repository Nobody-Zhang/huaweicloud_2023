import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

Gst.init(None)
loop = GObject.MainLoop()

pipeline = Gst.Pipeline()
filesrc = Gst.ElementFactory.make("filesrc")
decodebin = Gst.ElementFactory.make("decodebin")
videoconvert = Gst.ElementFactory.make("videoconvert")
x264enc = Gst.ElementFactory.make("x264enc")
rtph264pay = Gst.ElementFactory.make("rtph264pay")
udpsink = Gst.ElementFactory.make("udpsink")
autovideosink = Gst.ElementFactory.make("autovideosink")

if not pipeline or not filesrc or not decodebin or not videoconvert or not x264enc or not rtph264pay or not udpsink or not autovideosink:
    print("Not all elements could be created")
    exit(1)

filesrc.set_property("location", "/home/jetson/DeepStream-Yolo/test.mp4")  # Change the file path here
udpsink.set_property("host", "127.0.0.1")
udpsink.set_property("port", 5000)

pipeline.add(filesrc)
pipeline.add(decodebin)
pipeline.add(videoconvert)
pipeline.add(x264enc)
pipeline.add(rtph264pay)
pipeline.add(udpsink)
pipeline.add(autovideosink)

filesrc.link(decodebin)
decodebin.connect("pad-added", lambda dbin, pad: pad.link(videoconvert.get_static_pad("sink")))
videoconvert.link(x264enc)
x264enc.link(rtph264pay)
rtph264pay.link(udpsink)
videoconvert.link(autovideosink)

bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message", lambda bus, message: loop.quit())

pipeline.set_state(Gst.State.PLAYING)
try:
    loop.run()
except Exception as e:
    print("Error:", e)
finally:
    pipeline.set_state(Gst.State.NULL)
