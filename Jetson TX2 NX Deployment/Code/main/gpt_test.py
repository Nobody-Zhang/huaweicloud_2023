import gi
import time
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

# Initialize GStreamer
Gst.init(None)

# Create an empty pipeline
pipeline = Gst.Pipeline.new("test-pipeline")

# Create the elements
src = Gst.ElementFactory.make("filesrc", "src")
src.set_property("location", "/home/jetson/DeepStream-Yolo/day_man_053_31_1.mp4")
decode = Gst.ElementFactory.make("decodebin", "decode")
queue = Gst.ElementFactory.make("queue", "queue")
sink = Gst.ElementFactory.make("autovideosink", "sink")

# Add elements to the pipeline
pipeline.add(src)
pipeline.add(decode)
pipeline.add(queue)
pipeline.add(sink)

# Link the elements
src.link(decode)

# Connect a signal handler to the "pad-added" signal of the decodebin element
def on_pad_added(element, pad):
    queue_pad = queue.get_static_pad("sink")
    if not queue_pad.is_linked():
        pad.link(queue_pad)

decode.connect("pad-added", on_pad_added)

queue.link(sink)

# Define a callback function for the probe
def on_buffer(pad, info, udata):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return Gst.PadProbeReturn.PASS

    # Get the current time. If it has been more than 1 second since the last save, save the current frame as a video
    current_time = time.time()
    if current_time - udata['last_time'] >= 1.0:
        # Create a new pipeline for encoding the video
        encode_pipeline = Gst.parse_launch("appsrc name=src ! videoconvert ! x264enc ! mp4mux ! filesink location=video_{}.mp4".format(udata['video_count']))
        appsrc = encode_pipeline.get_by_name("src")

        # Push the buffer to the appsrc element
        appsrc.emit('push-buffer', gst_buffer)

        # Save the current time
        udata['last_time'] = current_time
        udata['video_count'] += 1

    return Gst.PadProbeReturn.PASS

# Add a probe to the src pad of the queue
udata = {'last_time': time.time(), 'video_count': 0}
queue_src_pad = queue.get_static_pad("src")
queue_src_pad.add_probe(Gst.PadProbeType.BUFFER, on_buffer, udata)

# Start playing
pipeline.set_state(Gst.State.PLAYING)

# Wait until error or EOS
bus = pipeline.get_bus()
msg = bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS)

# Free resources
pipeline.set_state(Gst.State.NULL)
