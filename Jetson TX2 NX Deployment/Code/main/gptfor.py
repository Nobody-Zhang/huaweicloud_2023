import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

def on_gst_message(bus, message, loop):
    mtype = message.type
    if mtype == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print("Error: %s" % err, debug)
        loop.quit()
    elif mtype == Gst.MessageType.EOS:
        print("End of stream")
        loop.quit()
    elif mtype == Gst.MessageType.ELEMENT:
        struct = message.get_structure()
        if struct.get_name() == 'splitmuxsink':
            filename = struct.get_string('location')
            print('New File:', filename)
    return True

def main():
    Gst.init(None)
    loop = GLib.MainLoop()

    # Create GStreamer elements
    pipeline = Gst.Pipeline()

    src = Gst.ElementFactory.make('filesrc', 'filesrc')
    src.set_property('location', 'in.mp4')

    demux = Gst.ElementFactory.make('qtdemux', 'demux')

    parse = Gst.ElementFactory.make('h264parse', 'parse')

    mux = Gst.ElementFactory.make('mp4mux', 'mux')

    sink = Gst.ElementFactory.make('splitmuxsink', 'sink')
    sink.set_property('location', "%02d.mp4")
    sink.set_property('max-size-time', 1000000000)

    pipeline.add(src)
    pipeline.add(demux)
    pipeline.add(parse)
    pipeline.add(mux)
    pipeline.add(sink)

    src.link(demux)
    demux.link(parse)
    parse.link(mux)
    mux.link(sink)

    # Start playing
    pipeline.set_state(Gst.State.PLAYING)

    # Listen to Gstreamer bus messages
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_gst_message, loop)

    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    main()
