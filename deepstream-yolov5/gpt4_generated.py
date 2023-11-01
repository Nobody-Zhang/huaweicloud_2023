import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

opt_effects = None
DEFAULT_EFFECTS = ["identity", "exclusion", "navigationtest", "agingtv", "videoflip", "vertigotv", "gaussianblur", "shagadelictv", "edgetv"]
effects = []

pipeline = None
cur_effect = None
conv_before = None
conv_after = None

def event_probe_cb(pad, info, loop):
    global cur_effect, pipeline, conv_before, conv_after, effects

    if Gst.Pad.get_current_caps(pad).to_string() != 'video/x-raw':
        return Gst.PadProbeReturn.PASS

    next_effect = effects.pop(0)
    if next_effect is None:
        loop.quit()
        return Gst.PadProbeReturn.DROP

    cur_effect.set_state(Gst.State.NULL)
    pipeline.remove(cur_effect)
    pipeline.add(next_effect)
    conv_before.link(next_effect)
    next_effect.link(conv_after)

    next_effect.set_state(Gst.State.PLAYING)
    cur_effect = next_effect

    return Gst.PadProbeReturn.DROP

def pad_probe_cb(pad, info, loop):
    srcpad = cur_effect.get_static_pad('src')
    srcpad.add_probe(Gst.PadProbeType.BLOCK_DOWNSTREAM, event_probe_cb, loop)
    return Gst.PadProbeReturn.OK

def timeout_cb(loop):
    global blockpad
    blockpad.add_probe(Gst.PadProbeType.BLOCK_DOWNSTREAM, pad_probe_cb, loop)
    return True

def bus_cb(bus, msg, loop):
    t = msg.type
    if t == Gst.MessageType.ERROR:
        err, dbg = msg.parse_error()
        print("ERROR:", msg.src.get_name(), ":", err.message)
        if dbg:
            print("Debug info:", dbg)
        loop.quit()
    return True

def main():
    global pipeline, cur_effect, conv_before, conv_after, blockpad, effects, opt_effects

    Gst.init(None)

    if opt_effects is not None:
        effect_names = opt_effects.split(',')
    else:
        effect_names = DEFAULT_EFFECTS

    for e in effect_names:
        el = Gst.ElementFactory.make(e, None)
        if el:
            effects.append(el)

    pipeline = Gst.Pipeline.new("pipeline")

    src = Gst.ElementFactory.make("videotestsrc", None)
    src.set_property("is-live", True)

    conv_before = Gst.ElementFactory.make("videoconvert", None)
    cur_effect = effects.pop(0)
    conv_after = Gst.ElementFactory.make("videoconvert", None)
    sink = Gst.ElementFactory.make("ximagesink", None)

    pipeline.add(src)
    pipeline.add(conv_before)
    pipeline.add(cur_effect)
    pipeline.add(conv_after)
    pipeline.add(sink)

    if not src.link(conv_before) or not conv_before.link(cur_effect) or not cur_effect.link(conv_after) or not conv_after.link(sink):
        print("ERROR: Could not link elements.")
        sys.exit(1)

    blockpad = conv_before.get_static_pad("src")

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_cb, loop)

    pipeline.set_state(Gst.State.PLAYING)

    GLib.timeout_add_seconds(1, timeout_cb, loop)

    try:
        loop.run()
    except:
        pass

    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    main()
