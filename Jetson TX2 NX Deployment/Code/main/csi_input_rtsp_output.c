/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include "gst-nvdssr.h"
//#include <gst/rtsp-server/rtsp-server.h>

GST_DEBUG_CATEGORY (NVDS_APP);

#define MAX_DISPLAY_LEN 64

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

/* By default, OSD process-mode is set to CPU_MODE. To change mode, set as:
 * 1: GPU mode (for Tesla only)
 * 2: HW mode (For Jetson only)
 */
#define OSD_PROCESS_MODE 0

/* By default, OSD will not display text. To display text, change this to 1 */
#define OSD_DISPLAY_TEXT 1

gint frame_number = 0;
gchar pgie_classes_str[7][32] = { "close_eye", "close_mouth", "face", "open_eye", "open_mouth", "phone", "side_face"};


/* Config file parameters used for recording
 * User needs to change these parameters to reflect the change in recordings
 * e.g duration, start-time etc. */

/* Container format of recorded file 0 for mp4 and 1 for mkv format
 */
#define SMART_REC_CONTAINER 0

/* Cache functionality of recording
 */
#define CACHE_SIZE_SEC 15

/* Timeout if duration of recording is not set by user
 */
#define SMART_REC_DEFAULT_DURATION 10

/* Time at which it recording is started
 */
#define START_TIME 2

/* Duration of recording
 */
#define SMART_REC_DURATION 7

/* Interval in seconds for
 * SR start / stop events generation.
 */
#define SMART_REC_INTERVAL 7

static gboolean bbox_enabled = TRUE;
static gint enc_type = 0; // Default: Hardware encoder
static gint sink_type = 3; // Default: Eglsink
static guint sr_mode = 1; // Default: Video Only

GOptionEntry entries[] = {
        {"bbox-enable", 'e', 0, G_OPTION_ARG_INT, &bbox_enabled,
                "0: Disable bboxes, \
       1: Enable bboxes, \
       Default: bboxes enabled", NULL}
        ,
        {"enc-type", 'c', 0, G_OPTION_ARG_INT, &enc_type,
                "0: Hardware encoder, \
       1: Software encoder, \
       Default: Hardware encoder", NULL}
        ,
        {"sink-type", 's', 0, G_OPTION_ARG_INT, &sink_type,
                "1: Fakesink, \
       2: Eglsink, \
       3: RTSP sink, \
       Default: Eglsink", NULL}
        ,
        {"sr-mode", 'm', 0, G_OPTION_ARG_INT, &sr_mode,
                "SR mode: 0 = Audio + Video, \
       1 = Video only, \
       2 = Audio only", NULL}
        ,
        {NULL}
        ,
};

static GstElement *pipeline = NULL, *tee_pre_decode = NULL;
static NvDsSRContext *nvdssrCtx = NULL;
static GMainLoop *loop = NULL;

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
    GMainLoop *loop = (GMainLoop *) data;
    switch (GST_MESSAGE_TYPE (msg)) {
        case GST_MESSAGE_EOS:
            g_print ("End of stream\n");
            g_main_loop_quit (loop);
            break;
        case GST_MESSAGE_ERROR:{
            gchar *debug;
            GError *error;
            gst_message_parse_error (msg, &error, &debug);
            g_printerr ("ERROR from element %s: %s\n",
                        GST_OBJECT_NAME (msg->src), error->message);
            if (debug)
                g_printerr ("Error details: %s\n", debug);
            g_free (debug);
            g_error_free (error);
            g_main_loop_quit (loop);
            break;
        }
        default:
            break;
    }
    return TRUE;
}


static gpointer
smart_record_callback (NvDsSRRecordingInfo * info, gpointer userData)
{
    static GMutex mutex;
    FILE *logfile = NULL;
    g_return_val_if_fail (info, NULL);

    g_mutex_lock (&mutex);
    logfile = fopen ("smart_record.log", "a");
    if (logfile) {
        fprintf (logfile, "%d:%s:%d:%d:%s:%d channel(s):%d Hz:%ldms:%s:%s\n",
                 info->sessionId, info->containsVideo ? "video" : "no-video",
                 info->width, info->height, info->containsAudio ? "audio" : "no-audio",
                 info->channels, info->samplingRate, info->duration,
                 info->dirpath, info->filename);
        fclose (logfile);
    } else {
        g_print ("Error in opeing smart record log file\n");
    }
    g_mutex_unlock (&mutex);

    return NULL;
}


static gboolean
smart_record_event_generator (gpointer data)
{
    NvDsSRSessionId sessId = 0;
    NvDsSRContext *ctx = (NvDsSRContext *) data;
    guint startTime = START_TIME;
    guint duration = SMART_REC_DURATION;

    if (ctx->recordOn) {
        g_print ("Recording done.\n");
        if (NvDsSRStop (ctx, 0) != NVDSSR_STATUS_OK)
            g_printerr ("Unable to stop recording\n");
    } else {
        g_print ("Recording started..\n");
        if (NvDsSRStart (ctx, &sessId, startTime, duration,
                         NULL) != NVDSSR_STATUS_OK)
            g_printerr ("Unable to start recording\n");
    }
    return TRUE;
}

int
main (int argc, char *argv[])
{
    GstElement *source = NULL, *nvvidconv_src = NULL, *tee_pre = NULL,
            *queue_tee = NULL, *caps_nvvidconv_src = NULL, *streammux = NULL,
            *pgie = NULL, *nvvidconv = NULL, *nvosd = NULL, *nvvidconv2 = NULL,
            *cap_filter = NULL, *sink = NULL, *queue_sr = NULL, *caps_sr = NULL,
            *encoder_pre = NULL, *parser_pre = NULL, *tes_fake = NULL, *tes_fake2 = NULL;

    GstCaps *caps = NULL, *caps_src = NULL, *caps_sr_dat = NULL;

    GstBus *bus = NULL;
    guint bus_watch_id = 0;
    guint i = 0, num_sources = 1;

    guint pgie_batch_size = 0;

    int current_device = -1;
    cudaGetDevice(&current_device);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);

    GstElement *transform = NULL;
    GOptionContext *gctx = NULL;
    GOptionGroup *group = NULL;
    GError *error = NULL;

    NvDsSRInitParams params = { 0 };

    gctx = g_option_context_new ("Nvidia DeepStream Test-SR app");
    group = g_option_group_new ("SR_test", NULL, NULL, NULL, NULL);
    g_option_group_add_entries (group, entries);

    g_option_context_set_main_group (gctx, group);
    g_option_context_add_group (gctx, gst_init_get_option_group ());

    GST_DEBUG_CATEGORY_INIT (NVDS_APP, "NVDS_APP", 0, NULL);

    if (!g_option_context_parse (gctx, &argc, &argv, &error)) {
        g_printerr ("%s", error->message);
        g_print ("%s", g_option_context_get_help (gctx, TRUE, NULL));
        return -1;
    }

    /* Standard GStreamer initialization */
    gst_init (&argc, &argv);
    loop = g_main_loop_new (NULL, FALSE);

    /* Create gstreamer elements */
    /* Create Pipeline element that will form a connection of other elements */
    pipeline = gst_pipeline_new ("dstest-sr-pipeline");

    source = gst_element_factory_make ("nvarguscamerasrc", "src-elem");

    g_object_set (G_OBJECT (source), "bufapi-version", TRUE, NULL);

    nvvidconv_src = gst_element_factory_make ("nvvideoconvert", "nvvidconv_src");

    /* Create tee which connects decoded source data and Smart record bin without bbox */
    tee_pre = gst_element_factory_make ("tee", "tee-pre");

    queue_tee = gst_element_factory_make ("queue", "queue-tee");

    /* for pgie, before streammux */
    caps_src = gst_caps_from_string ("video/x-raw(memory:NVMM), width=1280, height=720");
    caps_nvvidconv_src =
            gst_element_factory_make ("capsfilter", "caps-nvvidconv-src");
    g_object_set (G_OBJECT (caps_nvvidconv_src), "caps", caps_src, NULL);
    gst_caps_unref (caps_src);

    streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

    /* Use nvinfer to infer on batched frame. */
    pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");

    /* Use convertor to convert from NV12 to RGBA as required by nvosd */
    nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

    /* Use convertor to convert from RGBA to CAPS filter data format */
    nvvidconv2 =
            gst_element_factory_make ("nvvideoconvert", "nvvideo-converter2");

    /* Create OSD to draw on the converted RGBA buffer */
    nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

    /* Create a fakesink for test */

    tes_fake = gst_element_factory_make ("fakesink", "fakesink-test");

    /* Finally render the osd output */
    if(prop.integrated) {
        transform = gst_element_factory_make ("nvegltransform", "nvegl-transform");
        if (!transform) {
            g_printerr ("One tegra element could not be created. Exiting.\n");
            return -1;
        }
    }

    if (sink_type == 1) {
        sink = gst_element_factory_make ("fakesink", "nvvideo-renderer");
    }
    else if (sink_type == 2) {
        sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
        g_object_set (G_OBJECT (sink), "async", FALSE, NULL);
    }
    else if (sink_type == 3) {
        sink = gst_element_factory_make ("nvrtspoutsinkbin", "nvvideo-renderer");
        g_object_set (G_OBJECT (sink), "sync", TRUE, NULL);
        g_object_set (G_OBJECT (sink), "bitrate", 768000, NULL);
        g_object_set (G_OBJECT (sink), "enc-type", 1, NULL);
    }

    g_object_set (G_OBJECT (streammux), "live-source", 1, NULL);

    g_object_set (G_OBJECT (streammux), "batch-size", num_sources, NULL);

    g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
                  MUXER_OUTPUT_HEIGHT,
                  "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

    caps = gst_caps_from_string ("video/x-raw(memory:NVMM), format=(string)I420");
    cap_filter =
            gst_element_factory_make ("capsfilter", "src_cap_filter_nvvidconv");
    g_object_set (G_OBJECT (cap_filter), "caps", caps, NULL);
    gst_caps_unref (caps);

    /* Configure the nvinfer element using the nvinfer config file. */
    g_object_set (G_OBJECT (pgie),
                  "config-file-path", "config_infer_primary_yoloV5.txt", NULL);

    /* Override the batch-size set in the config file with the number of sources. */
    g_object_get (G_OBJECT (pgie), "batch-size", &pgie_batch_size, NULL);
    if (pgie_batch_size != num_sources) {
        g_printerr
                ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
                 pgie_batch_size, num_sources);
        g_object_set (G_OBJECT (pgie), "batch-size", num_sources, NULL);
    }

    g_object_set (G_OBJECT (nvosd), "process-mode", OSD_PROCESS_MODE,
                  "display-text", OSD_DISPLAY_TEXT, NULL);

    g_object_set (G_OBJECT (sink), "qos", 0, NULL);

    /* we add a message handler */
    bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
    bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
    gst_object_unref (bus);

    /* Set up the pipeline
     * source-> nvvidconv_src -> tee_pre -> queue_tee -> caps_nvvidconv_src -> streammux -> pgie -> nvvidconv -> nvosd -> nvvidconv2 -> caps_filter -> video-renderer
     *                                  |-> queue_sr -> caps_sr -> encoder -> parser -> recordbin
     */
    gst_bin_add_many (GST_BIN (pipeline), source, nvvidconv_src, tee_pre,
                      queue_tee, caps_nvvidconv_src, streammux, pgie,
                      nvvidconv, nvosd, nvvidconv2, cap_filter, sink, tes_fake, NULL);

    if(prop.integrated) {
        gst_bin_add (GST_BIN (pipeline), transform);
    }

    /* Link the elements together till caps_nvvidconv_src */
    if (!gst_element_link_many (source, nvvidconv_src, tee_pre,
                                queue_tee, caps_nvvidconv_src, NULL)) {
        g_printerr ("Elements could not be linked: 1. Exiting.\n");
        return -1;
    }

//    gst_element_link_many (source, nvvidconv_src, tee_pre, NULL);

    /* Link decoder with streammux */
    GstPad *sinkpad, *srcpad;
    gchar pad_name_sink[16] = "sink_0";
    gchar pad_name_src[16] = "src";

    sinkpad = gst_element_get_request_pad (streammux, pad_name_sink);
    if (!sinkpad) {
        g_printerr ("Streammux request sink pad failed. Exiting.\n");
        return -1;
    }

    srcpad = gst_element_get_static_pad (caps_nvvidconv_src, pad_name_src);
    if (!srcpad) {
        g_printerr ("Decoder request src pad failed. Exiting.\n");
        return -1;
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
        g_printerr ("Failed to link decoder to stream muxer. Exiting.\n");
        return -1;
    }

    gst_object_unref (sinkpad);
    gst_object_unref (srcpad);

    /* Link the remaining elements of the pipeline to streammux */
    if(prop.integrated) {
        if (!gst_element_link_many (streammux, pgie,
                                    nvvidconv, nvosd, nvvidconv2, cap_filter,
                                    sink, NULL)) {
            g_printerr ("Elements could not be linked. Exiting.\n");
            return -1;
        }
    } else {
        if (!gst_element_link_many (streammux, pgie,
                                    nvvidconv, nvosd, nvvidconv2,
                                    cap_filter, sink, NULL)) {
            g_printerr ("Elements could not be linked. Exiting.\n");
            return -1;
        }
    }

//    gst_element_link_many (streammux, pgie, tes_fake, nvvidconv, nvosd, nvvidconv2, cap_filter, sink, NULL);

    /* Parameters are set before creating record bin
     * User can set additional parameters e.g recorded file path etc.
     * Refer NvDsSRInitParams structure for additional parameters
     */
    params.containerType = SMART_REC_CONTAINER;
    params.cacheSize = CACHE_SIZE_SEC;
    params.defaultDuration = SMART_REC_DEFAULT_DURATION;
    params.callback = smart_record_callback;
    params.fileNamePrefix = "SmartRecord";
//    params.fileNamePrefix = bbox_enabled ? "With_BBox" : "Without_BBox";

    if (NvDsSRCreate (&nvdssrCtx, &params) != NVDSSR_STATUS_OK) {
        g_printerr ("Failed to create smart record bin");
        return -1;
    }

    gst_bin_add_many (GST_BIN (pipeline), nvdssrCtx->recordbin, NULL);

    //--------------------------------------------------------------------------------

    /* for SR, after tee_pre */
    caps_sr_dat = gst_caps_from_string ("video/x-raw(memory:NVMM), format=NV12");
    caps_sr =
            gst_element_factory_make ("capsfilter", "caps-sr");
    g_object_set (G_OBJECT (caps_sr), "caps", caps_sr_dat, NULL);
    gst_caps_unref (caps_sr_dat);

    /* using hardware to accelerate */
    encoder_pre = gst_element_factory_make ("nvv4l2h264enc", "encoder-pre");
    g_object_set (G_OBJECT (encoder_pre), "maxperf-enable", 1, NULL);
    g_object_set (G_OBJECT (encoder_pre), "bitrate", 4000000, NULL);

    /* Parse the encoded data after osd component */

    parser_pre = gst_element_factory_make ("h264parse", "parser-pre");

    /* Use queue to connect the tee_post_osd to nvencoder */
    queue_sr = gst_element_factory_make ("queue", "queue-sr");

    tes_fake2 = gst_element_factory_make ("fakesink", "tes-fake2");

    gst_bin_add_many (GST_BIN (pipeline), queue_sr, caps_sr, encoder_pre,
                      parser_pre, tes_fake2, NULL);

//    gst_element_link_many(nvvidconv_src, queue_sr, caps_sr, tes_fake2, NULL); // 先把这个tee给堵上，看来是其他地方出问题了

    if (!gst_element_link_many (tee_pre, queue_sr, caps_sr, encoder_pre,
                                parser_pre, nvdssrCtx->recordbin, NULL)) {
        g_print ("Elements not linked. Exiting. \n");
        return -1;
    }
    //--------------------------------------------------------------------------------

    if (nvdssrCtx) {
        g_timeout_add (SMART_REC_INTERVAL * 1000, smart_record_event_generator,
                       nvdssrCtx);
    }

    /* Set the pipeline to "playing" state */
    g_print ("Now using csi camera as input\n");

    gst_element_set_state (pipeline, GST_STATE_PLAYING);

    /* Wait till pipeline encounters an error or EOS */
    g_print ("Running...\n");
    g_main_loop_run (loop);
    if (pipeline && nvdssrCtx) {
        if(NvDsSRDestroy (nvdssrCtx) != NVDSSR_STATUS_OK)
            g_printerr ("Unable to destroy recording instance\n");
    }
    /* Out of the main loop, clean up nicely */
    g_print ("Returned, stopping playback\n");
    gst_element_set_state (pipeline, GST_STATE_NULL);
    g_print ("Deleting pipeline\n");
    gst_object_unref (GST_OBJECT (pipeline));
    g_source_remove (bus_watch_id);
    g_main_loop_unref (loop);
    return 0;
}
