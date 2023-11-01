import asyncio
import cv2
import websockets
import time
import logging

logging.basicConfig(level=logging.INFO)


async def video_stream(websocket, path):
    # Initialize video capture
    cap = None
    fps = 0
    frame_duration = 0
    width = 0
    height = 0
    
    # Check if the camera can be accessed or if a video file can be read
    try:
        gst_pipeline = (
            "nvarguscamerasrc sensor_id=0 ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)1280, height=(int)720, "
            "format=(string)NV12, framerate=(fraction)30/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
        )

        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

        if not cap.isOpened():
            logging.error("Cannot open camera or video file.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_duration = 1.0 / fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    except Exception as e:
        logging.error(f"Exception during video capture initialization: {e}")
        return

    start_time = time.time()
    flag = 0

    try:
        while True:
            if flag == 0:
                command = await websocket.recv()
            
            if command == 'start' or flag == 1:
                # Synchronize with the video's FPS
                expected_time = start_time + (cap.get(cv2.CAP_PROP_POS_FRAMES) * frame_duration)
                delay = expected_time - time.time()

                ret, frame = cap.read()
                if not ret:
                    logging.error("Failed to grab frame, exiting.")
                    break

                if delay < 0:
                    flag = 1
                    continue
                else:
                    await asyncio.sleep(delay)
                    flag = 0

                resized_frame = cv2.resize(frame, (480, 320))
                _, buffer = cv2.imencode('.jpg', resized_frame)
                await websocket.send(buffer.tobytes())

            elif command == 'stop':
                break

    except websockets.exceptions.ConnectionClosedOK:
        logging.info("Connection closed normally.")
    except websockets.exceptions.ConnectionClosedError:
        logging.warning("Connection lost. Attempting to reconnect...")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        if cap and cap.isOpened():
            cap.release()

if __name__ == '__main__':
    logging.info("Starting WebSocket server")
    start_server = websockets.serve(video_stream, "0.0.0.0", 7979)
    
    # Run the WebSocket server indefinitely
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
