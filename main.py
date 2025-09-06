import image_processor
import camera
import motion
import cv2
import time
import logging

# Configure your loggers here, either in code or via a config file.
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main_loop():
    debug = True
    
    motion_sim = motion.PrintingMotion()
    
    #camera instance for normal web cameras
    # cam = camera.OpenCVCamera(cam_id = 0)
    # camera instance for realsense cameras
    cam = camera.RealsenseCamera(exposure = 100)
    
    processor = image_processor.ImageProcessor(cam, debug=debug)

    processor.start()
    motion_sim.open()

    start = time.time()
    frame = 0
    frame_cnt = 0
    try:
        while True:
            # has argument aligned_depth that enables depth frame to color frame alignment. Costs performance
            frame_data = processor.process_frame(aligned_depth=False)

            # This is where you add the driving behaviour of your robot. It should be able to filter out
            # objects of interest and calculate the required motion for reaching the objects

            cv2.imshow('BGR Frame', frame_data)

            k = cv2.waitKey(1) & 0xff
            if k == ord('q'):
                break

            frame_cnt +=1

            frame += 1
            if frame % 30 == 0:
                frame = 0
                end = time.time()
                fps = 30 / (end - start)
                start = end
                logger.info(f"FPS: {fps}, frame count: {frame_cnt}")
    except KeyboardInterrupt:
        logger.info("Closing....")
    finally:
        cv2.destroyAllWindows()
        processor.stop()
        motion_sim.close()

main_loop()
