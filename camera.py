from typing import Any

import pyrealsense2 as rs
import numpy as np
import numpy.typing as npt
import cv2
import logging

from cv2 import Mat


#Basic camera interface that can be extended to use different API-s. Realsense example below
class ICamera:
    def open(self) -> None:
        raise NotImplementedError
    def close(self) -> None:
        raise NotImplementedError
    def has_depth_capability(self) -> bool:
        raise NotImplementedError
    def get_color_frame(self) -> cv2.typing.MatLike:
        raise NotImplementedError
    def get_frames(self, aligned = False) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
        raise NotImplementedError


# Camera implementation using the pyrealsense2 provided API 
class RealsenseCamera(ICamera):
    logger = logging.getLogger(__name__)

    def __init__(self,
                 rgb_width: int = 848,
                 rgb_height: int = 480,
                 rgb_fps: int = 60,
                 depth_width: int = 848,
                 depth_height: int = 480,
                 depth_fps: int = 60,
                 exposure: int = 50,
                 white_balance: int = 3500,
                 depth_enabled:bool = True):

        self.rgb_width = rgb_width
        self.rgb_height = rgb_height
        self.rgb_fps = rgb_fps
        self.exposure = exposure
        self.white_balance = white_balance

        self.depth_width = depth_width
        self.depth_height = depth_height
        self.depth_fps = depth_fps

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.rgb_width, self.rgb_height, rs.format.bgr8, self.rgb_fps)
        
        self.depth_enabled = depth_enabled
        if self.depth_enabled:
            self.config.enable_stream(rs.stream.depth, self.depth_width, self.depth_height, rs.format.z16, self.depth_fps)
            
        self.align = rs.align(rs.stream.color)
        self.depth_scale = -1

    def open(self) -> None:
        self.logger.info("Configuring and opening RealSense camera")
        profile = self.pipeline.start(self.config)
        color_sensor = profile.get_device().query_sensors()[1]
        color_sensor.set_option(rs.option.enable_auto_exposure, False)
        color_sensor.set_option(rs.option.enable_auto_white_balance, False)
        color_sensor.set_option(rs.option.white_balance, self.white_balance)
        color_sensor.set_option(rs.option.exposure, self.exposure)

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.logger.info("Opened camera")

    def close(self) -> None:
        self.logger.info("Closing camera")
        self.pipeline.stop()
        self.logger.info("Closed camera")
    
    def get_color_frame(self)-> cv2.typing.MatLike:
        frames = self.pipeline.wait_for_frames()
        return np.asanyarray(frames.get_color_frame().get_data())
    
    def has_depth_capability(self) -> bool:
        return self.depth_enabled

    def get_frames(self, aligned = False) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
        frames = self.pipeline.wait_for_frames()
        if aligned:
            frames = self.align.process(frames)
        return np.asanyarray(frames.get_color_frame().get_data()), np.asanyarray(frames.get_depth_frame().get_data())


# resolution numbers are sensitive with OpenCV. Implement a resolution setting mechanism here
# or use the default of the webcam to get a more robust solution
class OpenCVCamera(ICamera):
    logger = logging.getLogger(__name__)

    def __init__(self,
                 rgb_width: int = 1920,
                 rgb_height: int = 1080,
                 rgb_fps: int = 30,
                 cam_id: int = 0):

        self.rgb_width = rgb_width
        self.rgb_height = rgb_height
        self.rgb_fps = rgb_fps

        self.camera_id = cam_id
        self.camera_stream: cv2.VideoCapture | None = None

    def open(self) -> None:
        self.logger.info(f"Opening camera: {self.camera_id}")
        self.camera_stream = cv2.VideoCapture(self.camera_id)
        self.logger.info("Opened camera")


    def close(self) -> None:
        assert self.camera_stream, "No camera to close"
        self.logger.info(f"Closing camera: {self.camera_id}")
        self.camera_stream.release()
        self.logger.info("Closed camera")


    def has_depth_capability(self) -> bool:
        return False

    def get_color_frame(self) -> cv2.typing.MatLike:
        assert self.camera_stream
        _, frame = self.camera_stream.read()
        return frame


    def get_frames(self, aligned = False) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
        assert self.camera_stream
        _, frame = self.camera_stream.read()
        return frame, np.zeros(frame.shape, dtype=int)