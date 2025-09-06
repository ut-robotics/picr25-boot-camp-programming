import cv2
import segment
import _pickle as pickle
import numpy as np
import numpy.typing as npt
from camera import RealsenseCamera, OpenCVCamera


#Main processor class. processes segmented information
class ImageProcessor():
    def __init__(self, cam: RealsenseCamera | OpenCVCamera, color_config: str = "colors/colors.pkl", debug: bool = False) -> None:
        self.camera = cam

        self.color_config = color_config
        with open(self.color_config, 'rb') as conf:
            self.colors_lookup = pickle.load(conf)
            self.set_segmentation_table(self.colors_lookup)

        # Initialized matrix containing complete segmented color data. Populated by segment.segment(...)
        self.fragmented_frame	= np.zeros((self.camera.rgb_height, self.camera.rgb_width), dtype=np.uint8)

        # Initialized matrices containing specific colors segmented data. Populated by segment.segment(...)
        self.balls_frame = np.zeros((self.camera.rgb_height, self.camera.rgb_width), dtype=np.uint8)
        self.basket_blue_frame = np.zeros((self.camera.rgb_height, self.camera.rgb_width), dtype=np.uint8)
        self.basket_magenta_frame = np.zeros((self.camera.rgb_height, self.camera.rgb_width), dtype=np.uint8)

        self.debug = debug
        self.debug_frame = np.zeros((self.camera.rgb_height, self.camera.rgb_width), dtype=np.uint8)

    @staticmethod
    def set_segmentation_table(table: npt.NDArray) -> None:
        # Sets the global color segmentation table state
        segment.set_table(table)

    def start(self) -> None:
        self.camera.open()

    def stop(self) -> None:
        self.camera.close()

    def get_frame_data(self, aligned_depth: bool = False) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
        if self.camera.has_depth_capability():
            return self.camera.get_frames(aligned = aligned_depth)
        else:
            return self.camera.get_color_frame(), np.zeros((self.camera.rgb_height, self.camera.rgb_width), dtype=np.uint8)

    # TODO: Implement processing that does something with the camera data. Currently it retuns the BGR frame
    def process_frame(self, aligned_depth: bool = False) -> cv2.typing.MatLike:
        # Gets the color and depth frame information from the camera
        color_frame, _ = self.get_frame_data(aligned_depth = aligned_depth)

        segment.segment(color_frame, self.fragmented_frame, self.balls_frame, self.basket_magenta_frame, self.basket_blue_frame)

        return color_frame
