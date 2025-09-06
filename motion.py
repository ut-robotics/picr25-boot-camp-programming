import logging

class IRobotMotion:
    def open(self) -> None:
        raise NotImplementedError
    def close(self) -> None:
        raise NotImplementedError
    def move(self, x_speed: float, y_speed: float, rot_speed: float) -> None:
        raise NotImplementedError

class PrintingMotion(IRobotMotion):
    logger = logging.getLogger(__name__)

    ## change polarity if direction is not correct
    def __init__(self, polarity=1):
        self.polarity = polarity
    
    def open(self) -> None:
        self.logger.info("Starting up!")

    def close(self) -> None:
        self.logger.info("Shutting down...")

    # simple logic to print what the mainboard would receive
    def move(self, x_speed: float, y_speed: float, rot_speed: float) -> None:
        direction = "left" if rot_speed * self.polarity > 0 else "right"
        self.logger.info(f"Rotation direction: {direction!r};")
