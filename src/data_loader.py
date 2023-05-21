from os import listdir

import cv2


class DataLoader:

    def __init__(self, image_path: str, dest_folder: str = None, url: str = None, isLocal: bool = True):
        self.url = url
        self.dest_folder = dest_folder
        self.isLocal = isLocal
        self.image_path = image_path

    def load_image(self):
        if not self.isLocal:
            if not listdir(self.dest_folder):  # TODO: Add algorithm for downloading image from google drive
                pass
            else:
                pass
        else:
            image = cv2.imread(self.image_path)

            return image



