import cv2
from Stitch import Stitch

if __name__ == "__main__":
    image_left = cv2.imread('image/left_01.png')
    image_right = cv2.imread('image/right_01.png')

    stitch = Stitch()

    result = stitch.stitch([image_left, image_right], show_matches=False)