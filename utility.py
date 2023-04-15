import cv2
import imutils


def resize(image, width, height):
    (h, w) = image.shape[:2]

    if w > h:
        image = imutils.resize(image, width=width)
    else:
        image = imutils.resize(image,height=height)

    wpad = int((width - image.shape[1]) / 2.0)
    hpad = int((height - image.shape[0]) / 2.0)

    image = cv2.copyMakeBorder(image, hpad, hpad, wpad, wpad, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    return image
