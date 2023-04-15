import os
import os.path
import cv2
import glob
import imutils
from tqdm import tqdm
import matplotlib.pyplot as plt

IN_FOLDER = 'finessed_captchas'
OUT_FOLDER = 'finessed_letters'

letter_counts = {}
img_files = glob.glob(os.path.join(IN_FOLDER, '*'))

for i, file in enumerate(tqdm(img_files)):
    name = os.path.basename(file)
    captcha_answers = os.path.splitext(name)[0]

    img = cv2.imread(file)

    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey_img = cv2.copyMakeBorder(grey_img, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

    threshed_img = cv2.threshold(grey_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    contours = cv2.findContours(threshed_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = countours[1] if imutils.is_cv3() else contours[0]

    regions = []

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        ratio = w / h

        if ratio > 1.25:
            half = w // 2
            regions.append((x, y, half, h))
            regions.append((x+half, y, half, h))

        else:
            regions.append((x, y, w, h))

    if len(regions) != 4:
        continue

    regions = sorted(regions, key=lambda l: l[0])
    for box, text in zip(regions, captcha_answers):
        x, y, w, h = box
        letter_img = grey_img[y-2:y+h+2, x-2:x+w+2]

        out_path = os.path.join(OUT_FOLDER, text)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        cnt = letter_counts.get(text, 1)
        p = os.path.join(out_path, "{}.png".format(str(cnt).zfill(6)))
        cv2.imwrite(p, letter_img)

        letter_counts[text] = cnt + 1






