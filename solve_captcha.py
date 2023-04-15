import numpy as np
import cv2
import pickle
import imutils
from imutils import paths
from keras.models import load_model
from utility import resize

MODEL_FILE = 'captcha_model.hdf5'
MODEL_LABELS_FILE = 'model_labels.dat'
CAPTCHAS = 'finessed_captchas'

with open(MODEL_LABELS_FILE, "rb") as f:
    lb = pickle.load(f)

model = load_model(MODEL_FILE)

captcha_files = list(paths.list_images(CAPTCHAS))
captcha_files = np.random.choice(captcha_files, size=(10,), replace=False)

for file in captcha_files:
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
    threshed_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

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

    output = cv2.merge([image] * 3)
    predictions = []

    for box in regions:
        x, y, w, h = box
        letter_img = image[y - 2:y + h + 2, x - 2:x + w + 2]
        letter_img = resize(letter_img, 20, 20)
        letter_img = np.expand_dims(letter_img, axis=2)
        letter_img = np.expand_dims(letter_img, axis=0)

        prediction = model.predict(letter_img)
        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)

        cv2.rectangle(output, (x-2, y-2), (x+w+4, y+h+4), (0, 0, 255), 1)
        cv2.putText(output, letter, (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

    text = "".join(predictions)

    cv2.imshow("Output", output)
    cv2.waitKey(0)


