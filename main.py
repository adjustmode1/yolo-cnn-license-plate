import cv2
from pathlib import Path

from src.lp_recognition import E2E



img_path = Path('./samples/0207_07240_b.jpg')
# read image
img = cv2.imread(str(img_path))

# load model
model = E2E()

# recognize license plate
image = model.predict(img)


# show image
cv2.imshow('License Plate', image)
if cv2.waitKey(0) & 0xFF == ord('q'):
    exit(0)


cv2.destroyAllWindows()
