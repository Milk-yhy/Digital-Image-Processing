import cv2
from matplotlib import pyplot as plt

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def plt_show0(img):
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.show()

def plt_show(img):
    plt.imshow(img, cmap='gray')
    plt.show()
img = cv2.imread('./image/test1.png')
plt_show0(img)
height, weight = img.shape[0:2]
list_ = ['豫', 'A', '0', '4', 'S', '8', '9']
image = img.copy()
cv2.rectangle(image, (int(0.2 * weight), int(0.75 * height)), (int(weight * 0.8), int(height * 0.95)), (0, 255, 0), 5)
cv2.putText(image, "".join(list_), (int(0.2 * weight) + 30, int(0.75 * height) + 80), cv2.FONT_HERSHEY_COMPLEX, 3,(0, 255, 0), 12)
plt_show0(image)

