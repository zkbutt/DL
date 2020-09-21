import cv2

img = cv2.imread('test1.jpg')
print(type(img), img.shape)
img = cv2.imread('test2.png', -1)
img = cv2.resize(img, (500, 500))
print(type(img), img.shape)
cv2.circle(img, (200, 200), 200, (0, 0, 255, 255), -1)

cv2.imshow('title', img)
cv2.waitKey()
