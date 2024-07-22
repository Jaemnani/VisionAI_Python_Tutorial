import cv2

path = "./images/FoodSeg103/train/00000001.jpg"
img = cv2.imread(path)

cv2.imwrite("./test.jpg", img)

cv2.imshow("window_name", img)
cv2.waitKey(0)

print(img.shape)
print(img.dtype)
img2 = img.astype(int)
print(img2.dtype)
img3 = img.astype(float)
print(img3.dtype)

new_image = img[::2, ::2]
cv2.imwrite("test2.jpg", new_image)

print("done")