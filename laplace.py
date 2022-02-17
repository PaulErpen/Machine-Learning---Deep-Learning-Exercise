import cv2

def map_laplace_unshaped(img_data, dimensions):
    shaped = list(map(lambda img: img.reshape((dimensions, dimensions)), img_data))
    return map_laplace(shaped, dimensions)

def map_laplace_color(img_data, dimensions):
    grayscale = list(map(lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), img_data))
    return map_laplace(grayscale, dimensions)


def map_laplace(img_data, dimensions):
    laplace_data = []
    for image in img_data:
        blurred = cv2.GaussianBlur(image,(3,3),0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        final = laplacian.reshape((dimensions * dimensions))
        laplace_data.append(final)
    return laplace_data