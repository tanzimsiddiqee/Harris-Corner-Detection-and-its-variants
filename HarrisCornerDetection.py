import os, math, csv, cv2, glob
import gif2numpy, numpy as np
from matplotlib import pyplot as plt


def GaussianKernel(kernel_size, sigma=1, production=False):
    kernel_1D = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    for i in range(kernel_size):
        kernel_1D[i] = 1 / (np.sqrt(2 * np.pi) * sigma) * np.e ** (-np.power(kernel_1D[i] / sigma, 2) / 2)

    kernel_2D = np.outer(kernel_1D, kernel_1D)
    kernel_2D *= 1.0 / kernel_2D.max()

    plt.imshow(kernel_2D, interpolation='none', cmap='gray')
    plt.title("Gaussian Kernel Image")

    if production:
        plt.savefig(os.path.join("output", f"gaussian_{kernel_size}_{sigma}.png"))
    else: 
        plt.show()
    return kernel_2D

def BoxKernel(kernel_size, production=False):
    kernel = np.ones((kernel_size, kernel_size)) / 9

    plt.imshow(kernel, interpolation='none', cmap='gray')
    plt.title("Box Kernel Image")

    if production:
        plt.savefig(os.path.join("output", f"box_{kernel_size}.png"))
    else:
        plt.show()
    return kernel

def ImageFilter(image, kernel, average=False, production=False):
    if not production:
        plt.imshow(image, cmap='gray')
        plt.title("Image")
        plt.show()

    imageRow, imageCol = image.shape
    kernelRow, kernelCol = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernelRow - 1) / 2)
    pad_width = int((kernelCol - 1) / 2)

    padded_image = np.zeros((imageRow + (2 * pad_height), imageCol + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    if not production:
        plt.imshow(padded_image, cmap='gray')
        plt.title("Padded Image")
        plt.show()

    for row in range(imageRow):
        for col in range(imageCol):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernelRow, col:col + kernelCol])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]
                
    if not production:
        plt.imshow(output, cmap='gray')
        plt.title("Output Image using {}X{} Kernel".format(kernelRow, kernelCol))
        plt.show()

    return output

def scaleImage(img, scale):
    h, w = img.shape
    scale_img = cv2.resize(img, dsize=(int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
    return scale_img

def saveCornerResult(filename, output_path, corner_list, corner_img):
    with open(os.path.join(output_path, f"{filename}_corners.csv"), 'w') as corner_file:
        writer = csv.DictWriter(corner_file, fieldnames=["x", "y", "r"])
        writer.writeheader()
        for i in range(len(corner_list)):
            writer.writerow({
                "x": str(corner_list[i][0]),
                "y": str(corner_list[i][1]),
                "r": str(corner_list[i][2])
            })

    if corner_img is not None:
        cv2.imwrite(os.path.join(output_path, f"{filename}.png"), corner_img)

def myImgConvolve(Input_image, kernel, scale=1.0, production=False):
    filename, file_extension = os.path.splitext(Input_image)
    if file_extension == ".gif" :
        np_images, extensions, image_specs = gif2numpy.convert(Input_image)
        input_img = np_images[0]
    else:
        input_img = cv2.imread(Input_image)

    grayImg = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    if scale > 0 and scale != 1:
        grayImg = scaleImage(grayImg, scale)
    output_path = None

    if production:
        filename = Input_image.split(os.path.sep, 2)[-1]
        dynamic_dir = filename.replace('.', '_')
        output_path = os.path.join("output", dynamic_dir)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    else:
        plt.title("Input Image")
        plt.imshow(grayImg, cmap='gray')
        plt.show()

    Gxy = ImageFilter(grayImg, kernel, average=True, production=production)
    Gy = np.diff(Gxy, axis=0, append=0)
    plt.title("y derivative image")
    plt.imshow(Gy, cmap='gray')
    if production:
        plt.savefig(os.path.join(output_path, f"gy.png"))    
    else:
        plt.show()
        

    Gx = np.diff(Gxy, axis=1, append=0)
    plt.title("x derivative image")
    plt.imshow(Gx, cmap='gray')
    if production:
        plt.savefig(os.path.join(output_path, f"gx.png"))  
    else:
        plt.show()   

    window_size = 5

    harris_corners = []
    harris_output = cv2.cvtColor(grayImg.copy(), cv2.COLOR_GRAY2RGB)

    kanade_corners = []
    kanade_output = cv2.cvtColor(grayImg.copy(), cv2.COLOR_GRAY2RGB)

    nobel_corners = []
    nobel_output = cv2.cvtColor(grayImg.copy(), cv2.COLOR_GRAY2RGB)

    offset = int(window_size / 2)
    y_range = grayImg.shape[0] - offset
    x_range = grayImg.shape[1] - offset

    Ixx = Gx ** 2
    Ixy = Gy * Gx
    Iyy = Gy ** 2

    for y in range(offset, y_range):
        for x in range(offset, x_range):
            start_y = y - offset
            end_y = y + offset + 1
            start_x = x - offset
            end_x = x + offset + 1

            windowIxx = Ixx[start_y: end_y, start_x: end_x]
            windowIxy = Ixy[start_y: end_y, start_x: end_x]
            windowIyy = Iyy[start_y: end_y, start_x: end_x]

            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            M = np.array([[Sxx, Sxy], [Sxy, Syy]])

            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy

            #Harris & Stephens
            k = 0.04
            r = det - k * (trace ** 2)
            threshold = 10000.00
            if r > threshold:
                harris_corners.append([x, y, r])
                harris_output[y, x] = (0, 0, 255)

            #Kanade & Tomasi
            w, v = np.linalg.eig(M)
            r = np.min(w)
            threshold = 1.00
            if r > threshold:
                kanade_corners.append([x, y, r])
                kanade_output[y, x] = (0, 0, 255)

            #Nobel
            e = 1
            r = det / (trace + e)
            threshold = 100.00
            if r > threshold:
                nobel_corners.append([x, y, r])
                nobel_output[y, x] = (0, 0, 255)

    if production:     
        saveCornerResult("harris", output_path, harris_corners, harris_output)
        saveCornerResult("kanade", output_path, kanade_corners, kanade_output)
        saveCornerResult("nobel", output_path, nobel_corners, nobel_output)

        # cv2.imwrite(os.path.join(output_path, f"harris.png"), harris_output)
        # cv2.imwrite(os.path.join(output_path, f"kanade.png"), kanade_output)
        # cv2.imwrite(os.path.join(output_path, f"nobel.png"), nobel_output)
    else:
        plt.title("Harris Output Image")
        plt.imshow(harris_output, cmap='gray')
        plt.show()

        plt.title("Kanade & Tomasi Output Image")
        plt.imshow(kanade_output, cmap='gray')
        plt.show()

        plt.title(f"Nobel Output Image")
        plt.imshow(nobel_output, cmap='gray')
        plt.show()
        

if __name__ == "__main__":
    kernel = GaussianKernel(kernel_size=5, sigma=2, production=True)
    #kernel = BoxKernel(kernel_size=5, production=True)
    files = glob.glob(os.path.join("images", "MEHEDI HASAN siddiqee*"))
    for image in files:
        myImgConvolve(image, kernel, scale=1.0, production=True)
