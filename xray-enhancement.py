import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the X-ray image
image = cv2.imread('xray_image.jpg', cv2.IMREAD_GRAYSCALE)

# Display the original X-ray image
plt.figure(figsize=(12, 6))
plt.subplot(2, 4, 1)
plt.imshow(image, cmap='gray')
plt.title('Original X-ray Image')
plt.axis('off')

# Filter 1: Sobel filter for edge detection
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_image = cv2.magnitude(sobel_x, sobel_y)
plt.subplot(2, 4, 2)
plt.imshow(sobel_image, cmap='gray')
plt.title('Sobel Filter')
plt.axis('off')

# Filter 2: Averaging filter for noise reduction
averaged_image = cv2.blur(image, (5, 5))
plt.subplot(2, 4, 3)
plt.imshow(averaged_image, cmap='gray')
plt.title('Averaging Filter')
plt.axis('off')

# Filter 3: Gamma correction for contrast enhancement
gamma_value = 1.5
gamma_corrected = np.uint8(cv2.pow(averaged_image / 255.0, gamma_value) * 255)
plt.subplot(2, 4, 4)
plt.imshow(gamma_corrected, cmap='gray')
plt.title('Gamma Correction')
plt.axis('off')

# Filter 4: Laplacian filter for edge enhancement
laplacian_image = cv2.Laplacian(image, cv2.CV_64F)
plt.subplot(2, 4, 5)
plt.imshow(laplacian_image, cmap='gray')
plt.title('Laplacian Filter')
plt.axis('off')

# Filter 5: Gaussian blur for smoothing
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)
plt.subplot(2, 4, 6)
plt.imshow(gaussian_blur, cmap='gray')
plt.title('Gaussian Blur')
plt.axis('off')

# Filter 6: High-pass filter for edge sharpening
highpass_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
highpass_image = cv2.filter2D(image, -1, highpass_kernel)
plt.subplot(2, 4, 7)
plt.imshow(highpass_image, cmap='gray')
plt.title('High-pass Filter')
plt.axis('off')

# Filter 7: Histogram equalization for contrast improvement
equalized_image = cv2.equalizeHist(image)
plt.subplot(2, 4, 8)
plt.imshow(equalized_image, cmap='gray')
plt.title('Histogram Equalization')
plt.axis('off')

plt.tight_layout()
plt.show()