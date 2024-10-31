#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

__global__ void resizeKernel(unsigned char* grayImg, unsigned char* colorImg, int lowWidth, int lowHeight, int highWidth, int highHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < highWidth && y < highHeight) {
        float lowX = (float)x * lowWidth / highWidth;
        float lowY = (float)y * lowHeight / highHeight;

        int x1 = floor(lowX);
        int y1 = floor(lowY);
        int x2 = min(x1 + 1, lowWidth - 1);
        int y2 = min(y1 + 1, lowHeight - 1);

        float a = lowX - x1;
        float b = lowY - y1;

        unsigned char grayValue = (1 - a) * (1 - b) * grayImg[y1 * lowWidth + x1] +
                                  a * (1 - b) * grayImg[y1 * lowWidth + x2] +
                                  (1 - a) * b * grayImg[y2 * lowWidth + x1] +
                                  a * b * grayImg[y2 * lowWidth + x2];

        // Simple colorization
        int channel = 0; // Blue channel
        colorImg[(y * highWidth + x) * 3 + channel] = static_cast<unsigned char>(grayValue * 0.5f); // Blue

        channel = 1; // Green channel
        colorImg[(y * highWidth + x) * 3 + channel] = static_cast<unsigned char>(grayValue * 0.7f); // Green

        channel = 2; // Red channel
        colorImg[(y * highWidth + x) * 3 + channel] = grayValue; // Red
    }
}

void resizeAndColorizeCUDA(const Mat& input, Mat& output) {
    int lowWidth = input.cols;
    int lowHeight = input.rows;
    int highWidth = lowWidth * 2;  // Change this for different scaling
    int highHeight = lowHeight * 2; // Change this for different scaling

    output.create(highHeight, highWidth, CV_8UC3); // Create a 3-channel image

    unsigned char* d_grayImg;
    unsigned char* d_colorImg;
    cudaMalloc(&d_grayImg, lowWidth * lowHeight);
    cudaMalloc(&d_colorImg, highWidth * highHeight * 3);

    cudaMemcpy(d_grayImg, input.data, lowWidth * lowHeight, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((highWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (highHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

    resizeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_grayImg, d_colorImg, lowWidth, lowHeight, highWidth, highHeight);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data, d_colorImg, highWidth * highHeight * 3, cudaMemcpyDeviceToHost);

    cudaFree(d_grayImg);
    cudaFree(d_colorImg);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <path_to_image>" << endl;
        return -1;
    }

    Mat img = imread(argv[1], IMREAD_GRAYSCALE); // Load as grayscale
    if (img.empty()) {
        cout << "Cannot load image: " << argv[1] << endl;
        return -1;
    }

    Mat colorizedImg;
    resizeAndColorizeCUDA(img, colorizedImg);

    imshow("Original Grayscale Image", img);
    imshow("Colorized Upscaled Image", colorizedImg);

    waitKey(0);
    return 0;
}
