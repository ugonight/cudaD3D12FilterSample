#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdio.h>

#include "DxApplication.h"

using namespace cv;
using namespace std;

int main(int argc, const char* argv[])
{
	Mat frame;
	VideoCapture cap;
	int deviceID = 6;
	int apiID = cv::CAP_ANY;
	cap.open(deviceID, apiID);
	if (!cap.isOpened()) {
		cerr << "ERROR! Unable to open camera\n";
		return -1;
	}

	const int width = 1920;
	const int height = 1080;

	cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);

	Mat in_frame(height, width, CV_32FC4), out_frame(height, width, CV_32FC4);

	void* cuda_in_frame, * cuda_out_frame;
	cudaMalloc(&cuda_in_frame, width * height * sizeof(float4));
	cudaMalloc(&cuda_out_frame, width * height * sizeof(float4));
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	DxApplication dxApp(width, height);
	dxApp.OnInit();
	dxApp.SetPtr(cuda_in_frame, cuda_out_frame, stream);

	for (;;)
	{
		cap.read(frame);
		if (frame.empty()) {
			cerr << "ERROR! blank frame grabbed\n";
			break;
		}

		frame.convertTo(in_frame, CV_32FC4);
		cvtColor(in_frame, in_frame, cv::COLOR_BGR2BGRA);
		in_frame /= 255.0f;
		cudaMemcpy2DAsync(cuda_in_frame, width * sizeof(float4), in_frame.data, width * sizeof(float4), width * sizeof(float4), height, cudaMemcpyHostToDevice, stream);

		dxApp.OnRender();

		cudaMemcpy2DAsync(out_frame.data, width * sizeof(float4), cuda_out_frame, width * sizeof(float4), width * sizeof(float4), height, cudaMemcpyDeviceToHost, stream);
		cudaStreamSynchronize(stream);

		imshow("Live", out_frame);
		if (waitKey(1) >= 0)
			break;
	}

	dxApp.OnDestroy();

	cudaFree(cuda_in_frame);
	cudaFree(cuda_out_frame);

	return 0;
}