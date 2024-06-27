#include "opencv2/opencv_modules.hpp"

#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include <cuda_runtime.h>

#include "DxApplication.h"

int main(int argc, const char* argv[])
{
	if (argc != 2)
		return -1;

	const std::string fname(argv[1]);

	cv::TickMeter tm;
	cv::Mat frame;

	cv::cuda::GpuMat d_frame_8u, d_frame_32f;
	cv::Ptr<cv::cudacodec::VideoReader> d_reader = cv::cudacodec::createVideoReader(fname);

	double width, height;
	d_reader->get(cv::CAP_PROP_FRAME_WIDTH, width);
	d_reader->get(cv::CAP_PROP_FRAME_HEIGHT, height);
	cv::cudacodec::FormatInfo formatInfo = d_reader->format();

	void* cuda_in_frame, * cuda_out_frame;
	size_t pitch;
	cudaMallocPitch(&cuda_in_frame, &pitch, width * sizeof(float4), height);
	cudaMallocPitch(&cuda_out_frame, &pitch, width * sizeof(float4), height);

	cv::cuda::Stream stream;
	auto cuStream = cv::cuda::StreamAccessor::getStream(stream);

	DxApplication dxApp(width, height);
	dxApp.OnInit();
	dxApp.SetPtr(cuda_in_frame, cuda_out_frame, cuStream);

	for (;;)
	{
		auto start = std::chrono::system_clock::now();

		if (!d_reader->nextFrame(d_frame_8u, stream))
			break;

		d_frame_8u.convertTo(d_frame_32f, CV_32FC4, stream);
		cudaMemcpy2DAsync(cuda_in_frame, pitch, d_frame_32f.ptr(), d_frame_32f.step, d_frame_32f.cols * sizeof(float4), d_frame_32f.rows, cudaMemcpyDeviceToDevice, cuStream);

		dxApp.OnRender();

		cudaMemcpy2DAsync(d_frame_32f.ptr(), d_frame_32f.step, cuda_out_frame, pitch, width * sizeof(float4), height, cudaMemcpyDeviceToDevice, cuStream);
		d_frame_32f.convertTo(d_frame_8u, CV_8UC4, stream);

		d_frame_8u.download(frame);
		cv::imshow("GPU", frame);

		auto end = std::chrono::system_clock::now();
		auto elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.f;
		std::cout << "elapsed: " << elapsed_ms << " msec" << std::endl;

		if (cv::waitKey(1) > 0)
			break;
	}

	dxApp.OnDestroy();

	cudaFree(cuda_in_frame);
	cudaFree(cuda_out_frame);

	return 0;
}