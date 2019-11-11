#include <iostream>
#include <vector>

#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>


at::Tensor normalizeImage(int imgHeight, int imgWidth)
{
    const cv::String fileName = "../img.jpg";
    cv::Mat img = cv::imread(fileName, cv::IMREAD_UNCHANGED);
    cv::Size rsz = { imgHeight, imgwidth };

    cv::resize( img, img, rsz, 0, 0, CV_INTER_LINEAR );
    img.convertTo( img, CV_32FC3, 1/255.0 );

    at::Tensor tensorImage = torch::from_blob( img.data, { 1, img.rows, img.cols, 3 }, at::kFloat );

    tensorImage = tensorImage.permute({ 0, 3, 1, 2 });

    //  Normalize data
    tensorImage[0][0] = tensorImage[0][0].sub(0.485).div(0.229);
    tensorImage[0][1] = tensorImage[0][1].sub(0.456).div(0.224);
    tensorImage[0][2] = tensorImage[0][2].sub(0.406).div(0.225);

    std::vector<torch::jit::IValue> input;
    input.push_back(tensorImage);

    //  Normalize Image here
    // torch::data::transforms::Normalize<>(0.1307, 0.3081)}
    // for (auto& t : inputs) {
	// t = t.toTensor().sub(0.5).div(0.5);
    
}