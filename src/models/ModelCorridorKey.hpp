// SPDX-FileCopyrightText: 2026 vplab / Ling Tung University
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// ModelCorridorKey.hpp
// CorridorKey AI green screen keyer for obs-backgroundremoval
//
// Confirmed from ONNX inspection (ort 1.17.0, int8_512 model):
//   Input:  name="input_rgb_hint"  shape=[1,4,512,512]  float32  range [0,1]
//           Channel layout: [R, G, B, AlphaHint]  all in [0,1]
//   Output[0]: name="alpha"  shape=[1,1,512,512]  float32  range [0,1]  post-sigmoid
//   Output[1]: name="fg"     shape=[1,3,512,512]  float32  range [0,1]  post-sigmoid
//
// AlphaHint strategy: temporal feedback — use previous frame's alpha as next hint
//                     First frame: neutral 0.5

#ifndef MODELCORRIDORKEY_H
#define MODELCORRIDORKEY_H

#include "Model.h"

class ModelCorridorKey : public ModelBCHW {
private:
    cv::Mat prevAlpha;      // float32 [0,1], size = modelH x modelW
    int modelH = 512;
    int modelW = 512;

public:
    ModelCorridorKey() {}
    ~ModelCorridorKey() {}

    virtual void populateInputOutputNames(const std::unique_ptr<Ort::Session> &session,
                                          std::vector<Ort::AllocatedStringPtr> &inputNames,
                                          std::vector<Ort::AllocatedStringPtr> &outputNames)
    {
        Ort::AllocatorWithDefaultOptions allocator;
        inputNames.clear();
        outputNames.clear();
        for (size_t i = 0; i < session->GetInputCount(); i++)
            inputNames.push_back(session->GetInputNameAllocated(i, allocator));
        for (size_t i = 0; i < session->GetOutputCount(); i++)
            outputNames.push_back(session->GetOutputNameAllocated(i, allocator));
    }

    virtual bool populateInputOutputShapes(const std::unique_ptr<Ort::Session> &session,
                                           std::vector<std::vector<int64_t>> &inputDims,
                                           std::vector<std::vector<int64_t>> &outputDims)
    {
        inputDims.clear();
        outputDims.clear();

        // Input: [1, 4, H, W]
        for (size_t i = 0; i < session->GetInputCount(); i++) {
            auto shape = session->GetInputTypeInfo(i)
                             .GetTensorTypeAndShapeInfo().GetShape();
            if (!shape.empty() && shape[0] <= 0) shape[0] = 1;
            if (shape.size() >= 4) {
                shape[2] = 512;
                shape[3] = 512;
            }
            inputDims.push_back(shape);
            obs_log(LOG_INFO, "[CorridorKey] Input %zu: [%lld, %lld, %lld, %lld]",
                    i, shape[0], shape[1], shape[2], shape[3]);
        }

        // Detect model H/W
        if (!inputDims.empty() && inputDims[0].size() >= 4) {
            modelH = (int)inputDims[0][2];
            modelW = (int)inputDims[0][3];
            obs_log(LOG_INFO, "[CorridorKey] Model resolution: %d x %d", modelW, modelH);
        }

        // Output shapes hardcoded from confirmed model spec:
        // output[0] alpha: [1, 1, H, W]
        // output[1] fg:    [1, 3, H, W]
        int64_t outChannels[] = {1, 3};
        for (size_t i = 0; i < session->GetOutputCount(); i++) {
            int64_t ch = (i < 2) ? outChannels[i] : 1;
            std::vector<int64_t> shape = {1, ch, (int64_t)modelH, (int64_t)modelW};
            outputDims.push_back(shape);
            obs_log(LOG_INFO, "[CorridorKey] Output %zu: [1, %lld, %d, %d]",
                    i, ch, modelW, modelH);
        }

        return true;
    }

    virtual void getNetworkInputSize(const std::vector<std::vector<int64_t>> &inputDims,
                                     uint32_t &inputWidth, uint32_t &inputHeight)
    {
        if (!inputDims.empty() && inputDims[0].size() >= 4) {
            inputWidth  = (uint32_t)inputDims[0][3];
            inputHeight = (uint32_t)inputDims[0][2];
        } else {
            inputWidth  = 512;
            inputHeight = 512;
        }
    }

    // Input comes in as CV_32F [0,255] from ort-session-utils
    // CorridorKey needs float32 [0,1], CHW layout
    virtual void prepareInputToNetwork(cv::Mat &resizedImage, cv::Mat &preprocessedImage)
    {
        cv::Mat normalized;
        resizedImage.convertTo(normalized, CV_32FC3, 1.0 / 255.0);
        hwc_to_chw(normalized, preprocessedImage);
    }

    virtual void loadInputToTensor(const cv::Mat &preprocessedImage,
                                   uint32_t inputWidth, uint32_t inputHeight,
                                   std::vector<std::vector<float>> &inputTensorValues)
    {
        size_t rgbSize   = 3 * inputWidth * inputHeight;
        size_t totalSize = 4 * inputWidth * inputHeight;

        inputTensorValues[0].resize(totalSize, 0.0f);

        const float *src = preprocessedImage.ptr<float>(0);
        std::copy(src, src + rgbSize, inputTensorValues[0].begin());

        float *alphaPtr = inputTensorValues[0].data() + rgbSize;
        buildAlphaHint(inputWidth, inputHeight, alphaPtr);
    }

    virtual void assignOutputToInput(std::vector<std::vector<float>> &outputTensorValues,
                                     std::vector<std::vector<float>> &inputTensorValues)
    {
        UNUSED_PARAMETER(inputTensorValues);
        if (!outputTensorValues.empty() &&
            outputTensorValues[0].size() == (size_t)(modelH * modelW)) {
            prevAlpha = cv::Mat(modelH, modelW, CV_32FC1,
                                outputTensorValues[0].data()).clone();
        }
    }

    virtual cv::Mat getNetworkOutput(const std::vector<std::vector<int64_t>> &outputDims,
                                     std::vector<std::vector<float>> &outputTensorValues)
    {
        UNUSED_PARAMETER(outputDims);

        if (outputTensorValues.empty() || outputTensorValues[0].empty()) {
            obs_log(LOG_ERROR, "[CorridorKey] No output data");
            return cv::Mat();
        }

        size_t expected = (size_t)(modelH * modelW);
        if (outputTensorValues[0].size() != expected) {
            obs_log(LOG_ERROR, "[CorridorKey] Output size mismatch: got %zu expected %zu",
                    outputTensorValues[0].size(), expected);
            return cv::Mat();
        }

        cv::Mat alphaMat(modelH, modelW, CV_32FC1, outputTensorValues[0].data());
        prevAlpha = alphaMat.clone();

        float minVal, maxVal;
        cv::minMaxLoc(alphaMat, &minVal, &maxVal);
        obs_log(LOG_INFO, "[CorridorKey] Alpha: %dx%d  min=%.4f  max=%.4f",
                modelW, modelH, minVal, maxVal);

        cv::Mat result;
        alphaMat.convertTo(result, CV_8U, 255.0);
        return result;
    }

    virtual void postprocessOutput(cv::Mat &output)
    {
        UNUSED_PARAMETER(output);
        // Already CV_8U. ort-session-utils convertTo(CV_8U,255) is safe no-op.
    }

private:
    void buildAlphaHint(uint32_t width, uint32_t height, float *outPtr)
    {
        size_t pixelCount = (size_t)width * height;

        if (prevAlpha.empty()) {
            std::fill(outPtr, outPtr + pixelCount, 0.5f);
            obs_log(LOG_INFO, "[CorridorKey] First frame: neutral AlphaHint (0.5)");
        } else {
            cv::Mat resized;
            cv::resize(prevAlpha, resized,
                       cv::Size((int)width, (int)height), 0, 0, cv::INTER_LINEAR);
            cv::Mat resizedF;
            if (resized.depth() != CV_32F)
                resized.convertTo(resizedF, CV_32FC1, 1.0 / 255.0);
            else
                resizedF = resized;
            const float *src = resizedF.ptr<float>(0);
            std::copy(src, src + pixelCount, outPtr);
        }
    }
};

#endif // MODELCORRIDORKEY_H
