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
// AlphaHint strategy:
//   Frame 0: HSV chroma key on green screen (H:32-56, S>80, V>60) → 0=green, 1=foreground
//   Frame 1+: temporal feedback from previous alpha output

#ifndef MODELCORRIDORKEY_H
#define MODELCORRIDORKEY_H

#include "Model.h"

class ModelCorridorKey : public ModelBCHW {
private:
    cv::Mat prevAlpha;      // float32 [0,1], H x W
    int modelH = 512;
    int modelW = 512;

    // Chroma key HSV range for this green screen
    // Confirmed from vplab green screen image analysis
    const int CK_H_LOW  = 32;
    const int CK_H_HIGH = 56;
    const int CK_S_LOW  = 80;
    const int CK_V_LOW  = 60;

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

        if (!inputDims.empty() && inputDims[0].size() >= 4) {
            modelH = (int)inputDims[0][2];
            modelW = (int)inputDims[0][3];
            obs_log(LOG_INFO, "[CorridorKey] Model resolution: %d x %d", modelW, modelH);
        }

        // Hardcode output shapes from confirmed model spec
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
            inputWidth = 512; inputHeight = 512;
        }
    }

    // resizedImage: CV_32F [0,255] HWC RGB — from ort-session-utils
    // CorridorKey needs float32 [0,1] CHW
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

        // Copy CHW RGB planes [0,1]
        const float *src = preprocessedImage.ptr<float>(0);
        std::copy(src, src + rgbSize, inputTensorValues[0].begin());

        // Build AlphaHint from CHW preprocessedImage
        float *alphaPtr = inputTensorValues[0].data() + rgbSize;
        buildAlphaHintFromCHW(src, inputWidth, inputHeight, alphaPtr);
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

    // Returns CV_8U [0,255]
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

        double minVal, maxVal;
        cv::minMaxLoc(alphaMat, &minVal, &maxVal);
        obs_log(LOG_INFO, "[CorridorKey] Alpha: %dx%d  min=%.4f  max=%.4f",
                modelW, modelH, (float)minVal, (float)maxVal);

        cv::Mat result;
        alphaMat.convertTo(result, CV_8U, 255.0);
        return result;
    }

    virtual void postprocessOutput(cv::Mat &output)
    {
        UNUSED_PARAMETER(output);
        // Already CV_8U from getNetworkOutput
    }

private:
    // Build AlphaHint from CHW float [0,1] RGB data
    // Strategy:
    //   - If prevAlpha exists: use temporal feedback (resize prev alpha)
    //   - If first frame: use HSV chroma key to detect green screen
    //     green pixels → 0.0 (background), non-green → 1.0 (foreground)
    void buildAlphaHintFromCHW(const float *chw, uint32_t width, uint32_t height, float *outPtr)
    {
        size_t pixelCount = (size_t)width * height;

        if (!prevAlpha.empty()) {
            // Temporal feedback: resize previous alpha
            cv::Mat resized;
            cv::resize(prevAlpha, resized,
                       cv::Size((int)width, (int)height), 0, 0, cv::INTER_LINEAR);
            cv::Mat resizedF;
            if (resized.depth() != CV_32F)
                resized.convertTo(resizedF, CV_32FC1, 1.0 / 255.0);
            else
                resizedF = resized;
            const float *s = resizedF.ptr<float>(0);
            std::copy(s, s + pixelCount, outPtr);
            return;
        }

        // First frame: HSV chroma key
        obs_log(LOG_INFO, "[CorridorKey] First frame: HSV chroma key AlphaHint");

        // Reconstruct HWC uint8 from CHW float [0,1]
        cv::Mat rgbFloat(height, width, CV_32FC3);
        float *dst = rgbFloat.ptr<float>(0);
        const float *rPlane = chw;
        const float *gPlane = chw + pixelCount;
        const float *bPlane = chw + pixelCount * 2;
        for (size_t i = 0; i < pixelCount; i++) {
            dst[i * 3 + 0] = rPlane[i];
            dst[i * 3 + 1] = gPlane[i];
            dst[i * 3 + 2] = bPlane[i];
        }

        // Convert to uint8 BGR for OpenCV HSV conversion
        cv::Mat rgbUint8, bgrUint8, hsvMat;
        rgbFloat.convertTo(rgbUint8, CV_8UC3, 255.0);
        cv::cvtColor(rgbUint8, bgrUint8, cv::COLOR_RGB2BGR);
        cv::cvtColor(bgrUint8, hsvMat, cv::COLOR_BGR2HSV);

        // Green screen mask: green=0 (background), non-green=1 (foreground)
        // vplab green screen: H=32-56, S>80, V>60
        cv::Mat greenMask;
        cv::inRange(hsvMat,
                    cv::Scalar(CK_H_LOW, CK_S_LOW, CK_V_LOW),
                    cv::Scalar(CK_H_HIGH, 255, 255),
                    greenMask); // 255 = green, 0 = not green

        // Slight blur to soften edges
        cv::GaussianBlur(greenMask, greenMask, cv::Size(5, 5), 1.5);

        // Write to output: green→0.0, non-green→1.0
        const uint8_t *maskPtr = greenMask.ptr<uint8_t>(0);
        for (size_t i = 0; i < pixelCount; i++) {
            outPtr[i] = 1.0f - (maskPtr[i] / 255.0f);
        }
    }
};

#endif // MODELCORRIDORKEY_H
