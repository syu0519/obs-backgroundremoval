// SPDX-FileCopyrightText: 2026 vplab / Ling Tung University
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// ModelCorridorKey.hpp
// CorridorKey AI green screen keyer for obs-backgroundremoval
// Based on CorridorKey Runtime (https://github.com/alexandremendoncaalvaro/CorridorKey-Runtime)
//
// Input:  BCHW, 4 channels (RGB + AlphaHint), float32, range [0,1]
// Output: 2 tensors - alpha matte + foreground color
// AlphaHint strategy: use previous frame's alpha output as next frame's hint
//                     (temporal feedback loop, similar to GreenKiller's approach)

#ifndef MODELCORRIDORKEY_H
#define MODELCORRIDORKEY_H

#include "Model.h"

class ModelCorridorKey : public ModelBCHW {
private:
    // Store previous alpha for temporal AlphaHint feedback
    cv::Mat prevAlpha;
    int modelResolution = 512; // default, will be updated from model shape

public:
    ModelCorridorKey() {}
    ~ModelCorridorKey() {}

    // CorridorKey has 2 inputs: plate (RGB) + alpha_hint
    // and 2 outputs: alpha + foreground
    virtual void populateInputOutputNames(const std::unique_ptr<Ort::Session> &session,
                                          std::vector<Ort::AllocatedStringPtr> &inputNames,
                                          std::vector<Ort::AllocatedStringPtr> &outputNames)
    {
        Ort::AllocatorWithDefaultOptions allocator;
        inputNames.clear();
        outputNames.clear();

        // CorridorKey has 2 inputs: plate + alpha_hint
        for (size_t i = 0; i < session->GetInputCount(); i++) {
            inputNames.push_back(session->GetInputNameAllocated(i, allocator));
        }
        // CorridorKey has 2 outputs: alpha + foreground
        for (size_t i = 0; i < session->GetOutputCount(); i++) {
            outputNames.push_back(session->GetOutputNameAllocated(i, allocator));
        }
    }

    virtual bool populateInputOutputShapes(const std::unique_ptr<Ort::Session> &session,
                                           std::vector<std::vector<int64_t>> &inputDims,
                                           std::vector<std::vector<int64_t>> &outputDims)
    {
        inputDims.clear();
        outputDims.clear();

        // Read actual input shapes from model
        for (size_t i = 0; i < session->GetInputCount(); i++) {
            const Ort::TypeInfo inputTypeInfo = session->GetInputTypeInfo(i);
            const auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
            auto shape = inputTensorInfo.GetShape();
            // Fix dynamic dims (-1) to 1
            for (auto &d : shape) {
                if (d <= 0) d = 1;
            }
            inputDims.push_back(shape);
        }

// Read actual output shapes from model
        // Use input H/W to fix dynamic output dims (CUDA EP reports -1 for H/W)
        int64_t knownH = 512, knownW = 512;
        if (!inputDims.empty() && inputDims[0].size() >= 4) {
            knownH = (inputDims[0][2] > 0) ? inputDims[0][2] : 512;
            knownW = (inputDims[0][3] > 0) ? inputDims[0][3] : 512;
        }
        for (size_t i = 0; i < session->GetOutputCount(); i++) {
            const Ort::TypeInfo outputTypeInfo = session->GetOutputTypeInfo(i);
            const auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
            auto shape = outputTensorInfo.GetShape();
            for (size_t j = 0; j < shape.size(); j++) {
                if (shape[j] <= 0) {
                    if (shape.size() == 4) {
                        if (j == 0) shape[j] = 1;
                        else if (j == 1) shape[j] = 1;
                        else if (j == 2) shape[j] = knownH;
                        else if (j == 3) shape[j] = knownW;
                    } else {
                        shape[j] = 1;
                    }
                }
            }
            outputDims.push_back(shape);
        }

        // Detect model resolution from input shape (BCHW → dim[2] = H, dim[3] = W)
        if (!inputDims.empty() && inputDims[0].size() >= 4) {
            modelResolution = (int)inputDims[0][2]; // H
            obs_log(LOG_INFO, "[CorridorKey] Model resolution detected: %d x %d",
                    (int)inputDims[0][3], modelResolution);
        }

        // If model only has 1 input slot (4ch BCHW merged),
        // we handle AlphaHint by inserting it as channel 3
        if (inputDims.size() == 1) {
            obs_log(LOG_INFO, "[CorridorKey] Single input mode (4ch BCHW): plate+alpha merged");
        } else {
            obs_log(LOG_INFO, "[CorridorKey] Dual input mode: plate + alpha_hint separate");
        }

        return true;
    }

    // BCHW: input width/height from dim[3] and dim[2]
    virtual void getNetworkInputSize(const std::vector<std::vector<int64_t>> &inputDims,
                                     uint32_t &inputWidth, uint32_t &inputHeight)
    {
        if (inputDims.empty() || inputDims[0].size() < 4) {
            inputWidth = 512;
            inputHeight = 512;
            return;
        }
        inputWidth  = (uint32_t)inputDims[0][3]; // W
        inputHeight = (uint32_t)inputDims[0][2]; // H
    }

    // Preprocess: normalize to [0,1], convert HWC→CHW
    virtual void prepareInputToNetwork(cv::Mat &resizedImage, cv::Mat &preprocessedImage)
    {
        // Convert BGR to RGB
        cv::Mat rgbImage;
        cv::cvtColor(resizedImage, rgbImage, cv::COLOR_BGR2RGB);

        // Normalize to [0, 1]
        cv::Mat normalized;
        rgbImage.convertTo(normalized, CV_32FC3, 1.0 / 255.0);

        // HWC → CHW
        hwc_to_chw(normalized, preprocessedImage);
    }

    virtual void loadInputToTensor(const cv::Mat &preprocessedImage,
                                   uint32_t inputWidth, uint32_t inputHeight,
                                   std::vector<std::vector<float>> &inputTensorValues)
    {
        // CHW RGB data → input tensor [0]
        size_t rgbSize = 3 * inputWidth * inputHeight;

        if (inputTensorValues.size() == 1) {
            // Single input: 4ch BCHW (RGB + AlphaHint merged)
            // Layout: [R_plane, G_plane, B_plane, Alpha_plane]
            inputTensorValues[0].resize(4 * inputWidth * inputHeight, 0.0f);

            // Copy RGB (first 3 channels)
            const float *src = preprocessedImage.ptr<float>(0);
            std::copy(src, src + rgbSize, inputTensorValues[0].begin());

            // AlphaHint channel (channel index 3)
            float *alphaPtr = inputTensorValues[0].data() + rgbSize;
            buildAlphaHint(inputWidth, inputHeight, alphaPtr);

        } else if (inputTensorValues.size() >= 2) {
            // Dual input: plate + alpha_hint separate tensors
            const float *src = preprocessedImage.ptr<float>(0);
            inputTensorValues[0].assign(src, src + rgbSize);

            // AlphaHint as second input
            inputTensorValues[1].resize(inputWidth * inputHeight, 0.0f);
            buildAlphaHint(inputWidth, inputHeight, inputTensorValues[1].data());
        }
    }

    // After inference, save alpha output for next frame's AlphaHint
    virtual void assignOutputToInput(std::vector<std::vector<float>> &outputTensorValues,
                                     std::vector<std::vector<float>> &inputTensorValues)
    {
        UNUSED_PARAMETER(inputTensorValues);

        // outputTensorValues[0] = alpha matte (1CHW or 1HW1)
        if (!outputTensorValues.empty() && !outputTensorValues[0].empty()) {
            // Save as prevAlpha for next frame
            // We store it as flat float vector, will reshape on use
            size_t sz = outputTensorValues[0].size();
            int side = (int)std::sqrt((double)sz);
            prevAlpha = cv::Mat(side, side, CV_32FC1,
                                outputTensorValues[0].data()).clone();
        }
    }

    // Get final mask output
    // CorridorKey output[0] = alpha (1CHW float), range [0,1]
    // We convert to uint8 HWC for the rest of the pipeline
    virtual cv::Mat getNetworkOutput(const std::vector<std::vector<int64_t>> &outputDims,
                                     std::vector<std::vector<float>> &outputTensorValues)
    {
        if (outputDims.empty() || outputTensorValues.empty()) {
            return cv::Mat();
        }

        // outputDims[0] should be BCHW: [1, 1, H, W]
        int H = 1, W = 1;
        if (outputDims[0].size() >= 4) {
            H = (int)outputDims[0][2];
            W = (int)outputDims[0][3];
        } else if (outputDims[0].size() >= 3) {
            H = (int)outputDims[0][1];
            W = (int)outputDims[0][2];
        }

        // Alpha matte, float [0,1]
        cv::Mat alphaMat(H, W, CV_32FC1, outputTensorValues[0].data());

        // Save for next frame's AlphaHint
        prevAlpha = alphaMat.clone();

// Return float [0,1] — pipeline will convert to uint8
        return alphaMat.clone();
    }

    virtual void postprocessOutput(cv::Mat &output)
    {
        // Output is already uint8 from getNetworkOutput
        // No additional processing needed
        UNUSED_PARAMETER(output);
    }

private:
    // Build AlphaHint from previous frame's alpha
    // First frame: use rough chroma key (green detection) as initial hint
    void buildAlphaHint(uint32_t width, uint32_t height, float *outPtr)
    {
        size_t pixelCount = (size_t)width * height;

        if (prevAlpha.empty()) {
            // First frame: no previous alpha
            // Use neutral 0.5 hint (let CorridorKey figure it out)
            std::fill(outPtr, outPtr + pixelCount, 0.5f);
            obs_log(LOG_INFO, "[CorridorKey] First frame: using neutral AlphaHint (0.5)");
        } else {
            // Subsequent frames: resize prev alpha to current input size and use as hint
            cv::Mat resized;
            cv::resize(prevAlpha, resized, cv::Size((int)width, (int)height),
                       0, 0, cv::INTER_LINEAR);

            const float *src = resized.ptr<float>(0);
            std::copy(src, src + pixelCount, outPtr);
        }
    }
};

#endif // MODELCORRIDORKEY_H
