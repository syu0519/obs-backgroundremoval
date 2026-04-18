#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>
#include <filesystem>

#if defined(__APPLE__)
#include <coreml_provider_factory.h>
#endif

#ifdef _WIN32
#include <wchar.h>
#include <windows.h>
#endif // _WIN32

#include <obs-module.h>

#include "ort-session-utils.h"
#include "consts.h"
#include "plugin-support.h"

int createOrtSession(filter_data *tf)
{
	if (tf->model.get() == nullptr) {
		obs_log(LOG_ERROR, "Model object is not initialized");
		return OBS_BGREMOVAL_ORT_SESSION_ERROR_INVALID_MODEL;
	}

	Ort::SessionOptions sessionOptions;

sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
	if (tf->useGPU != USEGPU_CPU) {
		sessionOptions.DisableMemPattern();
		sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
	} else {
		sessionOptions.SetInterOpNumThreads(tf->numThreads);
		sessionOptions.SetIntraOpNumThreads(tf->numThreads);
	}

	char *modelFilepath_rawPtr = obs_module_file(tf->modelSelection.c_str());

	if (modelFilepath_rawPtr == nullptr) {
		obs_log(LOG_ERROR, "Unable to get model filename %s from plugin.", tf->modelSelection.c_str());
		return OBS_BGREMOVAL_ORT_SESSION_ERROR_FILE_NOT_FOUND;
	}

	std::string modelFilepath_s(modelFilepath_rawPtr);

#if _WIN32
	int outLength = MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, modelFilepath_rawPtr, -1, nullptr, 0);
	tf->modelFilepath = std::wstring(outLength, L'\0');
	MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, modelFilepath_rawPtr, -1, tf->modelFilepath.data(), outLength);
#else
	tf->modelFilepath = std::string(modelFilepath_rawPtr);
#endif

	bfree(modelFilepath_rawPtr);

	try {
#ifdef HAVE_ONNXRUNTIME_CUDA_EP
		if (tf->useGPU == USEGPU_CUDA) {
			// CorridorKey fix: use explicit CUDA provider options
			// to prevent memory arena crash on dual-GPU systems (e.g. dual RTX 3090)
			// with fp16 models. Default arena is too aggressive with VRAM allocation.
			// Fix: disable cuDNN precompiled engines which fail-fast on RTX 3090 (Ampere/sm_86)
			// with cuDNN 9.x fp16 Conv kernels (cask_cudnn::InitializeAllResources abort)
			_putenv_s("CUDNN_DISABLE_PRECOMPILED_KERNELS", "1");
			obs_log(LOG_INFO, "[CorridorKey] CUDNN_DISABLE_PRECOMPILED_KERNELS=1 set");

			OrtCUDAProviderOptions cuda_options;
			memset(&cuda_options, 0, sizeof(cuda_options));
			cuda_options.device_id = 0;
			cuda_options.arena_extend_strategy = 1;                 // kSameAsRequested: no greedy pre-alloc
			cuda_options.gpu_mem_limit = 8ULL * 1024 * 1024 * 1024; // 8 GB cap
			cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault; // avoid precompiled engine path
			cuda_options.do_copy_in_default_stream = 1;
			sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
			obs_log(LOG_INFO,
				"[CorridorKey] CUDA EP: arena_extend_strategy=kSameAsRequested, gpu_mem_limit=8GB, algo_search=Default");
		}
#endif
#ifdef HAVE_ONNXRUNTIME_ROCM_EP
		if (tf->useGPU == USEGPU_ROCM) {
			Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_ROCM(sessionOptions, 0));
		}
#endif
#ifdef HAVE_ONNXRUNTIME_MIGRAPHX_EP
		if (tf->useGPU == USEGPU_MIGRAPHX) {
			Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_MIGraphX(sessionOptions, 0));
		}
#endif
		// Re-register EP library before appending
		char module_path[MAX_PATH] = {0};
		GetModuleFileNameA(GetModuleHandleA("obs-backgroundremoval.dll"), module_path, MAX_PATH);
		std::string ep_lib_str = std::filesystem::path(module_path).parent_path().string() +
					 "\\obs-backgroundremoval\\onnxruntime_providers_nv_tensorrt_rtx.dll";
		std::wstring ep_lib_w(ep_lib_str.begin(), ep_lib_str.end());
		OrtStatus *reg_status =
			Ort::GetApi().RegisterExecutionProviderLibrary(*tf->env, ep_lib_str.c_str(), ep_lib_w.c_str());
		if (reg_status) {
			obs_log(LOG_WARNING, "[CorridorKey] Re-register EP failed: %s",
				Ort::GetApi().GetErrorMessage(reg_status));
			Ort::GetApi().ReleaseStatus(reg_status);
		} else {
			obs_log(LOG_INFO, "[CorridorKey] EP re-registered in createOrtSession");
		}

#ifdef HAVE_ONNXRUNTIME_TENSORRT_EP
		if (tf->useGPU == USEGPU_TENSORRT) {
			try {
				// Get EP devices
				OrtEpDevice **ep_devices = nullptr;
				size_t device_count = 0;
				Ort::ThrowOnError(Ort::GetApi().GetEpDevices(*tf->env, &ep_devices, &device_count));

				if (device_count > 0) {
					// Select first device
					std::vector<const OrtEpDevice *> selected = {ep_devices[0]};

					// Append using V2 API
					OrtKeyValuePairs *kv = nullptr;
					Ort::GetApi().CreateKeyValuePairs(&kv);
					Ort::GetApi().AddKeyValuePair(kv, "device_id", "0");
					Ort::GetApi().AddKeyValuePair(kv, "trt_fp16_enable", "1");

					Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider_V2(
						sessionOptions, *tf->env, selected.data(), selected.size(), kv));

					Ort::GetApi().ReleaseKeyValuePairs(kv);
					Ort::GetApi().ReleaseEpDevices(ep_devices, device_count);
					obs_log(LOG_INFO, "[CorridorKey] NvTensorRTRTX EP V2 initialized");
				}
			} catch (const Ort::Exception &e) {
				obs_log(LOG_ERROR, "[CorridorKey] NvTensorRTRTX EP failed: %s", e.what());
				return OBS_BGREMOVAL_ORT_SESSION_ERROR_STARTUP;
			}
		}
#endif

#if defined(__APPLE__)
		if (tf->useGPU == USEGPU_COREML) {
			uint32_t coreml_flags = 0;
			coreml_flags |= COREML_FLAG_ENABLE_ON_SUBGRAPH;
			Ort::ThrowOnError(
				OrtSessionOptionsAppendExecutionProvider_CoreML(sessionOptions, coreml_flags));
		}
#endif
		tf->session.reset(new Ort::Session(*tf->env, tf->modelFilepath.c_str(), sessionOptions));
	} catch (const std::exception &e) {
		obs_log(LOG_ERROR, "%s", e.what());
		return OBS_BGREMOVAL_ORT_SESSION_ERROR_STARTUP;
	}

	Ort::AllocatorWithDefaultOptions allocator;

	tf->model->populateInputOutputNames(tf->session, tf->inputNames, tf->outputNames);

	if (!tf->model->populateInputOutputShapes(tf->session, tf->inputDims, tf->outputDims)) {
		obs_log(LOG_ERROR, "Unable to get model input and output shapes");
		return OBS_BGREMOVAL_ORT_SESSION_ERROR_INVALID_INPUT_OUTPUT;
	}

	for (size_t i = 0; i < tf->inputNames.size(); i++) {
		obs_log(LOG_INFO, "Model %s input %d: name %s shape (%d dim) %d x %d x %d x %d",
			tf->modelSelection.c_str(), (int)i, tf->inputNames[i].get(), (int)tf->inputDims[i].size(),
			(int)tf->inputDims[i][0], ((int)tf->inputDims[i].size() > 1) ? (int)tf->inputDims[i][1] : 0,
			((int)tf->inputDims[i].size() > 2) ? (int)tf->inputDims[i][2] : 0,
			((int)tf->inputDims[i].size() > 3) ? (int)tf->inputDims[i][3] : 0);
	}
	for (size_t i = 0; i < tf->outputNames.size(); i++) {
		obs_log(LOG_INFO, "Model %s output %d: name %s shape (%d dim) %d x %d x %d x %d",
			tf->modelSelection.c_str(), (int)i, tf->outputNames[i].get(), (int)tf->outputDims[i].size(),
			(int)tf->outputDims[i][0], ((int)tf->outputDims[i].size() > 1) ? (int)tf->outputDims[i][1] : 0,
			((int)tf->outputDims[i].size() > 2) ? (int)tf->outputDims[i][2] : 0,
			((int)tf->outputDims[i].size() > 3) ? (int)tf->outputDims[i][3] : 0);
	}

	// Allocate buffers
	tf->model->allocateTensorBuffers(tf->inputDims, tf->outputDims, tf->outputTensorValues, tf->inputTensorValues,
					 tf->inputTensor, tf->outputTensor);

	return OBS_BGREMOVAL_ORT_SESSION_SUCCESS;
}

static bool runInferenceSEH(filter_data *tf)
{
	__try {
		tf->model->runNetworkInference(tf->session, tf->inputNames, tf->outputNames, tf->inputTensor,
					       tf->outputTensor);
		return true;
	} __except (EXCEPTION_EXECUTE_HANDLER) {
		obs_log(LOG_ERROR, "[CorridorKey] SEH exception in session->Run(), code=0x%08X", GetExceptionCode());
		return false;
	}
}

bool runFilterModelInference(filter_data *tf, const cv::Mat &imageBGRA, cv::Mat &output)
{
	if (tf->session.get() == nullptr) {
		return false;
	}
	if (tf->model.get() == nullptr) {
		return false;
	}

	// To RGB
	cv::Mat imageRGB;
	cv::cvtColor(imageBGRA, imageRGB, cv::COLOR_BGRA2RGB);

	// Resize to network input size
	uint32_t inputWidth, inputHeight;
	tf->model->getNetworkInputSize(tf->inputDims, inputWidth, inputHeight);

	cv::Mat resizedImageRGB;
	cv::resize(imageRGB, resizedImageRGB, cv::Size(inputWidth, inputHeight));

	// Prepare input to network
	// Note: resizedImageRGB is CV_8UC3 [0,255]
	// convertTo CV_32F keeps range [0,255] -> prepareInputToNetwork does /255 normalization
	cv::Mat resizedImage, preprocessedImage;
	resizedImageRGB.convertTo(resizedImage, CV_32F);

	tf->model->prepareInputToNetwork(resizedImage, preprocessedImage);

	tf->model->loadInputToTensor(preprocessedImage, inputWidth, inputHeight, tf->inputTensorValues);

// Run network inference
	obs_log(LOG_INFO, "[CorridorKey] Before session->Run() inputSize=%zu", tf->inputTensorValues[0].size());
	if (!runInferenceSEH(tf)) {
		return false;
	}
	obs_log(LOG_INFO, "[CorridorKey] After session->Run() OK");


	// Get output -> returns CV_8U [0,255] for CorridorKey,
	//              or CV_32FC1 [0,1] for other models
	cv::Mat outputImage = tf->model->getNetworkOutput(tf->outputDims, tf->outputTensorValues);

	if (outputImage.empty()) {
		obs_log(LOG_ERROR, "Network output is empty");
		return false;
	}

	// Assign output to input for models with temporal feedback
	tf->model->assignOutputToInput(tf->outputTensorValues, tf->inputTensorValues);

	// Post-process output
	tf->model->postprocessOutput(outputImage);

	// Convert to CV_8U [0,255]
	// If already CV_8U (CorridorKey): this is a safe copy, values unchanged
	// If CV_32FC1 [0,1] (other models): multiply by 255
	if (outputImage.type() == CV_32FC1) {
		outputImage.convertTo(output, CV_8U, 255.0);
	} else {
		// Already CV_8U
		output = outputImage;
	}

	return true;
}
