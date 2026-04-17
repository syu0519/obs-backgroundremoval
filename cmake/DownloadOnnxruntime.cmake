cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED PLATFORM)
  if(APPLE)
    set(PLATFORM "macos")
  elseif(WIN32)
    set(PLATFORM "windows")
  else()
    set(PLATFORM "linux")
  endif()
endif()

if(NOT DEFINED GPU)
  set(GPU OFF)
endif()

file(REMOVE_RECURSE onnxruntime)

if(PLATFORM STREQUAL "macos")
  file(
    DOWNLOAD
      https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-osx-universal2-1.23.2.tgz
      onnxruntime-osx-universal2-1.23.2.tgz
    EXPECTED_HASH SHA256=49ae8e3a66ccb18d98ad3fe7f5906b6d7887df8a5edd40f49eb2b14e20885809
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xf onnxruntime-osx-universal2-1.23.2.tgz
  )
  file(RENAME onnxruntime-osx-universal2-1.23.2 onnxruntime)
elseif(PLATFORM STREQUAL "windows")
  if(GPU)
    file(
  DOWNLOAD
    https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-win-x64-gpu-1.20.1.zip
    onnxruntime-win-x64-gpu-1.20.1.zip
  EXPECTED_HASH SHA256=3e9658d4aa7c21b3f5cbb5a7ce0356184f3c183c317b52f9cfff23a3f079634e
)
execute_process(
  COMMAND ${CMAKE_COMMAND} -E tar xf onnxruntime-win-x64-gpu-1.20.1.zip
)
file(RENAME onnxruntime-win-x64-gpu-1.20.1 onnxruntime)
  else()
    file(
  DOWNLOAD
    https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-win-x64-gpu-1.20.1.zip
    onnxruntime-win-x64-gpu-1.20.1.zip
  EXPECTED_HASH SHA256=3e9658d4aa7c21b3f5cbb5a7ce0356184f3c183c317b52f9cfff23a3f079634e
)
execute_process(
  COMMAND ${CMAKE_COMMAND} -E tar xf onnxruntime-win-x64-gpu-1.20.1.zip
)
file(RENAME onnxruntime-win-x64-gpu-1.20.1 onnxruntime)
  endif()
elseif(PLATFORM STREQUAL "linux")
 file(
  DOWNLOAD
    https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-win-x64-gpu-1.20.1.zip
    onnxruntime-win-x64-gpu-1.20.1.zip
  EXPECTED_HASH SHA256=3e9658d4aa7c21b3f5cbb5a7ce0356184f3c183c317b52f9cfff23a3f079634e
)
execute_process(
  COMMAND ${CMAKE_COMMAND} -E tar xf onnxruntime-win-x64-gpu-1.20.1.zip
)
file(RENAME onnxruntime-win-x64-gpu-1.20.1 onnxruntime)
  execute_process(COMMAND ln -s lib onnxruntime/lib64)
else()
  message(FATAL_ERROR "Unsupported platform: ${PLATFORM}")
endif()

if(EXISTS onnxruntime/lib/cmake/onnxruntime/onnxruntimeTargets.cmake)
  file(READ onnxruntime/lib/cmake/onnxruntime/onnxruntimeTargets.cmake FILE_CONTENT)

  set(OLD_STRING "INTERFACE_INCLUDE_DIRECTORIES \"\${_IMPORT_PREFIX}/include/onnxruntime\"")
  set(NEW_STRING "INTERFACE_INCLUDE_DIRECTORIES \"\${_IMPORT_PREFIX}/include\"")

  string(REPLACE "${OLD_STRING}" "${NEW_STRING}" MODIFIED_CONTENT "${FILE_CONTENT}")
  file(WRITE onnxruntime/lib/cmake/onnxruntime/onnxruntimeTargets.cmake "${MODIFIED_CONTENT}")
endif()
