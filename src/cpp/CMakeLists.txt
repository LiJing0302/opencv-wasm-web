cmake_minimum_required(VERSION 3.15)
project(OpenCVWASM)

set(CMAKE_CXX_STANDARD 11)

# 指定 OpenCV WASM 路径
set(OpenCV_DIR "/Users/bytedance/opencv/build_simd_threads")
# set(OpenCV_DIR "/Users/bytedance/opencv/build_simd")

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# # 设置导出函数列表
# set(EXPORTED_FUNCTIONS,"['_detect_objects_and_homography','_set_reference_image','_malloc','_free']")


# 应用标志到编译器和链接器
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s FORCE_FILESYSTEM=1 -s INITIAL_MEMORY=256MB -s EXIT_RUNTIME=1 -s EXPORTED_FUNCTIONS=['_detect_orb_features','_detect_objects_and_homography','_set_reference_image','_malloc','_free']")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EMSCRIPTEN_FLAGS} -s FORCE_FILESYSTEM=1 -s INITIAL_MEMORY=256MB -s EXIT_RUNTIME=1 -s EXPORTED_FUNCTIONS=['_detect_orb_features','_detect_objects_and_homography','_set_reference_image','_malloc','_free']")
set(CMAKE_EXECUTBLE_LINKER_FLAGS "${CMAKE_EXECUTBLE_LINKER_FLAGS} -s FORCE_FILESYSTEM=1 -s INITIAL_MEMORY=256MB -s EXIT_RUNTIME=1 -s EXPORTED_FUNCTIONS=['_detect_orb_features','_detect_objects_and_homography','_set_reference_image','_malloc','_free']")

# 添加源文件
add_executable(opencv_wasm
    # ./main.cpp
    ar_detector.cpp
)

# 链接 OpenCV 库
target_link_libraries(opencv_wasm ${OpenCV_LIBS})

