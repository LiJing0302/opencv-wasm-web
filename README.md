构建 opencvJs 参数：
emcmake python3 ./platforms/js/build_js.py
options:
-h, --help show this help message and exit
--opencv_dir OPENCV_DIR
Opencv source directory (default is "../.." relative
to script location)
--emscripten_dir EMSCRIPTEN_DIR
Path to Emscripten to use for build (deprecated in
favor of 'emcmake' launcher)
--build_wasm Build OpenCV.js in WebAssembly format
--disable_wasm Build OpenCV.js in Asm.js format
--disable_single_file
Do not merge JavaScript and WebAssembly into one
single file
--threads Build OpenCV.js with threads optimization
--simd Build OpenCV.js with SIMD optimization
--build_test Build tests
--build_perf Build performance tests
--build_doc Build tutorials
--build_loader Build OpenCV.js loader
--clean_build_dir Clean build dir
--skip_config Skip cmake config
--config_only Only do cmake config
--enable_exception Enable exception handling
--cmake_option CMAKE_OPTION
Append CMake options
--build_flags BUILD_FLAGS
Append Emscripten build options
--build_wasm_intrin_test
Build WASM intrin tests
--config CONFIG Specify configuration file with own list of exported
into JS functions
--webnn Enable WebNN Backend

构建 opencvJs - wasm 基础版：
emcmake python3 ./platforms/js/build_js.py build_wasm --build_wasm --build_loader --disable_single_file

构建 opencvJs - asm.js：
emcmake python3 ./platforms/js/build_js.py build_asm --disable_wasm --build_loader --disable_single_file --build_perf

构建 opencvJs - only simd：
emcmake python3 ./platforms/js/build_js.py build_simd --simd --build_loader --disable_single_file --build_wasm_intrin_test --build_perf

构建 opencvJs - only threads：
emcmake python3 ./platforms/js/build_js.py build_threads --threads --build_loader --disable_single_file --build_perf

构建 opencvJs - simdThreads：
emcmake python3 ./platforms/js/build_js.py build_simd_threads --simd --threads --build_loader --disable_single_file --build_wasm_intrin_test --build_perf
