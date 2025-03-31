#!/bin/bash

# 检查 EMSDK 环境变量是否设置
if [ -z "$EMSDK" ]; then
  echo "错误: EMSDK 环境变量未设置。请先设置 Emscripten SDK 路径。"
  exit 1
fi

# 获取项目根目录的绝对路径
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 创建构建目录
cd "$PROJECT_ROOT/src/cpp"
mkdir -p build
cd build

# 运行 CMake 配置
echo "正在配置 CMake..."
cmake -DCMAKE_TOOLCHAIN_FILE=$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake ..

# 检查 CMake 是否成功
if [ $? -ne 0 ]; then
  echo "CMake 配置失败。"
  exit 1
fi

# 运行 Make 编译
echo "正在编译..."
make -j4

# 检查编译是否成功
if [ $? -ne 0 ]; then
  echo "编译失败。"
  exit 1
fi

echo "编译成功！"
echo "输出文件位于: $(pwd)/opencv_wasm.js 和 $(pwd)/opencv_wasm.wasm"

# 创建 assets 目录（如果不存在）
mkdir -p ../assets

# 复制编译后的文件到 assets 目录
cp opencv_wasm.js opencv_wasm.wasm $PROJECT_ROOT/public

# 检查复制是否成功
if [ $? -ne 0 ]; then
  echo "文件复制失败。"
  exit 1
fi

echo "文件已成功复制到 public 目录。"

# 返回到原始目录
cd ..