import { loadOpenCV as _loadOpenCV } from "@/utils/loader";

export const loadOpenCV = (cb: () => void) => {
  const pathsConfig = {
    wasm: "/opencv_wasm.js", // 基础版本
    simd: "/opencv_wasm.js", // simd版本
    threads: "/opencv_wasm.js", // 多线程版本
    threadsSimd: "/opencv_wasm.js", // 多线程+SIMD版本
  };
  // 加载 OpenCV.js
  _loadOpenCV(pathsConfig, cb);
};
