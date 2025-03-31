declare global {
  interface Window {
    Module: {
      onRuntimeInitialized?: () => void;
    };
  }
  // // WebAssembly 相关的类型声明
  function _malloc(size: number): number;
  function _free(ptr: number): void;
  function _set_reference_image(
    ptr: number,
    width: number,
    height: number
  ): void;
  function _detect_orb_features(
    ptr: number,
    width: number,
    height: number
  ): number;
  // 目标检测和单应性变换函数
  function _detect_objects_and_homography(
    srcPtr: number,
    width: number,
    height: number,
    videoFrame: number,
    videoWidth: number,
    videoHeight: number
  ): void;
  let HEAP8: Int8Array;
  let HEAP16: Int16Array;
  let HEAP32: Int32Array;
  let HEAPU8: Uint8Array;
  let HEAPU16: Uint16Array;
  let HEAPU32: Uint32Array;
  let HEAPF32: Float32Array;
  let HEAPF64: Float64Array;
  interface ImageData {
    data: Uint8Array;
    width: number;
    height: number;
  }
}
export {};
