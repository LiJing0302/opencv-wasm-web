interface WasmProcessorProps {
  videoDOM: HTMLVideoElement;
  canvasDOM: HTMLCanvasElement;
  playVideoDOM: HTMLVideoElement;
}

interface WasmProcessor {
  start: () => void;
  stop: () => void;
}

let lastTime = Date.now();
let frames = 0;

export function createWasmProcessor({
  videoDOM,
  canvasDOM,
  playVideoDOM,
}: WasmProcessorProps): WasmProcessor {
  let dstPtr: number | undefined;
  let videoDstPtr: number | undefined;
  let animationFrameId: number | undefined;

  // 缓存 context
  const ctx = canvasDOM.getContext("2d");
  if (!ctx) {
    throw new Error("无法获取 canvas context");
  }

  // 创建复用的临时canvas
  const tempCanvas = document.createElement("canvas");
  const tempCtx = tempCanvas.getContext("2d");
  if (!tempCtx) {
    throw new Error("无法获取临时canvas context");
  }
  // 初始化canvas尺寸
  tempCanvas.width = playVideoDOM.videoWidth;
  tempCanvas.height = playVideoDOM.videoHeight;

  // 初始化内存
  function initMemory() {
    const imageSize = videoDOM.videoWidth * videoDOM.videoHeight * 4;
    dstPtr = _malloc(imageSize);
    const videoSize = playVideoDOM.videoWidth * playVideoDOM.videoHeight * 4;
    videoDstPtr = _malloc(videoSize);
  }

  const detectEdges = ({
    imageData,
    ctx,
    dst,
    videoDataRef,
    videoDst,
  }: {
    imageData: ImageData;
    ctx: CanvasRenderingContext2D;
    dst?: number;
    videoDataRef?: HTMLVideoElement | null;
    videoDst?: number;
  }) => {
    // 添加视频数据就绪检查
    if (!dst) return;
    if (!videoDataRef?.videoWidth || !videoDataRef?.videoHeight || !videoDst)
      return;
    const data = imageData.data; // Uint8ClampedArray 格式，RGBA 数据
    // 将图像数据复制到 WebAssembly 的内存中
    HEAPU8.set(data, dst);
    // 只在视频尺寸变化时更新canvas尺寸
    if (
      tempCanvas.width !== videoDataRef.videoWidth ||
      tempCanvas.height !== videoDataRef.videoHeight
    ) {
      tempCanvas.width = videoDataRef.videoWidth;
      tempCanvas.height = videoDataRef.videoHeight;
    }
    if (tempCtx) {
      // 确保视频尺寸正确
      const videoWidth = videoDataRef.videoWidth;
      const videoHeight = videoDataRef.videoHeight;
      const videoSize = videoWidth * videoHeight * 4;

      // 绘制视频帧到临时 canvas
      tempCtx.drawImage(videoDataRef, 0, 0, videoWidth, videoHeight);

      // 获取视频帧数据
      const videoImageData = tempCtx.getImageData(
        0,
        0,
        videoWidth,
        videoHeight
      );
      const videoData = videoImageData.data;

      // 检查内存边界
      if (videoData.length > videoSize) {
        console.error("视频数据大小超出分配的内存空间");
        return;
      }
      // 将视频数据复制到 WebAssembly 内存
      HEAPU8.set(videoData, videoDst);
      console.time();
      // 调用 WebAssembly 函数处理图像
      _detect_objects_and_homography(
        dst, // 输入图像的内存地址
        imageData.width, // 图像宽度
        imageData.height, // 图像高度
        videoDst || 0, // 视频数据内存地址，如果没有则传 0
        videoWidth, // 视频宽度
        videoHeight // 视频高度
      );
      console.timeEnd();
      const result = HEAPU8.subarray(dst, dst + data.length); // 从 WebAssembly 内存中读取处理后的图像数据（Uint8Array）
      imageData.data.set(result); // 将处理后的数据写入 ImageData 对象
      ctx.putImageData(imageData, 0, 0); //将 ImageData 绘制到 canvas 上; 0, 0 表示从 canvas 的左上角开始绘制
    }
  };

  // 处理视频帧
  function processFrame() {
    // const now = Date.now();
    // const delta = now - lastTime;
    // frames++;
    // if (delta >= 100) {
    //   // 改为500毫秒更新一次
    //   // 每0.5秒更新一次
    //   const fps = Math.round(frames / (delta / 1000));
    //   console.log(`当前FPS: ${fps}`);
    //   frames = 0;
    //   lastTime = now;
    // }
    if (!ctx || videoDOM.paused || videoDOM.ended) {
      animationFrameId = requestAnimationFrame(processFrame);
      return;
    }

    canvasDOM.width = videoDOM.videoWidth;
    canvasDOM.height = videoDOM.videoHeight;
    ctx.drawImage(videoDOM, 0, 0, canvasDOM.width, canvasDOM.height);

    const imageData = ctx.getImageData(0, 0, canvasDOM.width, canvasDOM.height);
    detectEdges({
      imageData,
      ctx,
      dst: dstPtr,
      videoDataRef: playVideoDOM,
      videoDst: videoDstPtr,
    });
    animationFrameId = requestAnimationFrame(processFrame);
  }

  return {
    start: () => {
      initMemory();
      processFrame();
    },
    stop: () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
      if (dstPtr) {
        _free(dstPtr);
        dstPtr = undefined;
      }
      if (videoDstPtr) {
        _free(videoDstPtr);
        videoDstPtr = undefined;
      }
    },
  };
}
