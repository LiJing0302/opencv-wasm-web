import { useCallback, useEffect, useRef } from "react";

interface UseWasmProcessorProps {
  videoDOM: HTMLVideoElement | null;
  canvasDOM: HTMLCanvasElement | null;
  playVideoDOM?: HTMLVideoElement | null;
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
  // 创建临时 canvas 获取视频帧数据
  const tempCanvas = document.createElement("canvas");
  const tempCtx = tempCanvas.getContext("2d");

  if (tempCtx) {
    // 确保视频尺寸正确
    const videoWidth = videoDataRef.videoWidth;
    const videoHeight = videoDataRef.videoHeight;
    const videoSize = videoWidth * videoHeight * 4;

    tempCanvas.width = videoWidth;
    tempCanvas.height = videoHeight;
    // 绘制视频帧到临时 canvas
    tempCtx.drawImage(videoDataRef, 0, 0, videoWidth, videoHeight);

    // 获取视频帧数据
    const videoImageData = tempCtx.getImageData(0, 0, videoWidth, videoHeight);
    const videoData = videoImageData.data;

    // 检查内存边界
    if (videoData.length > videoSize) {
      console.error("视频数据大小超出分配的内存空间");
      return;
    }
    // 将视频数据复制到 WebAssembly 内存
    HEAPU8.set(videoData, videoDst);
    console.log("调用 _detect_objects_and_homography 参数:", {
      dst,
      imageWidth: imageData.width,
      imageHeight: imageData.height,
      videoDst,
      videoWidth,
      videoHeight,
      imageDataSize: data.length,
      videoDataSize: videoData.length,
      heapSize: HEAPU8.length,
    });

    // 调用 WebAssembly 函数处理图像
    _detect_objects_and_homography(
      dst, // 输入图像的内存地址
      imageData.width, // 图像宽度
      imageData.height, // 图像高度
      videoDst || 0, // 视频数据内存地址，如果没有则传 0
      videoWidth, // 视频宽度
      videoHeight // 视频高度
    );
    const result = HEAPU8.subarray(dst, dst + data.length); // 从 WebAssembly 内存中读取处理后的图像数据（Uint8Array）
    imageData.data.set(result); // 将处理后的数据写入 ImageData 对象
    ctx.putImageData(imageData, 0, 0); //将 ImageData 绘制到 canvas 上; 0, 0 表示从 canvas 的左上角开始绘制
  }
};

export function useWasmProcessor({
  videoDOM,
  canvasDOM,
  playVideoDOM,
}: UseWasmProcessorProps) {
  const dstRef = useRef<number>(undefined);
  const videoDstRef = useRef<number>(undefined);
  const sFilter = useCallback(() => {
    const ctx = canvasDOM?.getContext("2d");

    if (!videoDOM || !canvasDOM || !ctx || videoDOM.paused || videoDOM.ended) {
      requestAnimationFrame(sFilter);
      return;
    } else {
      canvasDOM.width = videoDOM.videoWidth;
      canvasDOM.height = videoDOM.videoHeight;
      console.log(videoDOM.videoWidth, videoDOM.videoHeight);
      ctx.drawImage(videoDOM, 0, 0, canvasDOM.width, canvasDOM.height);
      const imageData = ctx.getImageData(
        0,
        0,
        canvasDOM.width,
        canvasDOM.height
      );
      detectEdges({
        imageData,
        ctx,
        dst: dstRef.current,
        videoDataRef: playVideoDOM,
        videoDst: videoDstRef.current,
      });
      requestAnimationFrame(sFilter);
    }
  }, [canvasDOM, videoDOM, playVideoDOM]);

  useEffect(() => {
    if (canvasDOM && canvasDOM.width && canvasDOM.height) {
      const imageSize = canvasDOM.width * canvasDOM.height * 4; // 4 bytes per pixel (RGBA)
      console.log(canvasDOM.width, canvasDOM.height);
      dstRef.current = _malloc(imageSize); // 通过 _malloc 分配的内存地址
      return () => {
        if (dstRef.current) {
          _free(dstRef.current);
          dstRef.current = undefined;
        }
      }; // 释放内存，防止内存泄漏
    }
  }, [canvasDOM]);

  useEffect(() => {
    if (playVideoDOM) {
      playVideoDOM.onloadeddata = () => {
        console.log("视频加载完成");
        playVideoDOM.play().catch((err) => console.error("视频播放失败:", err));
        const videoSize =
          playVideoDOM.videoWidth * playVideoDOM.videoHeight * 4;
        videoDstRef.current = _malloc(videoSize);
      };
      return () => {
        if (videoDstRef.current) {
          _free(videoDstRef.current);
          videoDstRef.current = undefined;
        }
      };
    }
  }, [playVideoDOM]);

  useEffect(() => {
    sFilter();
  }, [sFilter]);
}
