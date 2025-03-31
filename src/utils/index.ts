import { FeaturePoint } from "@/types/orbFeaturePoint";

export const urlToImageData = async (imageUrl: string) => {
  // 1. 创建Image对象加载图片
  const img = new Image();
  img.crossOrigin = "Anonymous"; // 处理跨域问题

  // 2. 等待图片加载完成
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = reject;
    img.src = imageUrl;
  });

  // 3. 创建临时Canvas
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d");

  // 4. 绘制图片到Canvas
  ctx?.drawImage(img, 0, 0);

  // 5. 提取ImageData
  return ctx?.getImageData(0, 0, canvas.width, canvas.height);
};

export const fetchStreamCamera = async () => {
  try {
    // 获取支持的约束
    const constraints: MediaStreamConstraints = {
      audio: false,
      video: {
        width: 1280, // 设置理想宽度
        height: 720, // 设置理想高度
        facingMode: "environment",
        // frameRate: { min: 20, ideal: 60, max: 60 },
        frameRate: 60,
      },
    };

    const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
    // 获取实际的视频轨道信息
    const videoTrack = mediaStream.getVideoTracks()[0];
    const settings = videoTrack.getSettings();
    console.log("实际视频设置:", settings);

    return mediaStream;
  } catch (err) {
    console.error(err);
  }
};

/** 调用 wasm 获取特征点 */
export const detectFeatures = (
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number
) => {
  try {
    const imageData = ctx.getImageData(0, 0, width, height);
    const dataPtr = _malloc(imageData.data.length);
    HEAPU8.set(imageData.data, dataPtr);

    const resultPtr = _detect_orb_features(dataPtr, width, height);
    const numFeatures = HEAP32[resultPtr >> 2];

    const points: FeaturePoint[] = [];
    let offset = 4; // 跳过特征点数量(4字节)

    for (let i = 0; i < numFeatures; i++) {
      // 读取浮点数据
      const x = HEAPF32[(resultPtr + offset) >> 2]; //
      const y = HEAPF32[(resultPtr + offset + 4) >> 2];
      const size = HEAPF32[(resultPtr + offset + 8) >> 2];
      const angle = HEAPF32[(resultPtr + offset + 12) >> 2];
      const response = HEAPF32[(resultPtr + offset + 16) >> 2];

      // 读取整数数据
      const octave = HEAP32[(resultPtr + offset + 20) >> 2];
      const class_id = HEAP32[(resultPtr + offset + 24) >> 2];

      // 读取描述符
      const descriptor = new Uint8Array(32);
      for (let j = 0; j < 32; j++) {
        descriptor[j] = HEAPU8[resultPtr + offset + 28 + j];
      }

      points.push({
        x, // 特征点 x 坐标
        y, // 特征点 y 坐标
        size, // 特征点大小
        angle, // 特征点主方向,单位是弧度,用于表示特征点周围区域的主要朝向，使特征具有旋转不变性
        response, //特征点的响应值,值越大表示该特征点越显著，越容易被识别
        octave, //特征点所在的图像金字塔层级,用于多尺度特征检测,较高的 octave 对应较低分辨率的图像层级
        class_id, //特征点的类别标识符,可用于对特征点进行分类,在 ORB 特征中通常不太使用
        descriptor, //特征点的描述符,是一个二进制描述向量，用于特征匹配,在 ORB 中是 32 字节（256 位）的二进制串
      });
      offset += 28 + 32; // 28字节特征点数据 + 32字节描述符，正确指向内存中下一个特征点数据的起始位置
    }

    window._free(dataPtr);
    window._free(resultPtr);

    return points;
  } catch (error) {
    console.error("Error detecting features:", error);
    return [];
  }
};

// 在画布上绘制特征点
export const drawFeaturePoints = (
  imageElement: HTMLImageElement,
  ctx: CanvasRenderingContext2D,
  points: FeaturePoint[]
) => {
  // 先绘制原始图像
  ctx.drawImage(imageElement, 0, 0);

  // 绘制特征点
  ctx.fillStyle = "red";
  ctx.strokeStyle = "yellow";

  points.forEach((point) => {
    ctx.beginPath();
    ctx.arc(point.x, point.y, 3, 0, 2 * Math.PI);
    ctx.fill();

    // 绘制方向线
    if (point.angle !== -1) {
      const length = 10;
      const endX = point.x + length * Math.cos(point.angle);
      const endY = point.y + length * Math.sin(point.angle);

      ctx.beginPath();
      ctx.moveTo(point.x, point.y);
      ctx.lineTo(endX, endY);
      ctx.stroke();
    }
  });
};
