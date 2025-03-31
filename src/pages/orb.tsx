import { VideoCanvas } from "@/components/VideoCanvas";
import { useFetchStreamCamera } from "@/hooks/useFetchStreamCamrea";
import { urlToImageData } from "@/utils";
import { Spin } from "@arco-design/web-react";
import { useEffect, useState } from "react";
const uploadReferenceImage = (imageData: ImageData): void => {
  // 分配内存
  const refImagePtr: number = _malloc(imageData.data.length);

  // 复制图像数据到 WebAssembly 内存
  HEAPU8.set(imageData.data, refImagePtr);

  // 调用 C++ 函数设置参考图像
  _set_reference_image(refImagePtr, imageData.width, imageData.height);

  // 释放内存
  _free(refImagePtr);
};
export default function ORB() {
  const { loading, mediaStream } = useFetchStreamCamera();
  const [loadedImg, setLoadedImg] = useState(false);
  const allLoaded = loading || !loadedImg;
  useEffect(() => {
    urlToImageData("./test.jpg").then((imageData) => {
      if (imageData) {
        uploadReferenceImage(imageData);
        setLoadedImg(true);
      }
    });
  }, []);
  return (
    <div style={{ width: "100vw", height: "100vh", overflow: "hidden" }}>
      {allLoaded ? (
        <Spin tip="加载中" />
      ) : (
        <div>
          <VideoCanvas mediaStream={mediaStream} />
        </div>
      )}
    </div>
  );
}
