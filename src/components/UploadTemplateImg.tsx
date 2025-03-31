import { urlToImageData } from "@/utils";
import { Button, Drawer, Upload } from "@arco-design/web-react";
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

export function UploadTemplateImg() {
  const [visible, setVisible] = useState(false);
  const [imgSrc, setImgSrc] = useState<string>();
  // 处理参考图像上传
  useEffect(() => {
    if (imgSrc) {
      urlToImageData(imgSrc).then((imageData) => {
        if (imageData) {
          uploadReferenceImage(imageData);
        }
      });
    }
  }, [imgSrc]);

  return (
    <>
      <Button
        className="!fixed top-20 left-0 z-10"
        onClick={() => setVisible(true)}
      >
        展开
      </Button>
      <Drawer
        width={332}
        title="上传匹配图像"
        visible={visible}
        onOk={() => {
          setVisible(false);
        }}
        onCancel={() => {
          setVisible(false);
        }}
      >
        <Upload
          autoUpload={false}
          showUploadList={false}
          style={{ marginTop: 40 }}
          onChange={(_, uploadFile) => {
            const file = uploadFile.originFile;
            if (file) {
              const imageUrl = URL.createObjectURL(file);
              setImgSrc(imageUrl);
            }
          }}
        >
          <Button>上传参考图像</Button>
        </Upload>
        {imgSrc ? (
          <img
            src={imgSrc}
            alt="参考图像"
            style={{
              width: "100%",
              display: "block",
            }}
          />
        ) : null}
      </Drawer>
    </>
  );
}
