import { useRef, useState } from "react";
import { Button, Message, Upload } from "@arco-design/web-react";
import { detectFeatures, drawFeaturePoints } from "@/utils";
import { FeaturePoint } from "@/types/orbFeaturePoint";

export default function ImgFeatureDetector({
  featurePoints,
  setFeaturePoints,
}: {
  featurePoints: FeaturePoint[];
  setFeaturePoints: React.Dispatch<React.SetStateAction<FeaturePoint[]>>;
}) {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  // 处理图像上传
  const handleImageUpload = (file: File) => {
    setLoading(true);
    const reader = new FileReader();

    reader.onload = (e) => {
      const dataUrl = e.target?.result as string;
      setImageUrl(dataUrl);

      // 创建图像对象以获取尺寸
      const img = new Image();
      img.onload = () => {
        if (imageRef.current) {
          imageRef.current.width = img.width;
          imageRef.current.height = img.height;
        }

        if (canvasRef.current) {
          canvasRef.current.width = img.width;
          canvasRef.current.height = img.height;
          const ctx = canvasRef.current.getContext("2d");
          if (ctx) {
            ctx.drawImage(img, 0, 0);
            const points = detectFeatures(ctx, img.width, img.height);
            console.log(points);
            setLoading(false);
            if (imageRef.current && points) {
              setFeaturePoints(points);
              drawFeaturePoints(imageRef.current, ctx, points);
            }
          }
        }
      };
      img.src = dataUrl;
    };

    reader.onerror = () => {
      Message.error("图像加载失败");
      setLoading(false);
    };

    reader.readAsDataURL(file);
    return false; // 阻止默认上传行为
  };

  return (
    <div style={{ padding: "20px" }}>
      <h2>特征点检测</h2>

      <div style={{ marginBottom: "20px" }}>
        <Upload
          autoUpload={false}
          showUploadList={false}
          style={{ marginTop: 40 }}
          onChange={(_, uploadFile) => {
            const file = uploadFile.originFile;
            if (file) {
              handleImageUpload(file);
            }
          }}
        >
          <Button loading={loading}>上传参考图像</Button>
        </Upload>
      </div>

      <div style={{ display: "flex", gap: "20px" }}>
        {imageUrl && (
          <>
            <div>
              <h3>原始图像</h3>
              <img
                ref={imageRef}
                src={imageUrl}
                alt="原始图像"
                style={{ maxWidth: "100%" }}
              />
            </div>

            <div>
              <h3>特征点检测结果</h3>
              <canvas
                ref={canvasRef}
                style={{ border: "1px solid #ccc", maxWidth: "100%" }}
              />
            </div>

            <div>
              <h3>检测到的特征点 ({featurePoints.length})</h3>
              <div
                style={{
                  maxHeight: "300px",
                  overflow: "auto",
                  border: "1px solid #eee",
                  padding: "10px",
                }}
              >
                {featurePoints.length > 0 ? (
                  <table style={{ width: "100%", borderCollapse: "collapse" }}>
                    <thead>
                      <tr>
                        <th
                          style={{
                            border: "1px solid #ddd",
                            padding: "8px",
                            textAlign: "left",
                          }}
                        >
                          序号
                        </th>
                        <th
                          style={{
                            border: "1px solid #ddd",
                            padding: "8px",
                            textAlign: "left",
                          }}
                        >
                          X
                        </th>
                        <th
                          style={{
                            border: "1px solid #ddd",
                            padding: "8px",
                            textAlign: "left",
                          }}
                        >
                          Y
                        </th>
                        <th
                          style={{
                            border: "1px solid #ddd",
                            padding: "8px",
                            textAlign: "left",
                          }}
                        >
                          大小
                        </th>
                        <th
                          style={{
                            border: "1px solid #ddd",
                            padding: "8px",
                            textAlign: "left",
                          }}
                        >
                          角度
                        </th>
                        <th
                          style={{
                            border: "1px solid #ddd",
                            padding: "8px",
                            textAlign: "left",
                          }}
                        >
                          响应值
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {featurePoints.map((point, index) => (
                        <tr key={index}>
                          <td
                            style={{
                              border: "1px solid #ddd",
                              padding: "8px",
                            }}
                          >
                            {index + 1}
                          </td>
                          <td
                            style={{
                              border: "1px solid #ddd",
                              padding: "8px",
                            }}
                          >
                            {point.x.toFixed(2)}
                          </td>
                          <td
                            style={{
                              border: "1px solid #ddd",
                              padding: "8px",
                            }}
                          >
                            {point.y.toFixed(2)}
                          </td>
                          <td
                            style={{
                              border: "1px solid #ddd",
                              padding: "8px",
                            }}
                          >
                            {point.size.toFixed(2)}
                          </td>
                          <td
                            style={{
                              border: "1px solid #ddd",
                              padding: "8px",
                            }}
                          >
                            {point.angle.toFixed(2)}
                          </td>
                          <td
                            style={{
                              border: "1px solid #ddd",
                              padding: "8px",
                            }}
                          >
                            {point.response.toFixed(4)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <p>暂无特征点数据</p>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
