export interface FeaturePoint {
  x: number;
  y: number;
  size: number;
  angle: number;
  response: number;
  octave: number;
  class_id: number;
  descriptor?: Uint8Array; // 添加描述符字段
}
