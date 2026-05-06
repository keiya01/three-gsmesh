import { Matrix3, Matrix4, Quaternion, Vector3 } from "three/webgpu";

type Covariance = [
  c00: number,
  c10: number,
  c20: number,
  c11: number,
  c21: number,
  c22: number,
];

const ZERO = new Vector3();
const scale = new Vector3();
const rotation = new Quaternion();
const transform4 = new Matrix4();
const transform3 = new Matrix3();
const transform3Transpose = new Matrix3();
const covariance = new Matrix3();

export function computeCovariance(
  scaleX: number,
  scaleY: number,
  scaleZ: number,
  quatX: number,
  quatY: number,
  quatZ: number,
  quatW: number,
) : Covariance {
  scale.set(scaleX, scaleY, scaleZ);
  rotation.set(quatX, quatY, quatZ, quatW);

  transform4.compose(ZERO, rotation, scale);
  transform3.setFromMatrix4(transform4);
  transform3Transpose.copy(transform3).transpose();
  covariance.multiplyMatrices(transform3, transform3Transpose);

  const e = covariance.elements;
  return [
    e[0],
    e[1],
    e[2],
    e[4],
    e[5],
    e[8],
  ];
}
