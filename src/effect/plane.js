import {
  cameraFar,
  cameraNear,
  positionView,
  viewZToOrthographicDepth,
} from 'three/tsl'
import { Mesh, MeshPhongNodeMaterial, PlaneGeometry } from 'three/webgpu'
import { normalNode, uDisplacementScale } from './nodes/normal.js'
import { FIGURE_EDGE, shadowNode } from './nodes/shadow.js'
import { coverUv } from './nodes/texture-fit.js'
import { depthNode, mapNode, uWallDepth } from './textures.js'

const SKY_GLOW = 2
const FIGURE_DEPTH = 0.56

export function createPlane() {
  const vUv = coverUv()
  const depth = depthNode.sample(vUv).r

  const material = new MeshPhongNodeMaterial({ specular: 0x000000 })
  // Not aoNode: in three r184 that only touches ambient, so the point light would cast no shadow.
  // The figure is exempt: guessed self-shadow on a face reads as a bruise, not as light.
  // A cutout's depth is its matte, so edge pixels take shadow only for the share of them that is wall.
  const exempt = uWallDepth
    .greaterThan(0)
    .select(depth.sub(uWallDepth).div(FIGURE_EDGE).clamp(0, 1), depth.smoothstep(FIGURE_DEPTH, FIGURE_DEPTH + 0.04))
  const shadow = shadowNode(vUv, depth).add(exempt).min(1)
  material.colorNode = mapNode.sample(vUv).rgb.mul(shadow)
  material.normalNode = normalNode(vUv)
  // The sky is far behind the plane, so it glows on its own instead of catching the light.
  material.emissiveNode = mapNode.sample(vUv).rgb.mul(SKY_GLOW).mul(depth.smoothstep(0.06, 0))
  material.depthNode = viewZToOrthographicDepth(
    positionView.z.add(depth.sub(1).mul(uDisplacementScale)),
    cameraNear,
    cameraFar,
  )


  return new Mesh(new PlaneGeometry(1, 1), material)
}
