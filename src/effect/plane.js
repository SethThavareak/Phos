import {
  cameraFar,
  cameraNear,
  positionView,
  vec3,
  viewZToOrthographicDepth,
} from 'three/tsl'
import { Mesh, MeshPhongNodeMaterial, PlaneGeometry } from 'three/webgpu'
import { diffuseNode } from './nodes/diffuse.js'
import { normalNode, uDisplacementScale } from './nodes/normal.js'
import { shadowNode } from './nodes/shadow.js'
import { coverUv } from './nodes/texture-fit.js'
import { depthNode, mapNode, uWallDepth } from './textures.js'

const SKY_GLOW = 2
const FIGURE_DEPTH = 0.56
const CUTOUT_FILL = vec3(0.09, 0.11, 0.16) // moonlit, so a portrait on a dark wall never drops to black but stays dimmer than the light

export function createPlane() {
  const vUv = coverUv()
  const depth = depthNode.sample(vUv).r

  const material = new MeshPhongNodeMaterial({ specular: 0x000000 })
  // Not aoNode: in three r184 that only touches ambient, so the point light would cast no shadow.
  // The figure is exempt: guessed self-shadow on a face reads as a bruise, not as light.
  const figure = depth.smoothstep(FIGURE_DEPTH, FIGURE_DEPTH + 0.04)
  const shadow = shadowNode(vUv, depth).add(figure).min(1)
  material.colorNode = diffuseNode(vUv, depth).mul(shadow)
  material.normalNode = normalNode(vUv)
  // The sky is far behind the plane, so it glows on its own instead of catching the light.
  const map = mapNode.sample(vUv).rgb
  const sky = map.mul(SKY_GLOW).mul(depth.smoothstep(0.06, 0))
  const fill = map.mul(CUTOUT_FILL).mul(figure).mul(uWallDepth.greaterThan(0).select(1, 0))
  material.emissiveNode = sky.add(fill)
  material.depthNode = viewZToOrthographicDepth(
    positionView.z.add(depth.sub(1).mul(uDisplacementScale)),
    cameraNear,
    cameraFar,
  )


  return new Mesh(new PlaneGeometry(1, 1), material)
}
