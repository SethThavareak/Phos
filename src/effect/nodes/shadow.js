import {
  Fn,
  Loop,
  float,
  modelScale,
  positionWorld,
  uniform,
  vec2,
  vec3,
} from 'three/tsl'
import { smoothDepthNode } from '../depth-map.js'
import { uLightPosition } from '../light.js'
import { depthNode, uWallDepth } from '../textures.js'
import { uDisplacementScale } from './normal.js'
import { coverScaleNode } from './texture-fit.js'

export const uShadowIntensity = uniform(0.55)
export const uShadowSoftness = uniform(0.3)

const SHADOW_STEPS = 12
const MIN_LIGHT_ANGLE = 0.15
const SOFTNESS_GROWTH = 3

const FIGURE_STANDOFF = 0.3 // how far the figure stands off the wall
const FIGURE_EDGE = 0.22 // depth from the wall to the figure's edge
const CUTOUT_SHADOW_INTENSITY = 0.82
const PENUMBRA_LOD = 4.9
const PENUMBRA_GROWTH = 1.2
const PENUMBRA_TAP = 0.012

// A flat wall has no relief for the march to catch, only a jagged fringe. Against a flat wall a figure's
// shadow is just its silhouette seen from the light, softened by reading the depth map's mips.
const cutoutShadowNode = Fn(([vUv, depth]) => {
  const surfaceToLight = uLightPosition.sub(
    vec3(positionWorld.xy, depth.sub(1).mul(uDisplacementScale)),
  )
  const reach = surfaceToLight.xy
    .div(surfaceToLight.z.max(MIN_LIGHT_ANGLE))
    .mul(FIGURE_STANDOFF)
  const lod = float(PENUMBRA_LOD).add(reach.length().mul(PENUMBRA_GROWTH)).min(6.9)
  const center = vUv.add(reach.mul(coverScaleNode()).div(modelScale.xy))

  // Four taps hide the blocky mips.
  const silhouette = (x, y) =>
    depthNode
      .sample(center.add(vec2(x, y).mul(PENUMBRA_TAP)))
      .level(lod)
      .r.sub(uWallDepth)
      .div(FIGURE_EDGE)
      .clamp(0, 1)
  const occlusion = silhouette(-1, -1)
    .add(silhouette(1, -1))
    .add(silhouette(-1, 1))
    .add(silhouette(1, 1))
    .mul(0.25)

  return occlusion.mul(CUTOUT_SHADOW_INTENSITY).oneMinus()
})

export const shadowNode = Fn(([vUv, depth]) => {
  const surfacePosition = vec3(
    positionWorld.xy,
    depth.sub(1).mul(uDisplacementScale),
  )
  const surfaceToLight = uLightPosition.sub(surfacePosition)
  const lightDirection = surfaceToLight.div(surfaceToLight.length().max(0.001))
  const surfaceDepth = smoothDepthNode.sample(vUv).r
  const remainingDepth = surfaceDepth.oneMinus()

  const rayOffset = lightDirection.xy
    .div(lightDirection.z.max(MIN_LIGHT_ANGLE))
    .mul(uDisplacementScale)
    .mul(coverScaleNode())
    .div(modelScale.xy)
    .mul(remainingDepth)

  const maxOcclusion = float(0).toVar()

  Loop(SHADOW_STEPS, ({ i }) => {
    const rayProgress = float(i).add(1).div(SHADOW_STEPS)
    const rayDepth = surfaceDepth.add(remainingDepth.mul(rayProgress))
    const blockerDepth = smoothDepthNode.sample(
      vUv.add(rayOffset.mul(rayProgress)),
    ).r
    const sampleSoftness = uShadowSoftness.mul(
      rayProgress.mul(SOFTNESS_GROWTH).add(1),
    )
    const sampleOcclusion = blockerDepth
      .sub(rayDepth)
      .div(sampleSoftness)
      .clamp(0, 1)

    maxOcclusion.assign(maxOcclusion.max(sampleOcclusion))
  })

  const lightFacingMask = lightDirection.z.greaterThan(0).select(1, 0)

  return uWallDepth
    .greaterThan(0)
    .select(
      cutoutShadowNode(vUv, depth),
      maxOcclusion.mul(uShadowIntensity).mul(lightFacingMask).oneMinus(),
    )
})
