import { Fn, step, uniform } from 'three/tsl'
import { mapNode } from '../textures.js'

const uDepthThreshold = uniform(0)

export const diffuseNode = Fn(([vUv, depth]) =>
  mapNode.sample(vUv).rgb.mul(step(uDepthThreshold, depth)),
)

