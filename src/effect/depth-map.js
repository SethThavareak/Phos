import { texture } from 'three/tsl'
import {
  DataTexture,
  DataUtils,
  HalfFloatType,
  LinearFilter,
  RedFormat,
} from 'three/webgpu'
import { smoothBands } from '../lib/blur.js'

const depthSmoothing = { percent: 0.25 }

const WORKING_WIDTH = 2048

const smoothDepthMap = new DataTexture(
  new Uint16Array([DataUtils.toHalfFloat(0.5)]),
  1,
  1,
  RedFormat,
  HalfFloatType,
)
smoothDepthMap.minFilter = LinearFilter
smoothDepthMap.magFilter = LinearFilter
smoothDepthMap.needsUpdate = true

export const smoothDepthNode = texture(smoothDepthMap)

const canvas = document.createElement('canvas')
const context = canvas.getContext('2d', { willReadFrequently: true })

let depthImage = null

export function setDepthImage(image) {
  depthImage = image
  if (!depthImage) return

  const scale = Math.min(1, WORKING_WIDTH / depthImage.width)
  const width = Math.max(1, Math.round(depthImage.width * scale))
  const height = Math.max(1, Math.round(depthImage.height * scale))

  canvas.width = width
  canvas.height = height
  context.setTransform(1, 0, 0, -1, 0, height)
  context.drawImage(depthImage, 0, 0, width, height)

  const { data } = context.getImageData(0, 0, width, height)
  const values = new Float32Array(width * height)
  for (let i = 0; i < values.length; i++) {
    values[i] = data[i * 4] / 255
  }

  smoothBands(
    { values, width, height },
    Math.round((depthSmoothing.percent / 100) * width),
  )

  const halfFloats = new Uint16Array(values.length)
  for (let i = 0; i < halfFloats.length; i++) {
    halfFloats[i] = DataUtils.toHalfFloat(values[i])
  }

  smoothDepthMap.dispose()
  smoothDepthMap.image = { data: halfFloats, width, height }
  smoothDepthMap.needsUpdate = true
}
