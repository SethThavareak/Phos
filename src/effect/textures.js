import { texture, uniform } from 'three/tsl'
import { SRGBColorSpace, Texture, TextureLoader, Vector2 } from 'three/webgpu'
import { setDepthImage } from './depth-map.js'

export const mapNode = texture(new Texture())
export const depthNode = texture(new Texture())
// Where the crop sits when the viewport can't show the whole image: keeps the figure in frame on phones.
export const uFocus = uniform(new Vector2(0.9, 0.5))
// Depth of a plain wall behind a cutout figure; 0 means the image has real depth to ray-march.
export const uWallDepth = uniform(0)

const loader = new TextureLoader()

let latestRequest = 0

export async function setupTextures({ map, depth, focusX = 0.9, focusY, wallDepth = 0 }) {
  const request = ++latestRequest
  const [mapTexture, depthTexture] = await Promise.all([
    loader.loadAsync(map),
    loader.loadAsync(depth),
  ])
  // Tabs can be clicked faster than images load; a stale pair must not overwrite the newer one.
  if (request !== latestRequest) {
    mapTexture.dispose()
    depthTexture.dispose()
    return
  }

  mapTexture.colorSpace = SRGBColorSpace
  mapNode.value.dispose()
  mapNode.value = mapTexture
  depthNode.value.dispose()
  depthNode.value = depthTexture
  setDepthImage(depthTexture.image)
  uFocus.value.set(focusX, focusY)
  uWallDepth.value = wallDepth
}
