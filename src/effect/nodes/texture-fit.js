import { Fn, screenSize, screenUV, vec2 } from 'three/tsl'
import { mapNode } from '../textures.js'

// Where the crop sits when the viewport can't show the whole image: keeps the figure in frame on phones.
const FOCUS = vec2(0.9, 0.5)

export const coverScaleNode = Fn(() => {
  const viewAspect = screenSize.x.div(screenSize.y).toVar()
  const mapSize = vec2(mapNode.size()).toVar()
  const imageAspect = mapSize.x.div(mapSize.y).toVar()

  return imageAspect
    .greaterThan(viewAspect)
    .select(
      vec2(viewAspect.div(imageAspect), 1),
      vec2(1, imageAspect.div(viewAspect)),
    )
})

export const coverUv = Fn(() => {
  const scale = coverScaleNode().toVar()
  return screenUV
    .flipY()
    .sub(0.5)
    .mul(scale)
    .add(0.5)
    .add(FOCUS.sub(0.5).mul(scale.oneMinus()))
})
