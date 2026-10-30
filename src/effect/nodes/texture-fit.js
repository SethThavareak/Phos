import { Fn, screenSize, screenUV, vec2 } from 'three/tsl'
import { mapNode, uFocus } from '../textures.js'

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
    .add(uFocus.sub(0.5).mul(scale.oneMinus()))
})
