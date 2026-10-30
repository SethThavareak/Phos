import {
  AgXToneMapping,
  Color,
  OrthographicCamera,
  Scene,
  WebGPURenderer,
} from 'three/webgpu'
import { setupLight, updateLight } from './effect/light.js'
import { createPlane } from './effect/plane.js'
import { setupTextures } from './effect/textures.js'
import depth from './textures/hero-depth.webp'
import map from './textures/hero.webp'

const MAX_PIXEL_RATIO = 2
const VIEW_HEIGHT = 4

const renderer = new WebGPURenderer({ antialias: true })
renderer.setPixelRatio(Math.min(devicePixelRatio, MAX_PIXEL_RATIO))
renderer.setSize(innerWidth, innerHeight)
renderer.toneMapping = AgXToneMapping

document.getElementById('app').append(renderer.domElement)
await renderer.init()

await setupTextures({ map, depth })

const scene = new Scene()
scene.background = new Color('#000000')

const camera = new OrthographicCamera()
camera.position.set(0, 0, 5)

setupLight(scene, camera)

const plane = createPlane()
scene.add(plane)

function onResize() {
  const aspect = innerWidth / innerHeight

  camera.top = VIEW_HEIGHT * 0.5
  camera.bottom = -camera.top
  camera.right = camera.top * aspect
  camera.left = -camera.right
  camera.updateProjectionMatrix()
  plane.scale.set(VIEW_HEIGHT * aspect, VIEW_HEIGHT, 1)
}

onResize()

addEventListener('resize', () => {
  renderer.setSize(innerWidth, innerHeight)
  onResize()
})

renderer.setAnimationLoop((time) => {
  updateLight(time, camera)
  renderer.render(scene, camera)
})

document.body.classList.add('ready')
