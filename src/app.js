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
import rickyDepth from './textures/hero-depth.webp'
import rickyMap from './textures/hero.webp'
import sethDepth from './textures/seth-depth.webp'
import sethMap from './textures/seth.webp'
import sokhengDepth from './textures/sokheng-depth.webp'
import sokhengMap from './textures/sokheng.webp'
import taxzyDepth from './textures/taxzy-depth.webp'
import taxzyMap from './textures/taxzy.webp'
import zhamptDepth from './textures/zhampt-depth.webp'
import zhamptMap from './textures/zhampt.webp'

const MAX_PIXEL_RATIO = 2
const VIEW_HEIGHT = 4
const FADE_MS = 500 // keep in sync with the canvas transition in style.css

const people = {
  ricky: { map: rickyMap, depth: rickyDepth, focusY: 0.5, title: 'Ricky Davan — Product Designer' },
  // focusY 0 pins these to the image's bottom edge: on wide windows only what's above the head gets cropped, never the body.
  // wallDepth: keep in sync with the depth each image was built with in tools/.
  sokheng: { map: sokhengMap, depth: sokhengDepth, focusY: 0, wallDepth: 0.4, title: 'Soeng Sokheng — Software Developer' },
  taxzy: { map: taxzyMap, depth: taxzyDepth, focusY: 0, wallDepth: 0.4, title: 'Taxzy — Product Developer' },
  zhampt: { map: zhamptMap, depth: zhamptDepth, focusY: 0, wallDepth: 0.4, title: 'Zhampt — Husband' },
  // focusX keeps a figure left of the frame in view on phones.
  seth: { map: sethMap, depth: sethDepth, focusX: 0.82, focusY: 0, wallDepth: 0.4, title: 'Seth Thavareak' },
}

function showCopy() {
  const hash = location.hash.slice(1)
  const id = Object.hasOwn(people, hash) ? hash : 'ricky'

  document.body.dataset.person = id
  document.title = people[id].title
  for (const element of document.querySelectorAll('[data-for]')) {
    element.hidden = element.dataset.for !== id
  }
  for (const tab of document.querySelectorAll('#tabs a')) {
    tab.ariaCurrent = tab.hash === `#${id}` ? 'page' : null
    if (tab.ariaCurrent) tab.scrollIntoView({ block: 'nearest', inline: 'nearest' })
  }
  return people[id]
}

// Before the renderer starts, so the tabs still work if it can't.
const person = showCopy()
addEventListener('hashchange', showCopy)

const renderer = new WebGPURenderer({ antialias: true })
renderer.setPixelRatio(Math.min(devicePixelRatio, MAX_PIXEL_RATIO))
renderer.setSize(innerWidth, innerHeight)
renderer.toneMapping = AgXToneMapping

document.getElementById('app').append(renderer.domElement)
await renderer.init()

await setupTextures(person)

const scene = new Scene()
scene.background = new Color('#000000')

const camera = new OrthographicCamera()
camera.position.set(0, 0, 5)

setupLight(scene)

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

addEventListener('hashchange', async () => {
  const next = people[document.body.dataset.person]
  document.body.classList.remove('ready')
  await Promise.all([
    setupTextures(next),
    new Promise((resolve) => setTimeout(resolve, FADE_MS)),
  ])
  if (next === people[document.body.dataset.person]) {
    document.body.classList.add('ready')
  }
})
