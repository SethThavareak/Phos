import { uniform } from 'three/tsl'
import { AmbientLight, PointLight } from 'three/webgpu'

const FOLLOW_EASE = 0.12
const IDLE_AFTER_MS = 4000
const REDUCED_MOTION = matchMedia('(prefers-reduced-motion: reduce)').matches

export const pointLight = new PointLight('#ffc98a', 4.2, 0, 3.4)
pointLight.position.set(0, 0, 0.95)

export const ambientLight = new AmbientLight('#8fa6d8', 0.16)

export const uLightPosition = uniform(pointLight.position)

const pointer = { x: 0, y: 0, lastMoved: -Infinity }

export function setupLight(scene) {
  scene.add(pointLight, ambientLight)

  const follow = (event) => {
    pointer.x = (event.clientX / innerWidth) * 2 - 1
    pointer.y = -((event.clientY / innerHeight) * 2 - 1)
    pointer.lastMoved = performance.now()
    document.body.classList.add('touched')
  }
  addEventListener('pointermove', follow)
  addEventListener('pointerdown', follow)
}

export function updateLight(time, camera) {
  const idle = time - pointer.lastMoved > IDLE_AFTER_MS
  const wide = innerWidth > innerHeight
  // Idle orbit, in NDC: around the figure on wide screens, up the middle on phones.
  const sway = idle && REDUCED_MOTION ? 0 : 1
  const targetX = idle
    ? (wide ? 0.42 : 0) + 0.3 * sway * Math.sin(time * 0.0004)
    : pointer.x
  const targetY = idle
    ? 0.1 + 0.3 * sway * Math.sin(time * 0.00063)
    : pointer.y

  const ease = REDUCED_MOTION ? 1 : FOLLOW_EASE
  const { position } = pointLight
  position.x += (targetX * camera.right - position.x) * ease
  position.y += (targetY * camera.top - position.y) * ease
}
