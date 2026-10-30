import { defineConfig } from 'vite'

export default defineConfig({
  base: './',
  optimizeDeps: { exclude: ['three'] },
})
