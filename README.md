# Ricky Davan — intro page

One-screen intro: a daylight photo at Angkor Wat, reworked into a night scene that a moving light relights.

```bash
npm install
npm run dev
```

`npm run build` writes the static site to `dist/`. Name, role and description are in `index.html`.

## Regenerating the image

Only needed if the source photo changes (the original is not in this repo). Needs Python with torch, transformers, opencv and Pillow; both steps run locally.

```bash
python tools/masks.py photo_2026-10-30_16-24-59.jpg maps
python tools/build_hero.py photo_2026-10-30_16-24-59.jpg maps src/textures
```

## Credits

Relighting effect adapted from [codrops-relightning-images](https://github.com/DGFX/codrops-relightning-images) by Dominik Fojcik (the repo's README states MIT). Depth from [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), segmentation from SegFormer.
