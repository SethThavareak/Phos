# Intro pages

One-screen intros behind two tabs, each a photo reworked into a night scene that a moving light relights: Ricky Davan at Angkor Wat and Soeng Sokheng against a dark studio wall. Link to a tab with `#ricky` or `#sokheng`.

```bash
npm install
npm run dev
```

`npm run build` writes the static site to `dist/`. Names, roles and descriptions are in `index.html`.

## Regenerating the images

Only needed if a source photo changes (the originals are not in this repo). Needs Python with torch, transformers, opencv and Pillow; all steps run locally.

```bash
python tools/masks.py photo_2026-10-30_16-24-59.jpg maps
python tools/build_hero.py photo_2026-10-30_16-24-59.jpg maps src/textures
```

```bash
python tools/masks.py photo_sokheng.jpg maps/sokheng
python tools/build_studio.py photo_sokheng.jpg maps/sokheng src/textures
```

## Credits

Relighting effect adapted from [codrops-relightning-images](https://github.com/DGFX/codrops-relightning-images) by Dominik Fojcik (the repo's README states MIT). Depth from [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), segmentation from SegFormer.
