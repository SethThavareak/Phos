# Intro pages

One-screen intros behind tabs, each a photo reworked into a night scene that a moving light relights: Ricky Davan at Angkor Wat, Soeng Sokheng against a dark studio wall, Taxzy under a wooden gate, Zhampt in a dark room and Seth Thavareak at a poster wall. Link to a tab with `#ricky`, `#sokheng`, `#taxzy`, `#zhampt` or `#seth`.

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
python tools/masks.py photo_taxzy.jpg maps/taxzy
python tools/build_night.py photo_taxzy.jpg maps/taxzy src/textures taxzy 0 0 0.4
```

`photo_taxzy.jpg` is the HEIC original at 1350×1800.

```bash
python tools/masks.py photo_zhampt.jpg maps/zhampt
python tools/build_night.py photo_zhampt.jpg maps/zhampt src/textures zhampt 0 9 0.4
```

```bash
python tools/masks.py photo_seth.jpg maps/seth
python tools/build_night.py photo_seth.jpg maps/seth src/textures seth 0 0 0.4
```

`photo_seth.jpg` is the HEIC original with the top 15% and right 18% cropped off, at 1302×1800.

```bash
python tools/masks.py photo_sokheng.jpg maps/sokheng
python tools/build_studio.py photo_sokheng.jpg maps/sokheng src/textures
```

## Credits

Relighting effect adapted from [codrops-relightning-images](https://github.com/DGFX/codrops-relightning-images) by Dominik Fojcik (the repo's README states MIT). Depth from [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), segmentation from SegFormer.
