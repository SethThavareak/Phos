# Intro pages

One-screen intros behind tabs, each a photo reworked into a night scene that a moving light relights: Ricky Davan at Angkor Wat, Soeng Sokheng against a dark studio wall, Taxzy under a wooden gate, Zhampt in a dark room, Reggy on the rocks above a waterfall and Seth Thavareak at a poster wall. Link to a tab with `#ricky`, `#sokheng`, `#taxzy`, `#zhampt`, `#reggy` or `#seth`.

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
python tools/even_light.py photo_seth.jpg maps/seth photo_seth_even.jpg
python tools/build_night.py photo_seth_even.jpg maps/seth src/textures seth 0 3 0.4 0.52
```

`even_light.py` divides the hard sunlight out of the photo first, so the moving light isn't fighting a shadow baked into the pixels.

`photo_seth.jpg` is the HEIC original with the top 15% and right 18% cropped off, at 1302×1800.

```bash
python tools/masks.py photo_reggy.jpg maps/reggy
python tools/build_night.py photo_reggy.jpg maps/reggy src/textures reggy 0 0 0.4
python tools/fix_treeline.py src/textures/reggy.webp src/textures/reggy-depth.webp
```

`photo_reggy.jpg` is the HEIC original (4284×5712) cropped to x 279–3849, y 876–5636 and resized to 1350×1800. The crop is needed: uncropped, he fills under 10% of the frame and `build_night.py` stops with "lost the figure". `fix_treeline.py` paints over the pale overcast sky left between the treetops, which would otherwise show as a white halo against the night sky.

```bash
python tools/masks.py photo_sokheng.jpg maps/sokheng
python tools/build_studio.py photo_sokheng.jpg maps/sokheng src/textures
```

## Credits

Relighting effect adapted from [codrops-relightning-images](https://github.com/DGFX/codrops-relightning-images) by Dominik Fojcik (the repo's README states MIT). Depth from [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), segmentation from SegFormer.
