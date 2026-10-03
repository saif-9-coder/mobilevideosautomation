# Reference video style — "Evolution of Samsung Galaxy A Series (2014-2025)" by Techvolution

Source: https://www.youtube.com/watch?v=aY2_O7J_gEI (analyzed 2026-10-03)

## Canvas
- 16:9, 1920x1080. Static flat light blue-gray background **#C7D0DE** with a subtle
  darker wavy grid/line pattern. Background never animates.

## Per-phone card (~20s each in reference; configurable in builder)
1. Title beat: phone name + "Released YYYY, Mon" fade in on empty background.
2. ~2s later: spec block + phone renders + top badges fade in (soft fade + slight scale/blur ease).
3. Hold ~15s, then blur-crossfade to next phone.

## Layout
- **Title**: top-left, very large bold rounded geometric sans (Nunito/Quicksand ExtraBold style),
  near-black. E.g. "Galaxy A56 5G".
- **Release date**: below title, medium gray, smaller. E.g. "Released 2025, Mar".
- **Top-right badges**: small icon + text — refresh rate ("120Hz"), Android version
  ("Android 15"), chipset ("Snapdragon 6 Gen 3").
- **Spec grid**: rows with gray thin-line outline icon + small gray CAPS label +
  big bold black value + smaller bold black sub-line:
  - DISPLAY: `6.7"` / `Super AMOLED`
  - CAMERA: `50/12/5 MP` / `FRONT 32 MP`
  - STORAGE: `128/256 GB`
  - RAM: `8/12 GB`
  - BATTERY: `5.000 mAh` (dot as thousands separator)
  - WEIGHT: `195 g`
- **Phone renders**: right ~40% of canvas, front + back + side overlapping,
  transparent background, soft drop shadow. (We render the single GSMArena image,
  background-removed.)
- **Year marker**: huge bold black year ("2014".."2026"), bottom center-right,
  partially behind content.
- NOT shown: price, colors, dimensions, IP rating, charging speed.

## Motion / audio
- Crossfade with soft blur between phones; no Ken Burns on images; static background.
- No voiceover — instrumental electronic music bed only (builder accepts `--music`).
- Intro: thumbnail-style title card ("BRAND / SERIES / EVOLUTION" + phone lineup).
- Outro: final card holds; end-screen cards fade in.
