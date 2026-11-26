# Task 2 experiment log (dev split only: 88 images; GT = SAM masks)

Metrics are means over dev images. IoU/Dice/BF = semantic (all eggs merged) overlap and boundary-F1 (3 px); inst-F1 = per-egg F1 at IoU>=0.5; count-acc = fraction of images with the right number of eggs.

| id | change | dev IoU | Dice | BF | inst-F1 | count-acc | decision |
|---|---|---|---|---|---|---|---|
| T0 | Gray + Otsu (baseline) | 0.319 | 0.430 | 0.200 | 0.255 | 0.32 | reference |
| T1 | HSV-V channel + Otsu | 0.287 | 0.388 | 0.190 | 0.235 | 0.33 | discarded |
| T2 | Whiteness (V - 0.8*S) + Otsu | 0.317 | 0.427 | 0.196 | 0.243 | 0.33 | discarded alone, reused later |
| T3 | Lab L* + Otsu | 0.277 | 0.383 | 0.171 | 0.196 | 0.28 | discarded |
| T4 | Adaptive (local mean) threshold | 0.162 | 0.256 | 0.165 | 0.039 | 0.01 | discarded |
| T5 | T0 + shape/size filter | 0.189 | 0.211 | 0.138 | 0.225 | 0.24 | kept as a component |
| T6 | Colour distance from border colours | 0.225 | 0.272 | 0.141 | 0.278 | 0.35 | discarded as a generator, feature reused |
| T7 | Multi-threshold sweep + shape score (gray) | 0.420 | 0.517 | 0.265 | 0.408 | 0.57 | kept |
| T8 | Canny edge loops + shape score | 0.200 | 0.213 | 0.200 | 0.207 | 0.14 | discarded |
| T9 | T7 + boundary-contrast score | 0.438 | 0.535 | 0.282 | 0.450 | 0.57 | kept |
| T10 | T9 + pool gray/whiteness/border-distance sweeps | 0.480 | 0.581 | 0.315 | 0.481 | 0.56 | kept |
| T11 | T10 + relaxed shape thresholds | 0.442 | 0.566 | 0.288 | 0.395 | 0.30 | discarded |
| T12 | T10 + white-egg colour gate | 0.494 | 0.596 | 0.319 | 0.485 | 0.59 | kept |
| T13 | T12 + stronger smoothing | 0.470 | 0.570 | 0.277 | 0.465 | 0.53 | discarded |
| T14 | T12 + prefer larger candidates | 0.497 | 0.593 | 0.332 | 0.523 | 0.60 | kept |
| T15 | T14 + GrabCut refinement | 0.573 | 0.652 | 0.472 | 0.617 | 0.59 | kept |
| T16 | T15 + GrabCut sanity limits | 0.559 | 0.640 | 0.458 | 0.583 | 0.59 | discarded |
| T17 | T16 + relaxed fallback when nothing found | 0.588 | 0.678 | 0.480 | 0.615 | 0.66 | informative only |
| T18 | T17 + centre prior | 0.587 | 0.680 | 0.477 | 0.606 | 0.67 | discarded |
| T19 | T15 + relaxed fallback (clean re-run) | 0.599 | 0.689 | 0.485 | 0.650 | 0.68 | kept |
| G1 | GrabCut grow 0.6 | 0.587 | 0.676 | 0.470 | 0.637 | 0.69 | discarded |
| G2 | GrabCut grow 1.5 | 0.600 | 0.690 | 0.487 | 0.637 | 0.67 | discarded (neutral) |
| G3 | GrabCut shrink 0.2 | 0.598 | 0.686 | 0.491 | 0.650 | 0.69 | discarded (neutral) |
| G4 | GrabCut shrink 0.5 | 0.605 | 0.693 | 0.494 | 0.651 | 0.68 | kept (in F1) |
| G5 | GrabCut 10 iterations | 0.615 | 0.702 | 0.517 | 0.655 | 0.66 | kept (in F1) |
| F1 | Combine: GrabCut 10 iters + shrink 0.5 | 0.618 | 0.703 | 0.512 | 0.666 | 0.68 | FINAL |
| F2 | F1 + 20 GrabCut iterations | 0.618 | 0.699 | 0.516 | 0.638 | 0.69 | discarded |

## Details

### T0 - Gray + Otsu (baseline)  (parent: -)
- **What changed:** Gaussian blur 5, grayscale, global Otsu threshold, open 5 / close 9, every component 0.4%-75% of the image kept
- **Why:** simplest possible classical baseline: eggs are brighter than most backgrounds
- **Parameters (delta vs parent):** `{}`
- **Dev result:** IoU 0.319, Dice 0.430, BF 0.200, inst-F1 0.255, count-acc 0.32, median IoU 0.24, images with IoU<0.5: 76%
- **Decision:** reference - Baseline, dev IoU 0.319. Failure analysis: (a) a white egg on a bright background merges with it into one huge blob that max_area rejects -> egg missed; (b) hands, tiles, metal pans selected as giant false positives.

### T1 - HSV-V channel + Otsu  (parent: T0)
- **What changed:** feature = V of HSV instead of gray
- **Why:** V is max(R,G,B): keeps white and cream eggs bright even if slightly tinted
- **Parameters (delta vs parent):** `{"feat": "v"}`
- **Dev result:** IoU 0.287, Dice 0.388, BF 0.190, inst-F1 0.235, count-acc 0.33, median IoU 0.22, images with IoU<0.5: 81%
- **Decision:** discarded - 0.287 < 0.319: V=max(R,G,B) brightens coloured backgrounds as well, so more background joins the egg blob.

### T2 - Whiteness (V - 0.8*S) + Otsu  (parent: T0)
- **What changed:** feature = V - 0.8*S (bright AND unsaturated)
- **Why:** eggs are white; skin, rugs, bowls are bright but saturated
- **Parameters (delta vs parent):** `{"feat": "whiteness"}`
- **Dev result:** IoU 0.317, Dice 0.427, BF 0.196, inst-F1 0.243, count-acc 0.33, median IoU 0.24, images with IoU<0.5: 76%
- **Decision:** discarded alone, reused later - 0.317 ~ T0 (no gain with a global threshold), but whiteness (V-0.8S) separates white from skin/rug, so it became one of the feature maps pooled in T10 and the video detector.

### T3 - Lab L* + Otsu  (parent: T0)
- **What changed:** feature = L* of CIELAB
- **Why:** L* is perceptual lightness, less dominated by green than gray
- **Parameters (delta vs parent):** `{"feat": "lab_l"}`
- **Dev result:** IoU 0.277, Dice 0.383, BF 0.171, inst-F1 0.196, count-acc 0.28, median IoU 0.23, images with IoU<0.5: 82%
- **Decision:** discarded - 0.277, worst of the single-channel thresholds; L* behaves like gray.

### T4 - Adaptive (local mean) threshold  (parent: T0)
- **What changed:** adaptive threshold block 51, C=-8 instead of global Otsu
- **Why:** global threshold fails when background brightness varies; local one only needs the egg to be brighter than its neighbourhood
- **Parameters (delta vs parent):** `{"thr": "adaptive"}`
- **Dev result:** IoU 0.162, Dice 0.256, BF 0.165, inst-F1 0.039, count-acc 0.01, median IoU 0.13, images with IoU<0.5: 93%
- **Decision:** discarded - 0.162, instance F1 0.039: a local-mean threshold fires on every texture edge and outlines instead of filling the egg.

### T5 - T0 + shape/size filter  (parent: T0)
- **What changed:** keep only components with area <= 50% of image, solidity >= 0.90, ellipse-fit IoU >= 0.85, axis ratio <= 2.2
- **Why:** T0 failures were giant background blobs and hands; eggs are small, convex, elliptical
- **Parameters (delta vs parent):** `{"max_area": 0.5, "min_solidity": 0.9, "min_ellipse_iou": 0.85, "max_axis_ratio": 2.2}`
- **Dev result:** IoU 0.189, Dice 0.211, BF 0.138, inst-F1 0.225, count-acc 0.24, median IoU 0.00, images with IoU<0.5: 78%
- **Decision:** kept as a component - IoU falls to 0.189 and instance F1 from 0.255 to 0.225 on its own, because rejecting giant blobs leaves nothing in their place. Kept only as a component: every later candidate generator (T6-T8 and the final sweep) needs the shape/size filter to rank candidates.

### T6 - Colour distance from border colours  (parent: T5)
- **What changed:** feature = min Lab distance to 3 k-means colour clusters of a 24 px border ring (x3), Otsu, plus T5 shape filter
- **Why:** eggs are usually centred; whatever colours appear on the border are background, so 'unlike the border' = egg
- **Parameters (delta vs parent):** `{"feat": "bgdist"}`
- **Dev result:** IoU 0.225, Dice 0.272, BF 0.141, inst-F1 0.278, count-acc 0.35, median IoU 0.00, images with IoU<0.5: 73%
- **Decision:** discarded as a generator, feature reused - 0.225: backgrounds with several colours (rug, keyboard, tiles) make 'different from the border' true almost everywhere. The bgdist feature was still useful as one of three maps in T10.

### T7 - Multi-threshold sweep + shape score (gray)  (parent: T5)
- **What changed:** threshold gray at levels 40..240 step 6; every component at every level is a candidate scored by ellipse-IoU*solidity; greedy NMS
- **Why:** no single threshold separates the egg in every image; some level usually isolates an egg-shaped blob
- **Parameters (delta vs parent):** `{"cand": "sweep"}`
- **Dev result:** IoU 0.420, Dice 0.517, BF 0.265, inst-F1 0.408, count-acc 0.57, median IoU 0.42, images with IoU<0.5: 58%
- **Decision:** kept - 0.420 (+0.10 over T0): sweeping thresholds and scoring components by ellipse-IoU x solidity finds an egg-shaped blob at some level even when no single threshold works. Remaining failures: the sweep ranks egg-shaped non-eggs (foil patch, thumb, cloth corner) above the egg.

### T8 - Canny edge loops + shape score  (parent: T5)
- **What changed:** Canny 30/90 on blurred gray, close gaps with 7 px ellipse, fill every closed contour, shape-score, NMS
- **Why:** white eggs on white backgrounds still have a faint closed shadow/edge outline that thresholds cannot see
- **Parameters (delta vs parent):** `{"cand": "edges"}`
- **Dev result:** IoU 0.200, Dice 0.213, BF 0.200, inst-F1 0.207, count-acc 0.14, median IoU 0.00, images with IoU<0.5: 78%
- **Decision:** discarded - 0.200: Canny edges rarely close into one loop around the egg; white-on-white eggs have almost no edge, and textured backgrounds produce many spurious loops.

### T9 - T7 + boundary-contrast score  (parent: T7)
- **What changed:** candidate score = ellipse-IoU*solidity * min(dE,40)/40, dE = Lab distance between mean colour inside and in a 9 px ring outside
- **Why:** T7 picked egg-shaped blobs with no real boundary (foil patch, cloth corner, thumb); a real egg differs from its surroundings
- **Parameters (delta vs parent):** `{"score": "shape_contrast"}`
- **Dev result:** IoU 0.438, Dice 0.535, BF 0.282, inst-F1 0.450, count-acc 0.57, median IoU 0.44, images with IoU<0.5: 53%
- **Decision:** kept - 0.438 (+0.018): requiring a visible colour boundary removes boundary-less shape coincidences.

### T10 - T9 + pool gray/whiteness/border-distance sweeps  (parent: T9)
- **What changed:** run the sweep on three feature maps (gray, whiteness, bgdist) and pool all candidates before NMS
- **Why:** eggs that are not brighter than the background (ash, tray) may still stand out in whiteness or colour-vs-border
- **Parameters (delta vs parent):** `{"sweep_feats": ["gray", "whiteness", "bgdist"]}`
- **Dev result:** IoU 0.480, Dice 0.581, BF 0.315, inst-F1 0.481, count-acc 0.56, median IoU 0.50, images with IoU<0.5: 50%
- **Decision:** kept - 0.480 (+0.042): the same egg is isolated in different feature maps (gray / whiteness / border distance); pooling candidates and letting NMS pick the best recovers ash and tray eggs.

### T11 - T10 + relaxed shape thresholds  (parent: T10)
- **What changed:** min solidity 0.90->0.85, min ellipse-IoU 0.85->0.80, axis ratio 2.2->2.5
- **Why:** textured/occluded eggs give ragged masks that fail the strict filter
- **Parameters (delta vs parent):** `{"min_solidity": 0.85, "min_ellipse_iou": 0.8, "max_axis_ratio": 2.5}`
- **Dev result:** IoU 0.442, Dice 0.566, BF 0.288, inst-F1 0.395, count-acc 0.30, median IoU 0.45, images with IoU<0.5: 57%
- **Decision:** discarded - 0.442 < 0.480, count accuracy fell 0.56 -> 0.30: relaxing solidity/ellipse thresholds admits many non-egg blobs.

### T12 - T10 + white-egg colour gate  (parent: T10)
- **What changed:** reject candidates whose mean HSV saturation > 70 (0-255)
- **Why:** eggs here are white/cream; coloured blobs (skin, rug, orange box) should lose
- **Parameters (delta vs parent):** `{"max_sat": 70}`
- **Dev result:** IoU 0.494, Dice 0.596, BF 0.319, inst-F1 0.485, count-acc 0.59, median IoU 0.53, images with IoU<0.5: 48%
- **Decision:** kept - 0.494 (+0.014): rejecting candidates with mean saturation > 70 removes skin, rugs and orange boxes. This encodes a dataset prior (white/cream eggs) and would fail for brown eggs.

### T13 - T12 + stronger smoothing  (parent: T12)
- **What changed:** Gaussian blur 5 -> 11 and open/close 5/9 -> 7/15
- **Why:** textured ash eggs fragment under thresholding; blurring merges the texture into one blob
- **Parameters (delta vs parent):** `{"blur": 11, "open_k": 7, "close_k": 15}`
- **Dev result:** IoU 0.470, Dice 0.570, BF 0.277, inst-F1 0.465, count-acc 0.53, median IoU 0.49, images with IoU<0.5: 51%
- **Decision:** discarded - 0.470 < 0.494: stronger blur/morphology merges eggs with neighbouring objects and blunts the boundary; the ash eggs were not fixed.

### T14 - T12 + prefer larger candidates  (parent: T12)
- **What changed:** score *= area_fraction ** 0.5
- **Why:** the sweep often returns the bright highlight of an egg (small, very egg-shaped) instead of the whole egg
- **Parameters (delta vs parent):** `{"area_w": 0.5}`
- **Dev result:** IoU 0.497, Dice 0.593, BF 0.332, inst-F1 0.523, count-acc 0.60, median IoU 0.58, images with IoU<0.5: 43%
- **Decision:** kept - 0.497 (+0.003 IoU, but boundary F1 0.319 -> 0.332 and instance F1 0.485 -> 0.523): preferring larger candidates stops the egg's bright highlight winning over the whole egg.

### T15 - T14 + GrabCut refinement  (parent: T14)
- **What changed:** GrabCut (5 iters) seeded from each selected candidate: sure-FG = candidate eroded 0.35 r, sure-BG = beyond 1.0 r from it
- **Why:** partial detections (egg highlights) should grow to the real egg boundary; ragged edges should snap to colour edges
- **Parameters (delta vs parent):** `{"grabcut": true}`
- **Dev result:** IoU 0.573, Dice 0.652, BF 0.472, inst-F1 0.617, count-acc 0.59, median IoU 0.66, images with IoU<0.5: 34%
- **Decision:** kept - 0.573 (+0.076, boundary F1 0.332 -> 0.472): the largest single gain. GrabCut grows highlight-only detections to the real egg boundary and snaps ragged edges to colour edges.

### T16 - T15 + GrabCut sanity limits  (parent: T15)
- **What changed:** fall back to the un-refined candidate if GrabCut result is > 2.5x the seed area or its ellipse-IoU < 0.80
- **Why:** in T15 GrabCut sometimes flooded the background (huge false positives) when the seed was wrong or the background was similar
- **Parameters (delta vs parent):** `{"gc_max_growth": 2.5, "gc_min_ellipse": 0.8}`
- **Dev result:** IoU 0.559, Dice 0.640, BF 0.458, inst-F1 0.583, count-acc 0.59, median IoU 0.65, images with IoU<0.5: 38%
- **Decision:** discarded - 0.559 < 0.573: rejecting large GrabCut growth (>2.5x) or non-elliptical results also rejects legitimate growth from small highlights.

### T17 - T16 + relaxed fallback when nothing found  (parent: T16)
- **What changed:** if no candidate passes, rerun with solidity >= 0.80, ellipse-IoU >= 0.70, no colour gate and keep the single best
- **Why:** every image has >= 1 egg; complete misses (IoU 0) cost far more than a wrong guess
- **Parameters (delta vs parent):** `{"fallback": true}`
- **Dev result:** IoU 0.588, Dice 0.678, BF 0.480, inst-F1 0.615, count-acc 0.66, median IoU 0.65, images with IoU<0.5: 34%
- **Decision:** informative only - 0.588: measured on top of the discarded T16 (fallback effect +0.029 vs T16). Superseded by the clean re-run T19.

### T18 - T17 + centre prior  (parent: T17)
- **What changed:** candidate score *= exp(-d^2 / (2*0.5^2)), d = distance of its centroid from the image centre (normalised)
- **Why:** eggs are photographed roughly centred; off-centre blobs (bag corners, cloth) are usually distractors
- **Parameters (delta vs parent):** `{"center_sigma": 0.5}`
- **Dev result:** IoU 0.587, Dice 0.680, BF 0.477, inst-F1 0.606, count-acc 0.67, median IoU 0.66, images with IoU<0.5: 35%
- **Decision:** discarded - 0.587 vs 0.588: the centre prior changes nothing measurable, and would hurt off-centre eggs and the video.

### T19 - T15 + relaxed fallback (clean re-run)  (parent: T15)
- **What changed:** fallback = True on top of T15 (T16's GrabCut limits and T18's centre prior are NOT included)
- **Why:** T17 measured the fallback on top of the discarded T16; this isolates its effect on the kept pipeline
- **Parameters (delta vs parent):** `{"fallback": true}`
- **Dev result:** IoU 0.599, Dice 0.689, BF 0.485, inst-F1 0.650, count-acc 0.68, median IoU 0.67, images with IoU<0.5: 31%
- **Decision:** kept - 0.599 (+0.026 vs T15): when no candidate passes, relax solidity/ellipse thresholds and keep the best one. Every dataset image contains an egg, so a guess beats an empty mask. Dataset-specific: for video frames with no egg this fallback must be off.

### G1 - GrabCut grow 0.6  (parent: T19)
- **What changed:** gc_grow 1.0 -> 0.6
- **Why:** smaller search radius = less leakage into background
- **Parameters (delta vs parent):** `{"gc_grow": 0.6}`
- **Dev result:** IoU 0.587, Dice 0.676, BF 0.470, inst-F1 0.637, count-acc 0.69, median IoU 0.66, images with IoU<0.5: 34%
- **Decision:** discarded - 0.587 < 0.599: a smaller GrabCut search radius prevents small seeds growing to the full egg.

### G2 - GrabCut grow 1.5  (parent: T19)
- **What changed:** gc_grow 1.0 -> 1.5
- **Why:** larger radius lets tiny highlight seeds grow to the whole egg
- **Parameters (delta vs parent):** `{"gc_grow": 1.5}`
- **Dev result:** IoU 0.600, Dice 0.690, BF 0.487, inst-F1 0.637, count-acc 0.67, median IoU 0.67, images with IoU<0.5: 32%
- **Decision:** discarded (neutral) - 0.600 vs 0.599: no measurable change, larger radius raises leakage risk.

### G3 - GrabCut shrink 0.2  (parent: T19)
- **What changed:** gc_shrink 0.35 -> 0.2 (sure-FG core eroded less)
- **Why:** bigger sure-foreground core anchors the colour model
- **Parameters (delta vs parent):** `{"gc_shrink": 0.2}`
- **Dev result:** IoU 0.598, Dice 0.686, BF 0.491, inst-F1 0.650, count-acc 0.69, median IoU 0.68, images with IoU<0.5: 31%
- **Decision:** discarded (neutral) - 0.598 vs 0.599: no change.

### G4 - GrabCut shrink 0.5  (parent: T19)
- **What changed:** gc_shrink 0.35 -> 0.5
- **Why:** smaller core lets GrabCut correct seeds that overshoot the egg
- **Parameters (delta vs parent):** `{"gc_shrink": 0.5}`
- **Dev result:** IoU 0.605, Dice 0.693, BF 0.494, inst-F1 0.651, count-acc 0.68, median IoU 0.69, images with IoU<0.5: 31%
- **Decision:** kept (in F1) - 0.605 (+0.006): a smaller sure-foreground core lets GrabCut correct seeds that overshoot.

### G5 - GrabCut 10 iterations  (parent: T19)
- **What changed:** gc_iters 5 -> 10
- **Why:** more EM iterations may converge to better colour models
- **Parameters (delta vs parent):** `{"gc_iters": 10}`
- **Dev result:** IoU 0.615, Dice 0.702, BF 0.517, inst-F1 0.655, count-acc 0.66, median IoU 0.68, images with IoU<0.5: 30%
- **Decision:** kept (in F1) - 0.615 (+0.016, boundary F1 0.485 -> 0.517): more EM iterations improve the colour models.

### F1 - Combine: GrabCut 10 iters + shrink 0.5  (parent: T19)
- **What changed:** gc_iters 10 and gc_shrink 0.5 together (G5 + G4)
- **Why:** the two best single-parameter changes from the grid; check they add up
- **Parameters (delta vs parent):** `{"gc_iters": 10, "gc_shrink": 0.5}`
- **Dev result:** IoU 0.618, Dice 0.703, BF 0.512, inst-F1 0.666, count-acc 0.68, median IoU 0.71, images with IoU<0.5: 31%
- **Decision:** FINAL - 0.618: G4 + G5 combine. Frozen as the final algorithm; this config is the only one evaluated on the eval split.

### F2 - F1 + 20 GrabCut iterations  (parent: F1)
- **What changed:** gc_iters 10 -> 20
- **Why:** the trend G5 > T19 suggests more iterations may still help; check for saturation
- **Parameters (delta vs parent):** `{"gc_iters": 20}`
- **Dev result:** IoU 0.618, Dice 0.699, BF 0.516, inst-F1 0.638, count-acc 0.69, median IoU 0.70, images with IoU<0.5: 31%
- **Decision:** discarded - 0.618 = F1, instance F1 fell 0.666 -> 0.638: iterations have saturated; 2x slower.
