# 036 · Images as arrays: HWC uint8 vs float, channels, grayscale ★

**Category:** Arrays in practice · **Difficulty:** Medium

## What it is

An image in NumPy is just an array: `(H, W)` for grayscale, `(H, W, 3)` for RGB, `(H, W, 4)`
with an alpha channel. The usual dtypes are **uint8** with values 0-255 (how files store
pixels) and **float32** with values 0.0-1.0 (how you compute with them).

## Why it exists

Every image library hands you arrays, but they don't agree on the details: channel order, value
range, axis order. Most image bugs are a wrong dtype (overflow, everything white), a wrong range,
or a wrong axis, not wrong math. Knowing the conventions makes those bugs obvious.

## How it works

**Indexing is (row, column) = (y, x)**, with the origin at the top-left:

```
           x = 0   1   2   3      (columns, width W = 4)
y = 0       .   .   .   .
y = 1       .   .   #   .         img[1, 2] is the pixel at x=2, y=1
y = 2       .   .   .   .
(rows, height H = 3)              img.shape == (3, 4, 3) for RGB
```

`img[..., 0]` is the red channel, `img[:, :, ::-1]` swaps RGB and BGR.

**uint8 to float and back:**

```python
f = img.astype(np.float32) / 255          # 0..255 -> 0.0..1.0
u = np.round(np.clip(f, 0, 1) * 255).astype(np.uint8)   # clip, scale, ROUND, cast
```

Skipping the clip turns 1.02 into 260, which wraps to 4 (a dark pixel in a bright area).
Skipping the round truncates: 0.999 · 255 = 254.7 becomes 254.

**Grayscale** is a weighted sum of the channels (the eye is most sensitive to green). The
classic ITU-R BT.601 weights:

```
gray = 0.299 R + 0.587 G + 0.114 B
```

`img @ np.array([0.299, 0.587, 0.114])` does this for every pixel at once: `@` contracts the
last axis, so `(H, W, 3) @ (3,)` gives `(H, W)`. It works for batches `(N, H, W, 3)` too.

**Alpha blending** puts a foreground over a background with a per-pixel opacity `alpha`
(1 = foreground only, 0 = background only):

```
out = alpha · fg + (1 - alpha) · bg
```

`alpha` is `(H, W)` but the images are `(H, W, 3)`: give alpha a trailing axis,
`alpha[..., None]` (015), so it broadcasts over the channels. Do the arithmetic in float:
`fg + bg` in uint8 overflows (002).

## Gotchas

- **uint8 arithmetic wraps:** `img + 50` turns 220 into 14. Convert to float (or a wider int) first.
- **Float images must be in 0..1.** `img.astype(np.float32)` without `/ 255` gives values up to
  255; most display and processing code then shows an all-white image.
- **OpenCV loads BGR**, PIL/matplotlib/imageio use RGB. `img[..., ::-1]` converts (it's a view,
  so call `np.ascontiguousarray` if a library needs contiguous memory).
- **Axis order:** NumPy/PIL/OpenCV use HWC; PyTorch models expect CHW (`np.moveaxis`, 014; 039).
- Sizes: OpenCV functions often take `(width, height)` while `img.shape` is `(height, width, ...)`.

## Builds on

002, 013, 015, 031.

## Your task

Implement in `problem.py`, with **no Python loops**:

1. `to_float(img)` - uint8 `0..255` -> float32 `0..1`; a float image is returned as float32
   with its values unchanged.
2. `to_uint8(img)` - float image -> uint8: clip to `[0, 1]`, scale by 255, round to nearest.
3. `rgb_to_gray(img)` - `(..., 3)` -> `(...)` with the BT.601 weights. uint8 input gives uint8
   (rounded); float input keeps its float dtype.
4. `alpha_blend(fg, bg, alpha)` - two uint8 `(H, W, 3)` images and a float `(H, W)` alpha ->
   uint8 `(H, W, 3)`, rounded.

## Hints

<details>
<summary>Hint 1</summary>

`to_float`: check `img.dtype == np.uint8`. `to_uint8`: `np.round(np.clip(img, 0, 1) * 255).astype(np.uint8)`.

</details>

<details>
<summary>Hint 2</summary>

`rgb_to_gray`: compute `img.astype(np.float64) @ weights`, then round and cast back for uint8 input.

</details>

<details>
<summary>Hint 3</summary>

`alpha_blend`: `a = alpha[..., None]`, compute in float64, `np.round`, `astype(np.uint8)`.

</details>

## Self-check

1. What is `np.uint8(220) + np.uint8(50)` as uint8? Why?
2. Why do you clip before casting a float image to uint8?
3. How does `(H, W, 3) @ (3,)` give an `(H, W)` result?
4. You load an image with OpenCV and the sky looks orange. What happened?

<details>
<summary>Answers</summary>

1. `14`: 270 wraps around modulo 256.
2. Values outside `[0, 1]` would map outside 0..255 and wrap (or be undefined) in the cast.
3. `@` with a 1-D right operand sums over the last axis of the left operand, once per pixel.
4. The channels are in BGR order; reverse them with `img[..., ::-1]`.

</details>

## Go deeper

- [Tutorial: linear algebra on n-dimensional arrays (uses an RGB image)](https://numpy.org/numpy-tutorials/content/tutorial-svd.html)
