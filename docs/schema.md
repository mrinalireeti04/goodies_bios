# Output schema: `hands.jsonl`

`hand_detect.py` writes one JSON object per processed frame, one per line. Every line has exactly these keys, in this order. Nothing is added and nothing is left out.

```ts
type PS1Frame = {
  schema_version: "1.0";
  clip_id: string;
  frame_idx: number;
  t_ms: number;

  frame_width: number;
  frame_height: number;

  hands: {
    hand_id: string;
    side: "left" | "right" | "unknown";

    handedness_score: number | null;

    bbox_px: [number, number, number, number];

    kp2d_px: [number, number][]; // exactly 21

    kp3d_hand_m:
      [number, number, number][] | null; // exactly 21

    source: "detected" | "interpolated";
  }[];
};
```

## Field rules

| Field | Rule |
|---|---|
| `clip_id` | Video file name without its extension. |
| `frame_idx` | Index of the frame in the original video (0-based). With the default `--step 2` the values are 0, 2, 4 and so on. |
| `t_ms` | `round(frame_idx * 1000 / fps)`, an integer that strictly increases. |
| `frame_width`, `frame_height` | Size in pixels of the decoded original frame. |
| `hands` | Always present. `[]` when no hand is found. Ordered left, right, unknown. |
| `hand_id` | `L`, `R`, `U1`, `U2`. Unique inside one frame. It is not a track ID across frames. |
| `side` | MediaPipe's handedness. If two hands in one frame receive the same side, the higher-scoring hand keeps it and the other becomes `unknown`, so no hand is discarded. |
| `handedness_score` | MediaPipe's confidence in the left/right label, 0 to 1, 3 decimals. `null` if missing. |
| `bbox_px` | `[x1, y1, x2, y2]`: min and max of the 21 points. Not clipped to the image. |
| `kp2d_px` | 21 `[x, y]` pairs in MediaPipe landmark order, in pixels of the original frame (2 decimals). |
| `kp3d_hand_m` | 21 `[x, y, z]` triples from MediaPipe world landmarks, in metres, centred on the hand (not on the camera). `null` if unavailable. |
| `source` | `detected` for everything this function writes. `interpolated` is reserved for a later gap-filling step. |

## Coordinates

All `_px` values are in pixels of the original frame, even though detection runs on a copy resized to 480 px wide. MediaPipe returns normalized values and they are multiplied by `frame_width` and `frame_height`.

## Left and right

MediaPipe assumes a mirrored selfie image, so its labels are often reversed on head-camera footage. On the demo clips checked so far, the labels were already correct without swapping, so swapping is **off** by default. Use `--swap` to force it. The check used: in frames with both hands, the hand labelled left should be on the left of the image.

## Checking a file

```
python hand_detect.py --validate out/<clip_id>/hands.jsonl
```

This checks every line against the rules above, and that `frame_idx` and `t_ms` increase.

## Example

`examples/example_frame.json` shows the format with synthetic values. It is not derived from any footage.
