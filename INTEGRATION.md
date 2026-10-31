# PS2 integration contract (for backend, frontend and PS5 teammates)

## 1. How to call it

```python
from ps2 import run_ps2

result = run_ps2(
    video_path=r"D:\Bios hack_ForgeStudio\Hack-data\<path to video>.mp4",
    hands=r"<path to PS1 output>.jsonl",   # path to PS1 output, OR the list of PS1Frame dicts in memory
)
# result is a dict matching PS2Output below. The backend decides where to save or forward it.
```

- `hands` accepts: a `.jsonl` path (one PS1Frame per line), a `.json` path (array of PS1Frame), or a Python `list[dict]` of PS1Frame.
- Runs on GPU (`cuda:0`) if available, otherwise CPU. Processes every frame.
- Raises `ValueError` with a clear message if inputs are missing or malformed.

## 2. Input from PS1 (owned by the PS1 teammate, unchanged)

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
    kp2d_px: [number, number][];                  // exactly 21
    kp3d_hand_m: [number, number, number][] | null; // exactly 21
    source: "detected" | "interpolated";
  }[];
};
```

How PS2 uses it:
| PS1 field | PS2 use |
|---|---|
| `clip_id` | copied to output `clip_id` (falls back to video file name) |
| `frame_idx`, `t_ms` | frames are matched by `frame_idx`; PS2 output reuses the same `frame_idx` and `t_ms` |
| `frame_width/height` | if different from the video size, PS2 rescales PS1 coordinates to video pixels |
| `hand_id` | copied as-is into PS2 `hand_states[].hand_id` and `contacts[].hand_id` |
| `side` | not needed for logic; PS2 never changes it |
| `kp2d_px` | fingertip distances, hand size, finger curl |
| `bbox_px` | removing the hand/glove from object detections |
| `source: "interpolated"` | contact confidence is lowered and reason `hand_interpolated` added |
| `kp3d_hand_m` | not used |

A frame missing from the PS1 data, or with an empty `hands` list, means "no hand seen" for PS2.

## 3. Output from PS2

`schema_version` stays `"1.0"`. All required fields are exactly the team schema. Fields marked **OPTIONAL** are extra: they may be absent, readers must not crash when they are missing, and they never change the meaning of required fields.

```ts
type PS2Output = {
  schema_version: "1.0";
  clip_id: string;

  meta?: {                                  // OPTIONAL
    models: string[];                       // e.g. ["yoloe-26s-seg.pt"]
    device: string;                         // e.g. "cuda:0 RTX 3050 Ti"
    frames_processed: number;
    processing_fps: number;
    vocab_size: number;
    vocab_source?: "ledger" | "gemini" | "base";  // OPTIONAL: where this video's object vocabulary came from (see section 6)
    gemini_requests?: number;               // OPTIONAL: Gemini generateContent requests used for this clip
    tool_ledger?: {                         // OPTIONAL: the validated tool ledger when vocab_source is "ledger" (section 6)
      tools: { tool_key: string; name: string; synonyms: string[]; appearance: string; type: "portable" | "fixed";
               powered: boolean; workpiece: boolean;
               uses: { start_s: number; end_s: number; hand: "left" | "right" | "both" | "unknown"; action: string;
                       certainty: "high" | "medium" | "low" }[] }[];
      running: { tool_key: string; start_s: number; end_s: number }[];
      unsure: { description: string; start_s: number; end_s: number }[];
      duration_s: number;
    };
    ledger_run?: { model: string; windows: number; requests: number; from_cache: boolean; seconds: number;
                   upload_deleted: boolean | null; temp_deleted: boolean | null; error: string | null };
    sam2?: { contacts: number; done: number; frames: number; seconds: number; peak_vram_gb: number | null; skipped_for_time: number };
    vocab?: {                               // OPTIONAL: only when vocab_source is "gemini"
      label: string;
      status: "confirmed" | "vlm_only";     // confirmed = the detector also found it on a frame that cites it
      hazard: boolean;
      seen_ms: number[];                    // times of the sampled frames where it was seen
    }[];
    ps1_checks?: {                          // OPTIONAL: result of the PS1 assumption checks (see section 5)
      ok: boolean;
      warnings: string[];                   // human-readable, one per violated assumption
      missing_frames: number; t_ms_mismatch: number; frame_idx_beyond_video: number;
      coords_out_of_bounds: number; frame_size_mismatch: number; keypoint_order_suspect: number;
      distinct_hand_ids: number;
    };
  };

  frames: {
    frame_idx: number;
    t_ms: number;
    objects: {
      object_id: string;                    // stable for the whole clip, incl. after leaving and re-entering view
      label: string;                        // specific name, or a generic name if unidentified
      type: "portable" | "fixed" | "unknown";
      bbox_px: [number, number, number, number];
      confidence: number | null;
      is_hazard?: boolean;                  // OPTIONAL: e.g. saw blade, grinder disc
      mask_polygon_px?: [number, number][] | null;  // OPTIONAL: only for objects currently in contact
    }[];
    hand_states: {
      hand_id: string;                      // from PS1
      state: "none" | "portable" | "fixed" | "unknown";
      object_id: string | null;             // null when state is "none", and may be null when "unknown"
      confidence: number | null;
      reasons?: string[];                   // OPTIONAL: why "unknown" or low confidence (codes in §4)
    }[];
  }[];

  contacts: {
    event_id: string;
    start_ms: number;                       // grab / start of touch
    end_ms: number;                         // release / end of touch
    hand_id: string;
    object_id: string | null;
    object_label: string | null;
    state: "portable" | "fixed";
    interaction: "touch" | "hold" | "unknown";
    confidence: number | null;
    reasons?: string[];                     // OPTIONAL
    review_needed?: boolean;                // OPTIONAL: true = show to a human reviewer
  }[];

  tools?: {                                 // OPTIONAL: one row per physical object
    object_id: string;
    label: string;
    label_status: "identified" | "unidentified";
    type: "portable" | "fixed" | "unknown";
    is_hazard: boolean;
    first_seen_ms: number;
    last_seen_ms: number;
    visible_ms: number;
    held_ms: number;                        // total time in "hold" contacts
    use_count: number;                      // number of hold contacts (gaps under 1 s merged)
    appearances: { start_ms: number; end_ms: number }[];  // each time it was in view
    idle_visible_ms?: number;               // OPTIONAL: visible_ms - held_ms, never below 0
    uses?: { start_ms: number; end_ms: number; hand_id: string; interaction: "hold" }[];  // OPTIONAL: hold contacts, gaps < 1 s merged
    tool_key?: string | null;               // OPTIONAL: the ledger's key for this tool; set only when "identified" by both kinds of evidence
    id_confidence?: "high" | "low";         // OPTIONAL: "high" only when time AND crop evidence agreed on the same tool
    evidence?: string[];                    // OPTIONAL: ledger_time, crop_match, crop_other, crop_different
  }[];

  review_queue?: {                          // OPTIONAL: what a human should check
    item_id: string;
    start_ms: number;
    end_ms: number;
    hand_id: string | null;
    object_id: string | null;
    reason: string;                         // a code from §4
    suggested_label: string | null;         // a name from the vision-language model for an unidentified object ("other:<name>" answers), never applied automatically; null if no key was available or the model had no answer
  }[];

  safety_events?: {                         // OPTIONAL, EXPERIMENTAL
    event_id: string;
    start_ms: number;
    end_ms: number;
    hand_id: string;
    hazard_object_id: string;
    min_distance_norm: number;              // closest fingertip distance, in hand-size units (image plane)
    hazard_moving: boolean | null;
    confidence: number | null;
    reasons: string[];                      // always includes "2d_distance_only"
  }[];
};
```

### Rules every consumer can rely on
- `"unknown"` is the only "unsure" value (the earlier team note's "unsure" = `"unknown"`).
- A hand with no contact: `state: "none"`, `object_id: null`.
- Every `object_id` used in `contacts`, `tools`, `review_queue`, `safety_events` exists in at least one `frames[].objects`.
- `start_ms < end_ms` for every interval. All times are milliseconds from video start.
- Grab and release are the start and end of `contacts` with `interaction: "hold"`. They are not separate events.
- PS2 never renames or merges anything after output. Renaming and "these are the same object" merging are frontend actions. When merging two `object_id`s, the frontend should re-sum `held_ms`, `visible_ms`, `use_count` and join `appearances`.

## 4. Reason codes (used in `reasons` and `review_queue.reason`)

| Code | Meaning |
|---|---|
| `hand_not_detected` | PS1 had no hand while a contact was ongoing |
| `hand_interpolated` | PS1 filled this hand in, it was not directly detected |
| `low_object_confidence` | the object detector was unsure what this is |
| `ambiguous_candidates` | two objects were equally likely to be the one touched |
| `grasp_pose_no_object` | the hand looks like it is gripping something, but no object was found |
| `fixed_contact_unverified` | hand over a table/machine, but not clearly resting on it |
| `type_overridden_by_motion` | labelled as fixed, but it moved with the hand, so treated as portable |
| `unidentified_object` | object found but no specific name |
| `low_contact_confidence` | the contact as a whole has low confidence (no single cause) |
| `vocab_unverified` | a label from the per-video vocabulary was never found by the detector on the frames that cite it |
| `vlm_unsure_object` | the vision-language model saw an object it could not name (`suggested_label` holds its visual description) |
| `label_from_vlm` | the object's label was chosen by the vision-language model (Gemini) from the vocabulary list, not by the detector |
| `identity_uncertain` | re-entry match between two appearances is not certain |
| `long_unknown` | hand state stayed unknown for more than 1 second |
| `2d_distance_only` | safety distance measured in the image, depth not known |
| `hazard_motion_estimated` | "machine part is moving" is estimated from video motion |
| `hazard_running_from_ledger` | the tool ledger says this machine was running at that time |
| `tool_single_evidence` | only one of the two kinds of evidence (ledger time, crop look) points to a tool, so it is not "identified"; `suggested_label` has the name |
| `tool_evidence_conflict` | the two kinds of evidence name different tools, or two objects seen together got the same tool |
| `tool_use_ai_only` | the ledger has a use of a tool at that time (`start_ms`/`end_ms` from the ledger) that no local contact explains; `object_id` is null |

## 5. Assumptions about PS1 data

PS2 relies on these properties of the PS1 output. `hands.py` checks the ones that can be checked when the data is loaded, logs a warning for each violation and records the counts in `meta.ps1_checks`. A violation never stops PS2.

1. `frame_idx` is 0-based in the original video.
2. `t_ms = round(frame_idx * 1000 / fps)`.
3. Every video frame has a record (a record with an empty `hands` list is fine). PS1 fills gaps in detection of up to 10 frames (`source: "interpolated"`).
4. **`hand_id` is NOT persistent** (PS1 teammate's answer): the same physical hand can get different ids over the clip. PS2 therefore builds its own hand tracks from `side`, box overlap and wrist distance, decides contacts per track, and writes `contacts[].hand_id` = the PS1 `hand_id` seen most often in the interval. `frames[].hand_states[].hand_id` keeps each frame's own PS1 `hand_id`.
5. Keypoints are in MediaPipe 21-point order.
6. **PS1 does NOT filter bystanders** (PS1 teammate's answer): PS2 keeps at most 2 likely wearer hands per frame (a hand smaller than half the largest, or with its wrist in the top 35% of the frame, is dropped; of the rest the larger / lower ones stay) and counts the drops in `meta.ps1_checks.bystander_hands_dropped`.
7. The image is not mirrored, and coordinates match `frame_width` x `frame_height`. The video should have a constant frame rate (checked with `ffprobe` if installed; warning only, recorded as `meta.ps1_checks.constant_frame_rate`).

Checked at load time: missing frames (assumption 3), `t_ms` more than one frame off (2), `frame_idx` beyond the video length (1), keypoints more than 10% of the frame size outside the frame (7; landmarks slightly beyond the edge are normal), a frame size different from the video's (7), more than 4 distinct `hand_id`s (4), and a simple plausibility test of keypoint order (5; a warning only if more than 30% of the hands fail it, since one foreshortened hand can). Not checkable: 6 and whether the image is mirrored.

## 6. The tool ledger and fusion (optional cloud step, the default vocabulary source)

Needs `GEMINI_API_KEY`. Gemini watches the whole clip once and writes a **tool ledger**; the local pipeline gives frame-exact hand/object evidence; a tool is `identified` only when both agree. Everything else goes to review.

1. **Upload.** A temporary reduced copy of the video (at most 720 p, no audio, CRF 30) is uploaded with the Gemini Files API (resumable upload, `urllib`); the uploaded file and the copy are deleted in a `finally` block and the outcome is recorded in `meta.ledger_run.upload_deleted` / `temp_deleted`. Videos longer than 300 s are clipped into windows with `video_metadata` start/end offsets (never prompt text) and merged.
2. **The request** (one `generateContent` call per window, `fps` 2, temperature 0, strict JSON schema, times as `MM:SS`) asks for `tools[]` (key, name, up to 3 synonyms, appearance, portable/fixed, powered cutting/rotating, workpiece, `uses[]` with start, end, hand, action, certainty), `running[]` (when a powered tool or machine ran) and `unsure[]` (handled objects it cannot name). The prompt forbids inferring tools from context and says the tool is the instrument in the hand, not the workpiece. The validated result is `meta.tool_ledger`; only JSON text is cached in `cache/`.
3. **Vocabulary.** The detector looks for the ledger's names and synonyms (workpieces excluded) plus the base file's fixed non-hazard surfaces, generic catch-alls and ignore list: `meta.vocab_source = "ledger"`. If the ledger is unavailable the base vocabulary is used (`"base"`).
4. **Fusion**, for every object with at least one contact: *time evidence* = ledger tools with a use overlapping its contacts within 1 s (same hand when both are known); *crop evidence* = Gemini looks at up to 8 object crops per request and answers per crop `tool_key | other:<name> | not_a_tool | unknown`. Same tool by both: label = tool name, `label_status` `identified`, `id_confidence` `high`, evidence `["ledger_time","crop_match"]`. Only one points to a tool: `unidentified`, review `tool_single_evidence`, `suggested_label` = that name. They name different tools: `unidentified`, review `tool_evidence_conflict`, `suggested_label` = the crop's answer. Fixed objects are never relabelled (the answer only becomes `suggested_label`). A held portable object with no ledger match is `unidentified` (review `unidentified_object`).
5. **Re-entry.** Objects identified as the same `tool_key` that are never visible together become one `object_id`; if they are visible together they stay separate and get a `tool_evidence_conflict` review item.
6. **Ledger uses with no matching local contact** become `tool_use_ai_only` review items (ledger times, `object_id` null, `suggested_label` = tool name); ledger `unsure[]` entries become `vlm_unsure_object` items.
7. **Safety.** Hazards are objects identified as ledger tools that are powered cutting/rotating, or fixed machines the ledger lists as running (and whose box is smaller than 15% of the frame). A hazard counts as moving when it is inside a ledger running interval OR local motion says so; the event's `reasons` say which (`hazard_running_from_ledger`, `hazard_motion_estimated`) plus `2d_distance_only`.
8. **Masks.** For hold contacts of portable objects, SAM 2.1 tiny tracks the object (box prompt at the contact start, through the contact plus 0.5 s) and `frames[].objects[].mask_polygon_px` (at most 40 points) is written on those frames; the polygon is also used in the fingertip distances. `meta.sam2` has time and peak VRAM.
9. All Gemini requests of a clip share one budget (`vlm_max_calls_per_clip`): the ledger first, then crops by longest contact.

## 7. Naming of held objects and the per-video vocabulary (earlier optional cloud steps)

These two steps remain in the code: the per-video vocabulary from 24 sampled frames (`vocab_auto_enabled`, off by default) and, when no ledger is available, the naming of held objects from their crops (`vlm_naming_enabled`). Both need the environment variable `GEMINI_API_KEY`. Without it, or if the API keeps failing, PS2 uses the base vocabulary file and skips naming; the reason is logged once. Requests are limited per clip (`vlm_max_calls_per_clip`) and retried with exponential backoff on rate limits. All images are sent inline, never uploaded as files; only JSON text is cached (`cache/`, gitignored).

**Per-video vocabulary** (`meta.vocab_source` = `"gemini"`): 24 frames spread over the video (preferring frames with a hand), in 3 batches of 8, are shown to Gemini, which lists the objects it clearly sees (name, up to 3 synonyms, portable/fixed, whether it is a powered cutting/rotating tool, frame indexes) and an `unsure` list. A name is kept only if at least 2 of the 3 batches name it; the detector is then run with its prompts on the frames that cite it (`confirmed` or `vlm_only`); a hazard must be powered cutting/rotating and cited in at least 2 frames; at most 40 objects. The final vocabulary is these objects plus the base file's fixed non-hazard surfaces, generic catch-alls and ignore list. `vlm_only` labels get a review item `vocab_unverified` and unsure objects a review item `vlm_unsure_object`.

**Naming of held objects**

PS2 sends the clearest crop of every object that has a `hold` or `touch` contact (longest contacts first, within the call limit) to Gemini together with the vocabulary labels, and asks for one label from the list, `other:<short name>` or `unknown`.

- A label from the list becomes the object's label in `frames`, `contacts` and `tools` (type and hazard follow the vocabulary entry), the object's `label_status` is `identified`, and its contacts get the reason `label_from_vlm` if the label changed.
- `other:<name>` keeps the detector's label, sets `label_status` to `unidentified` and puts the name into `review_queue[].suggested_label`.
- `unknown`, no key, or any API failure: nothing changes (failures are logged, never raised).
- `identity_uncertain` review items are only created for objects that have a `tools[]` row and at least one contact.
- Answers are cached in `cache/` (gitignored) per clip and object. The contact `state` is decided before this step and is not changed by it.
