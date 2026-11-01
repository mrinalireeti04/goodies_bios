"""Camera-motion compensation: how much did a point or a box really move, with head motion removed."""
import cv2
import numpy as np


class Motion:
    """Call step() once per frame, in order. Then residual() / aligned_diff() describe that frame pair."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.prev = None          # previous frame, small grayscale
        self.old = None           # the frame before `gray`, for aligned_diff
        self.gray = None
        self.s = 1.0              # small-frame pixels per full-frame pixel
        self.A = np.eye(2)        # camera motion previous -> current frame (full-frame pixels)
        self.t = np.zeros(2)
        self.reliable = True      # False when too few background points were found
        self._diff = None         # aligned frame difference, built on first use per frame

    def step(self, bgr: np.ndarray, boxes) -> None:
        """Estimate camera motion since the previous frame. `boxes` (x1,y1,x2,y2) are hands/objects to ignore."""
        h, w = bgr.shape[:2]
        self.s = self.cfg["motion_width"] / w
        gray = cv2.cvtColor(cv2.resize(bgr, (self.cfg["motion_width"], int(h * self.s))), cv2.COLOR_BGR2GRAY)
        self.A, self.t, self.reliable, self._diff = np.eye(2), np.zeros(2), True, None
        if self.prev is not None:
            M = self._fit(gray, boxes)
            if M is None:
                self.reliable = False
            else:  # similarity transform in small-frame pixels -> full-frame pixels
                self.A, self.t = M[:, :2], M[:, 2] / self.s
        self.old, self.gray, self.prev = self.prev, gray, gray

    def _fit(self, gray, boxes):
        mask = np.full(gray.shape, 255, np.uint8)
        for x1, y1, x2, y2 in boxes:
            mask[max(int(y1 * self.s), 0):int(y2 * self.s) + 1, max(int(x1 * self.s), 0):int(x2 * self.s) + 1] = 0
        pts = cv2.goodFeaturesToTrack(self.prev, maxCorners=self.cfg["motion_max_corners"], qualityLevel=0.01,
                                      minDistance=8, mask=mask)
        if pts is None or len(pts) < self.cfg["motion_min_points"]:
            return None
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, pts, None)
        ok = status.ravel() == 1
        if ok.sum() < self.cfg["motion_min_points"]:
            return None
        M, _ = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC)
        return M

    def residual(self, prev_point, point) -> np.ndarray:
        """Movement of a point from the previous to this frame, minus camera motion (full-frame pixels)."""
        expected = self.A @ np.asarray(prev_point, float) + self.t
        return np.asarray(point, float) - expected

    def aligned_diff(self, box, exclude=()) -> float:
        """Mean grey-level change inside a box after aligning the previous frame to this one (0-255).

        Pixels inside the `exclude` boxes (hands) are ignored, so a moving hand does not look like a moving machine.
        """
        if self.old is None:
            return 0.0
        if self._diff is None:  # one warp per frame, shared by all boxes
            M = np.hstack([self.A, (self.t * self.s)[:, None]])
            warped = cv2.warpAffine(self.old, M, (self.gray.shape[1], self.gray.shape[0]), borderMode=cv2.BORDER_REPLICATE)
            self._diff = cv2.absdiff(warped, self.gray)
        x1, y1, x2, y2 = (int(v * self.s) for v in box)
        x1, y1 = max(x1, 0), max(y1, 0)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        region = self._diff[y1:y2, x1:x2]
        valid = np.ones(region.shape, bool)
        for ex1, ey1, ex2, ey2 in exclude:
            valid[max(int(ey1 * self.s) - y1, 0):max(int(ey2 * self.s) + 1 - y1, 0),
                  max(int(ex1 * self.s) - x1, 0):max(int(ex2 * self.s) + 1 - x1, 0)] = False
        return float(region[valid].mean()) if valid.any() else 0.0
