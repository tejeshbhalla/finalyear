import cv2
import numpy as np
import os
import sys
import tensorflow as tf

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, "models")
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")

model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)

VIDEO_PATH = os.path.join(ROOT_DIR, "videos", "basketball.mp4")

OUTPUT_VIDEO_PATH = os.path.join(ROOT_DIR, "output_videos", "basketball_segmented.mp4")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (int(cap.get(3)), int(cap.get(4))), isColor=True)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.detect([frame], verbose=0)

    r = results[0]
    masks = r['masks']
    player_masks = np.zeros_like(masks)
    for i in range(masks.shape[-1]):
        if r['class_ids'][i] == 1:
            player_masks[:, :, i] = masks[:, :, i]

    segmented_frame = np.zeros_like(frame)
    for i in range(player_masks.shape[-1]):
        segmented_frame[:, :, 0] = np.where(player_masks[:, :, i], frame[:, :, 0], segmented_frame[:, :, 0])
        segmented_frame[:, :, 1] = np.where(player_masks[:, :, i], frame[:, :, 1], segmented_frame[:, :, 1])
        segmented_frame[:, :, 2] = np.where(player_masks[:, :, i], frame[:, :, 2], segmented_frame[:, :, 2])

    out.write(segmented_frame)

cap.release()
out.release()
cv2.destroyAllWindows()