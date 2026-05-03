from __future__ import annotations

import numpy as np

ORIGINAL_LABELS = (0, 200, 500, 600, 1220, 2221)
LABEL_TO_INDEX = {label: index for index, label in enumerate(ORIGINAL_LABELS)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}

CLASS_NAMES = {
    0: "background",
    1: "normal_myo",
    2: "lv_pool",
    3: "rv_pool",
    4: "edema",
    5: "scar",
}

CLASS_COLORS = {
    0: (0, 0, 0),
    1: (0, 200, 0),
    2: (20, 120, 255),
    3: (255, 190, 0),
    4: (255, 80, 80),
    5: (190, 0, 220),
}

DEFAULT_CLASS_FREQUENCIES = np.array(
    [
        0.9591866519703998,
        0.010631788904068712,
        0.01104565891823038,
        0.013195560585252475,
        0.0026617789615664576,
        0.003278560660482217,
    ],
    dtype=np.float32,
)

DEFAULT_MODALITIES = ("C0", "T2", "LGE")
MYOCARDIUM_CLASS_INDICES = (1,)
EDEMA_CLASS_INDEX = 4
SCAR_CLASS_INDEX = 5
PATHOLOGY_CLASS_INDICES = (4, 5)
DEFAULT_CROP_SIZE = (320, 320)
DEFAULT_VIS_MODALITY = "LGE"
