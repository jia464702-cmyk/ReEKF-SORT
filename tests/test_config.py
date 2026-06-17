from boxmot.trackers import (
    OccluBoost,
    BoostTrack,
    BotSort,
    ByteTrack,
    DeepOcSort,
    OcSort,
    ReEkfSort,
    StrongSort,
    HybridSort,
    SFSORT,
)

MOTION_N_APPEARANCE_TRACKING_NAMES = [
    "botsort",
    "deepocsort",
    "strongsort",
    "boosttrack",
    "occluboost",
    "hybridsort",
]
MOTION_ONLY_TRACKING_NAMES = ["ocsort", "reekfsort", "bytetrack", "sfsort"]

MOTION_N_APPEARANCE_TRACKING_METHODS = [StrongSort, BotSort, DeepOcSort, BoostTrack, OccluBoost, HybridSort]
MOTION_ONLY_TRACKING_METHODS = [OcSort, ReEkfSort, ByteTrack, SFSORT]

ALL_TRACKERS = [
    "botsort",
    "deepocsort",
    "ocsort",
    "reekfsort",
    "bytetrack",
    "sfsort",
    "strongsort",
    "boosttrack",
    "occluboost",
    "hybridsort",
]
PER_CLASS_TRACKERS = [
    "botsort",
    "deepocsort",
    "ocsort",
    "reekfsort",
    "bytetrack",
    "sfsort",
    "boosttrack",
    "occluboost",
    "hybridsort",
]
