from detectron2.config import CfgNode as CN


def add_grit_config(cfg):
    _C = cfg

    _C.MODEL.BEAM_SIZE = 1
    _C.MODEL.TRAIN_TASK = ["ObjectDet", "DenseCap"]
    _C.MODEL.TEST_TASK = "DenseCap"  # This can be varied if the model is jointly trained on multiple tasks

    _C.MODEL.ROI_BOX_HEAD.USE_BIAS = 0.0  # >= 0: not use
    _C.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE = False

    _C.MODEL.ROI_HEADS.MASK_WEIGHT = 1.0
    _C.MODEL.ROI_HEADS.OBJECT_FEAT_POOLER_RES = 14
    _C.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False

    # Backbones
    _C.MODEL.VIT_LAYERS = 12

    # Text Decoder
    _C.TEXT_DECODER = CN()
    _C.TEXT_DECODER.VOCAB_SIZE = 30522
    _C.TEXT_DECODER.HIDDEN_SIZE = 768
    _C.TEXT_DECODER.NUM_LAYERS = 6
    _C.TEXT_DECODER.ATTENTION_HEADS = 12
    _C.TEXT_DECODER.FEEDFORWARD_SIZE = 768 * 4

    # Multi-dataset dataloader
    _C.DATALOADER.DATASET_RATIO = [1, 1]  # sample ratio
    _C.DATALOADER.DATASET_BS = 1
    _C.DATALOADER.DATASET_INPUT_SIZE = [1024, 1024]
    _C.DATALOADER.DATASET_INPUT_SCALE = [(0.1, 2.0), (0.1, 2.0)]
    _C.DATALOADER.DATASET_MIN_SIZES = [(640, 800), (640, 800)]
    _C.DATALOADER.DATASET_MAX_SIZES = [1333, 1333]

    _C.SOLVER.USE_CUSTOM_SOLVER = True
    _C.SOLVER.OPTIMIZER = "ADAMW"
    _C.SOLVER.VIT_LAYER_DECAY = True
    _C.SOLVER.VIT_LAYER_DECAY_RATE = 0.7

    _C.INPUT.CUSTOM_AUG = "EfficientDetResizeCrop"
    _C.INPUT.TRAIN_SIZE = 1024
    _C.INPUT.TEST_SIZE = 1024
    _C.INPUT.SCALE_RANGE = (0.1, 2.0)
    # 'default' for fixed short / long edge
    _C.INPUT.TEST_INPUT_TYPE = "default"

    _C.FIND_UNUSED_PARAM = True
    _C.USE_ACT_CHECKPOINT = True
