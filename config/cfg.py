import numpy as np
cfg = {
        'batch_size': 8,
        'rand_scales': [1],
        "num_classes": 80,
        "max_tag_len": 128,
	"pull_weight": 0.1,
	"push_weight": 0.1,
	"regr_weight": 1,
        'mean' : np.array([0.40789654, 0.44719302, 0.47026115]),
        'std' : np.array([0.28863828, 0.27408164, 0.27809835]),
        'scales': np.arange(0.6,1.5,0.1),
        'input_size': (511,511),
        'output_size': (128,128),
        'border': 128,
        'rand_crop': True,
        'rand_color': True,
        'gaussian_bump': True,
        "gaussian_iou": 0.7,
        "gaussian_radius": -1,

        "test_scales": 1,
}
