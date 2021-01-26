_base_ = '../faster_rcnn/faster_rcnn_r101_fpn_1x_coco.py'
model = dict(
    neck=dict(
        type='AwareFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
)
