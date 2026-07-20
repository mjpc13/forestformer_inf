custom_imports = dict(imports=['oneformer3d'])

num_channels = 32
num_semantic_classes = 3

radius = 16
score_th = 0.4
chunk = 10_000

model = dict(
    type='ForAINetV2OneFormer3D_XAwarequery',
    in_channels=3,
    num_channels=num_channels,
    voxel_size=0.2,
    min_spatial_shape=128,
    query_point_num=300,
    radius=radius,
    score_th=score_th,
    chunk=chunk,

    backbone=dict(
        type='SpConvUNet',
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True,
    ),

    decoder=dict(
        type='ForAINetv2QueryDecoder_XAwarequery',
        num_layers=6,
        num_semantic_queries=num_semantic_classes,
        in_channels=32,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        attn_mask=True,
        fix_attention=True,
    ),

    test_cfg=dict(
        topk_insts=300,
        inst_score_thr=0.0,
        pan_score_thr=0.0,
        npoint_thr=10,
        obj_normalization=True,
        obj_normalization_thr=0.01,
        sp_score_thr=0.15,
        nms=True,
        matrix_nms_kernel='linear',
        num_sem_cls=num_semantic_classes,
        stuff_cls=[0],
        thing_cls=[0],
    ),
)
