custom_imports = dict(imports=['oneformer3d'])

num_channels = 32
num_instance_classes = 3
num_semantic_classes = 3

radius = 16
score_th = 0.4
chunk = 10_000

model = dict(
    type='ForAINetV2OneFormer3D_XAwarequery',
    #data_preprocessor=dict(type='Det3DDataPreprocessor'),
    in_channels=3,
    num_channels=num_channels,
    voxel_size=0.2,
    num_classes=num_instance_classes,
    min_spatial_shape=128,
    stuff_classes=[0],
    thing_cls=[1, 2],
    prepare_epoch=1000,
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
        num_classes=1,
        num_instance_queries=0,
        num_semantic_queries=num_semantic_classes,
        num_instance_classes=num_instance_classes,
        in_channels=32,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        iter_pred=True,
        attn_mask=True,
        fix_attention=True,
        objectness_flag=True,
    ),
    # because __init__ builds self.criterion.
    #criterion=dict(
    #    type='ForAINetv2UnifiedCriterion_XAwarequery',
    #    num_semantic_classes=num_semantic_classes,
    #    sem_criterion=dict(
    #        type='S3DISSemanticCriterion',
    #        loss_weight=0.2,
    #    ),
    #    inst_criterion=dict(
    #        type='InstanceCriterionForAI_OneToManyMatch',
    #        matcher=dict(type='One2ManyMatcher'),
    #        loss_weight=[1.0, 1.0, 0.5],
    #        fix_dice_loss_weight=True,
    #        iter_matcher=True,
    #        fix_mean_loss=True,
    #    ),
    #),

    train_cfg=dict(),

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