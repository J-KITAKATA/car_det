_base_ = '../../../configs/rtmdet/rtmdet-ins_s_8xb32-300e_coco.py'
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth'  # noqa

data_root = 'data/CarDet/data/'

# Windows環境において複数GPU使えるようにするもの（必要ないならコメントアウト）
dist_params = dict(backend='gloo', port=29500)

metainfo = dict(
    classes=('Kei_car', 'Sedan', 'Coupe', 'SUV', 'Minivan', 'Small_Truck', 
             'Large_Truck', 'Bus', 'Scooter', 'Compact_car', 'Station_Wagon', 
             'Commercial_Van', 'Hatchback', 'Motorcycle'),
    palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
             (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
             (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30)]
)

# 計算経路の最適化 (必要ないのであればコメントアウト)
env_cfg = dict(
    cudnn_benchmark=True, # 画像サイズ固定（800x800）なら絶対ONにすべき設定
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='gloo'),
)

model = dict(
    bbox_head=dict(
        num_classes=14,
        loss_cls=dict(
            _delete_=True, # 上書き先の引数全てを削除して、書いたやつに置き換える
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=2.0 # <=== loss_clsの大きさ
        )
        ),
    # RTMDet-Insはbbox_headでクラス数を制御する
    test_cfg=dict( # <======================== Out Of Memory の原因は大体ここ
            nms_pre=1000,  # デフォルト 1000
            min_bbox_size=0,
            max_per_img=100, # デフォルト 100
            nms=dict(
                type='soft_nms',
                iou_threshold=0.3,
                sigma=0.5,
                min_score=0.05,
                method='linear' # linear or gaussian
                ),
            score_thr=0.5, # スコアが低いものは除外
            class_agnostic=True,
            agnostic_nms=True,
            multi_label=False,
            mask_thr_binary=0.5
        )
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='CachedMosaic', img_scale=(800, 800), pad_val=114.0, max_cached_images=10), # cached 10 change
    dict(
        type='RandomResize',
        scale=(1600, 1600),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=(800, 800),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(800, 800), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=(800, 800),
        ratio_range=(1.0, 1.0),
        max_cached_images=5, # cached 20 -> 5
        pad_val=(114, 114, 114)),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='RandomResize',
        scale=(800, 800),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=(800, 800),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(800, 800), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

# --- test_pipeline の修正 ---
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    # 理由1: 解像度を 1333 -> 800 に落として、マスク計算用のテンソルを小さくする
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    # 理由2: RTMDetは正方形入力を想定した設計のため、Padでサイズを固定するとVRAMの断片化が防げる
    dict(type='Pad', size=(800, 800), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataloader = dict(
    batch_size=16, # メモリに応じて調整
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/coco-carDataset.json',
        data_prefix=dict(img='images/train_images/'),
        pipeline=train_pipeline
    )
)

# 最適化アルゴリズムの指定
optim_wrapper = dict(
    type='AmpOptimWrapper',
    accumulative_counts=1, # 勾配蓄積ステップ数（必要に応じて調整）
    optimizer=dict(
        type='AdamW',      # ここで AdamW を明示
        lr=0.0005,          # 学習率（Configに合わせて調整）
        weight_decay=0.05  # AdamWで重要な重み減衰パラメータ
    ),
    # 勾配クリッピング（爆発防止）を入れるのがRTMDetの推奨
    clip_grad=dict(max_norm=35, norm_type=2)
)


# val_dataloader setting
val_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        pipeline=test_pipeline,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/coco-carDataset_Val.json',
        data_prefix=dict(img='images/val_images/')
    )
)

# test_dataloader setting (validationと同じデータを使う場合はval_dataloaderをコピーしてもOK)
test_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        pipeline=test_pipeline,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/coco-carDataset_TEST.json',
        data_prefix=dict(img='images/test_images/')
    )
)

# val setting
val_evaluator = dict(ann_file=data_root + 'annotations/coco-carDataset_Val.json')

# test setting
test_evaluator = dict(ann_file=data_root + 'annotations/coco-carDataset_TEST.json')

# train schedule
max_epochs = 310 # <========= MAX EPOCH
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=1 # 何epoch 毎にValitationを行うか
    )

# 学習率のスケジュール設定
param_scheduler = [
    # 1. ウォームアップ（最初の1000 iterationだけ徐々に学習率を上げる）
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=1000),
    # 2. Cosine Annealing（最後まで滑らかに学習率を下げる）
    dict(
        type='CosineAnnealingLR',
        eta_min=0.00005, # 最終的に base_lr の 1/10 まで下げる
        begin=0,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True)
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,             # 10エポックごとに保存
        save_best='coco/bbox_mAP', # 精度(mAP)が最高のものを保存
        rule='greater',          # 数値が大きい方を「良」とする
        max_keep_ckpts=15,        # 最新のpthファイルを6個だけ残す（古いのは自動削除）
        published_keys=['meta', 'state_dict']
    ),
    # ログ表示の頻度など（念のため）
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    timer=dict(type='IterTimerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002, update_buffers=True, priority=49),
    dict(type='PipelineSwitchHook', switch_epoch=(max_epochs - 30), switch_pipeline=train_pipeline_stage2),
    dict(type='EmptyCacheHook', before_epoch=True, after_epoch=True)
]

work_dir = './data/CarDet/work_dirs/rtm_ins_s_800x800'