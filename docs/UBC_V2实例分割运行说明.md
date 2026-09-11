# UBC v2 屋顶实例分割运行说明

本文档对应脚本：

```text
scripts/tools/train_ubc_roof_instance.py
```

该脚本使用冻结的 OlmoEarth-v1_2-Base 骨干和可训练的 Mask R-CNN，完成 UBC v2 的两项屋顶实例分割任务：

- `single`：RGB 单模态实例分割。
- `multimodal`：RGB 和 SAR 多模态实例分割。

模型输出每栋建筑独立的掩膜、边界框、12 类屋顶类别和置信度。验证和测试使用 COCO `segm` 指标，原任务重点关注 `AP50`。

## 1. Linux 目录约定

以下命令假设项目、数据和权重位于：

```text
/root/autodl-tmp/olmoearth_pretrain
/root/autodl-tmp/UBC_v2.0
/root/autodl-tmp/olmoearth_pretrain/weights/OlmoEarth-v1_2-Base
```

数据目录应保持下面的结构，不需要提前把 COCO 标签转换成掩膜图：

```text
UBC_v2.0/
├── fine-grained_building_roof_instance_segmentation/
│   ├── annotations/
│   │   ├── roof_fine_12_train.json
│   │   └── roof_fine_12_val.json
│   ├── train/
│   └── val/
├── multi-modal_fine-grained_building_roof_instance_segmentation/
│   ├── annotations/
│   │   ├── roof_fine_train.json
│   │   └── roof_fine_val.json
│   ├── train/
│   │   ├── rgb/
│   │   └── sar/
│   └── val/
│       ├── rgb/
│       └── sar/
└── test_set/
    ├── single_modal_test/
    └── multi_modal_test/
```

如果服务器上的实际路径不同，只需修改命令中的 `--data-root` 和 `--weights`。

## 2. 激活环境

```bash
cd /root/autodl-tmp/olmoearth_pretrain
source .venv/bin/activate
```

检查关键依赖和 CUDA：

```bash
python -c "import torch, torchvision, pycocotools, PIL; print('torch=', torch.__version__, 'torchvision=', torchvision.__version__, 'cuda=', torch.cuda.is_available())"
```

预期 `cuda=True`。如果提示缺少 `pycocotools`，在已经激活的环境中安装：

```bash
python -m pip install pycocotools
```

## 3. 单模态冒烟测试

正式训练前先用 8 张训练影像和 2 张验证影像跑一个 epoch：

```bash
python scripts/tools/train_ubc_roof_instance.py train \
  --data-root /root/autodl-tmp/UBC_v2.0 \
  --weights /root/autodl-tmp/olmoearth_pretrain/weights/OlmoEarth-v1_2-Base \
  --task single \
  --out-dir runs/ubc_single_smoke \
  --epochs 1 \
  --max-train-images 8 \
  --val-max-images 2 \
  --samples-per-image 1 \
  --batch-size 1 \
  --accum-steps 1 \
  --tile-batch-size 1 \
  --workers 0
```

冒烟测试成功后会生成：

```text
runs/ubc_single_smoke/last.pt
runs/ubc_single_smoke/best.pt
runs/ubc_single_smoke/history.json
runs/ubc_single_smoke/val_epoch_001.json
```

## 4. 单模态完整训练

```bash
python scripts/tools/train_ubc_roof_instance.py train \
  --data-root /root/autodl-tmp/UBC_v2.0 \
  --weights /root/autodl-tmp/olmoearth_pretrain/weights/OlmoEarth-v1_2-Base \
  --task single \
  --out-dir runs/ubc_single \
  --epochs 24 \
  --batch-size 4 \
  --accum-steps 2 \
  --workers 4 \
  --samples-per-image 2 \
  --tile-size 128 \
  --patch-size 4 \
  --stride 96 \
  --tile-batch-size 4 \
  --eval-every 4 \
  --val-max-images 0 \
  --lr 0.0002
```

`--val-max-images 0` 表示每次验证都使用完整的 2507 张单模态验证影像。`--eval-every 4` 表示每 4 个 epoch 完整验证一次，并根据完整验证集的 `AP50` 保存 `best.pt`。

如果要求每个 epoch 都完整验证，改为：

```text
--eval-every 1 --val-max-images 0
```

完整验证耗时较长。单模态验证集共有 2507 张影像，每次完整验证约处理 62675 个滑窗。

## 5. 单模态完整验证

```bash
python scripts/tools/train_ubc_roof_instance.py eval \
  --data-root /root/autodl-tmp/UBC_v2.0 \
  --weights /root/autodl-tmp/olmoearth_pretrain/weights/OlmoEarth-v1_2-Base \
  --task single \
  --ckpt runs/ubc_single/best.pt \
  --split val \
  --out-json runs/ubc_single/val_predictions.json \
  --tile-size 128 \
  --patch-size 4 \
  --stride 96 \
  --tile-batch-size 4
```

输出文件：

```text
runs/ubc_single/val_predictions.json
runs/ubc_single/val_predictions.metrics.json
```

## 6. 单模态 test 本地评测

UBC v2 的本地 test JSON 包含真值，可以直接计算 COCO 指标：

```bash
python scripts/tools/train_ubc_roof_instance.py eval \
  --data-root /root/autodl-tmp/UBC_v2.0 \
  --weights /root/autodl-tmp/olmoearth_pretrain/weights/OlmoEarth-v1_2-Base \
  --task single \
  --ckpt runs/ubc_single/best.pt \
  --split test \
  --out-json runs/ubc_single/test_predictions.json \
  --tile-size 128 \
  --patch-size 4 \
  --stride 96 \
  --tile-batch-size 4
```

主要指标包括：

```text
AP
AP50
AP75
AP_small
AP_medium
AP_large
AP50_<各屋顶类别>
```

原始任务重点查看 `AP50`。

## 7. 多模态冒烟测试

```bash
python scripts/tools/train_ubc_roof_instance.py train \
  --data-root /root/autodl-tmp/UBC_v2.0 \
  --weights /root/autodl-tmp/olmoearth_pretrain/weights/OlmoEarth-v1_2-Base \
  --task multimodal \
  --out-dir runs/ubc_multimodal_smoke \
  --epochs 1 \
  --max-train-images 8 \
  --val-max-images 2 \
  --samples-per-image 1 \
  --batch-size 1 \
  --accum-steps 1 \
  --tile-batch-size 1 \
  --workers 0
```

## 8. 多模态完整训练

建议使用单模态最佳检查点初始化兼容的 RPN、分类、边界框和掩膜头：

```bash
python scripts/tools/train_ubc_roof_instance.py train \
  --data-root /root/autodl-tmp/UBC_v2.0 \
  --weights /root/autodl-tmp/olmoearth_pretrain/weights/OlmoEarth-v1_2-Base \
  --task multimodal \
  --init-ckpt runs/ubc_single/best.pt \
  --out-dir runs/ubc_multimodal \
  --epochs 24 \
  --batch-size 2 \
  --accum-steps 4 \
  --workers 4 \
  --samples-per-image 2 \
  --tile-size 128 \
  --patch-size 4 \
  --stride 96 \
  --tile-batch-size 2 \
  --eval-every 4 \
  --val-max-images 0 \
  --lr 0.0002
```

`--val-max-images 0` 表示每次使用完整的 1681 张多模态验证影像，约处理 42025 个 RGB/SAR 对齐滑窗。

## 9. 多模态完整验证

```bash
python scripts/tools/train_ubc_roof_instance.py eval \
  --data-root /root/autodl-tmp/UBC_v2.0 \
  --weights /root/autodl-tmp/olmoearth_pretrain/weights/OlmoEarth-v1_2-Base \
  --task multimodal \
  --ckpt runs/ubc_multimodal/best.pt \
  --split val \
  --out-json runs/ubc_multimodal/val_predictions.json \
  --tile-size 128 \
  --patch-size 4 \
  --stride 96 \
  --tile-batch-size 2
```

## 10. 多模态 test 本地评测

```bash
python scripts/tools/train_ubc_roof_instance.py eval \
  --data-root /root/autodl-tmp/UBC_v2.0 \
  --weights /root/autodl-tmp/olmoearth_pretrain/weights/OlmoEarth-v1_2-Base \
  --task multimodal \
  --ckpt runs/ubc_multimodal/best.pt \
  --split test \
  --out-json runs/ubc_multimodal/test_predictions.json \
  --tile-size 128 \
  --patch-size 4 \
  --stride 96 \
  --tile-batch-size 2
```

## 11. 断点续训

单模态断点续训：

```bash
python scripts/tools/train_ubc_roof_instance.py train \
  --data-root /root/autodl-tmp/UBC_v2.0 \
  --weights /root/autodl-tmp/olmoearth_pretrain/weights/OlmoEarth-v1_2-Base \
  --task single \
  --out-dir runs/ubc_single \
  --resume runs/ubc_single/last.pt \
  --epochs 24 \
  --batch-size 4 \
  --accum-steps 2 \
  --workers 4 \
  --samples-per-image 2 \
  --tile-size 128 \
  --patch-size 4 \
  --stride 96 \
  --tile-batch-size 4 \
  --eval-every 4 \
  --val-max-images 0 \
  --lr 0.0002
```

续训时，`--task`、`--tile-size`和`--patch-size`必须与检查点保持一致。

## 12. 只生成预测 JSON

如果只需要结果文件而不计算指标，可以把 `eval` 改成 `predict`：

```bash
python scripts/tools/train_ubc_roof_instance.py predict \
  --data-root /root/autodl-tmp/UBC_v2.0 \
  --weights /root/autodl-tmp/olmoearth_pretrain/weights/OlmoEarth-v1_2-Base \
  --task single \
  --ckpt runs/ubc_single/best.pt \
  --split test \
  --out-json runs/ubc_single/test_predictions_only.json \
  --tile-size 128 \
  --patch-size 4 \
  --stride 96 \
  --tile-batch-size 4
```

## 13. 显存不足时的调整

训练显存不足时，首先降低训练批量，并增加梯度累积：

```text
--batch-size 1 --accum-steps 8
```

验证或测试显存不足时，降低滑窗推理批量：

```text
--tile-batch-size 1
```

不要为了降低显存把整幅 `512×512` 影像缩放到 `128×128`，否则小建筑和屋顶边界会明显损失。

如果 GPU 不支持 bfloat16，可以关闭自动混合精度：

```text
--no-amp
```

## 14. 滑窗和结果融合

默认设置为：

```text
窗口：128×128
步长：96
重叠：32 像素
```

程序会自动：

1. 将一张 `512×512` 影像划分为 25 个滑窗。
2. 分别运行 OlmoEarth 和 Mask R-CNN。
3. 把局部边界框和掩膜恢复到原图坐标。
4. 使用同类别 Mask-NMS 去除重复实例。
5. 使用跨类别 Mask-NMS 去除同一建筑的类别冲突结果。
6. 使用包含率校验清理被窗口边缘截断的重复掩膜。
7. 输出完整的 COCO RLE 实例结果。

相关默认阈值为：

```text
--score-threshold 0.05
--mask-threshold 0.5
--same-class-nms 0.5
--cross-class-nms 0.7
--containment-nms 0.85
--max-instances 500
```

这些阈值应使用 `val` 调整，确定后再对 `test` 做最终评测，不建议反复根据 test 指标调参。

## 15. 输出文件说明

训练目录主要包含：

```text
best.pt                 完整验证集 AP50 最佳的可训练参数
last.pt                 最后一个 epoch 的可训练参数
history.json            每个 epoch 的损失和验证指标
val_epoch_XXX.json      对应 epoch 的 COCO 验证预测
```

检查点不包含冻结的 OlmoEarth 骨干权重。因此训练、验证和测试时都必须传入相同的 `--weights` 路径。

`eval` 生成：

```text
<out-json>              COCO 实例预测结果
<out-json>.metrics.json COCO AP、AP50和各类别AP50
```

最终预测 JSON 中，每栋建筑对应一条记录：

```json
{
  "image_id": 123,
  "category_id": 6,
  "segmentation": {
    "size": [512, 512],
    "counts": "COCO_RLE"
  },
  "score": 0.91,
  "bbox": [100.0, 80.0, 35.0, 42.0]
}
```

其中 `image_id` 和 `category_id` 均保持 UBC 原始 COCO 标签中的编号。
