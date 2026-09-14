# UBC 512×512 RGB复制到12通道实验

本实验保持 OlmoEarth 编码器冻结，将每个 RGB 图块按 `B,G,R` 顺序循环四次，构造以下12通道输入：

```text
B,G,R,B,G,R,B,G,R,B,G,R
```

这些通道随后分别使用 OlmoEarth 的 Sentinel-2 L2A 波段统计量进行归一化。训练、验证和测试必须同时指定：

```text
--tile-size 512 --s2-rgb-mode repeat
```

## 单模态训练

```bash
nohup python scripts/tools/train_ubc_roof_instance.py train \
  --data-root /root/autodl-tmp/UBC_v2.0 \
  --weights /root/autodl-tmp/olmoearth_pretrain/weights/OlmoEarth-v1_2-Base \
  --task single \
  --init-ckpt runs/ubc_single_256_v2/best.pt \
  --out-dir runs/ubc_single_512_repeat \
  --epochs 24 \
  --tile-size 512 \
  --patch-size 4 \
  --s2-rgb-mode repeat \
  --samples-per-image 1 \
  --balanced-extra-samples 1 \
  --batch-size 1 \
  --accum-steps 4 \
  --workers 4 \
  --stride 256 \
  --tile-batch-size 1 \
  --eval-every 4 \
  --val-max-images 0 \
  --log-every 500 \
  --lr 0.0002 \
  > train_single_512_repeat.log 2>&1 &
```

UBC 原图为512×512，因此每张图每轮只需要一个基础样本；额外的一份类别平衡样本用于增加少数类别出现频率。有效批量大小为4。

## 单模态验证

```bash
python scripts/tools/train_ubc_roof_instance.py eval \
  --data-root /root/autodl-tmp/UBC_v2.0 \
  --weights /root/autodl-tmp/olmoearth_pretrain/weights/OlmoEarth-v1_2-Base \
  --task single \
  --ckpt runs/ubc_single_512_repeat/best.pt \
  --split val \
  --out-json runs/ubc_single_512_repeat/val_predictions.json \
  --tile-size 512 \
  --patch-size 4 \
  --s2-rgb-mode repeat \
  --stride 256 \
  --tile-batch-size 1
```

## 单模态测试

```bash
python scripts/tools/train_ubc_roof_instance.py eval \
  --data-root /root/autodl-tmp/UBC_v2.0 \
  --weights /root/autodl-tmp/olmoearth_pretrain/weights/OlmoEarth-v1_2-Base \
  --task single \
  --ckpt runs/ubc_single_512_repeat/best.pt \
  --split test \
  --out-json runs/ubc_single_512_repeat/test_predictions.json \
  --tile-size 512 \
  --patch-size 4 \
  --s2-rgb-mode repeat \
  --stride 256 \
  --tile-batch-size 1
```

若512图块在训练开始时显存溢出，不能继续降低 `batch-size`，因为已经为1。此时需要启用显存更小的适配方案，或返回256图块进行对照实验。

## 多模态训练

多模态模式将上述12通道复制结果送入 Sentinel-2 分支，单通道 SAR 仍单独送入 Sentinel-1 分支。建议先完成单模态512实验，再用其最佳检查点初始化兼容的检测和分割头：

```bash
nohup python scripts/tools/train_ubc_roof_instance.py train \
  --data-root /root/autodl-tmp/UBC_v2.0 \
  --weights /root/autodl-tmp/olmoearth_pretrain/weights/OlmoEarth-v1_2-Base \
  --task multimodal \
  --init-ckpt runs/ubc_single_512_repeat/best.pt \
  --out-dir runs/ubc_multimodal_512_repeat \
  --epochs 24 \
  --tile-size 512 \
  --patch-size 4 \
  --s2-rgb-mode repeat \
  --samples-per-image 1 \
  --balanced-extra-samples 1 \
  --batch-size 1 \
  --accum-steps 4 \
  --workers 4 \
  --stride 256 \
  --tile-batch-size 1 \
  --eval-every 4 \
  --val-max-images 0 \
  --log-every 500 \
  --lr 0.0002 \
  > train_multimodal_512_repeat.log 2>&1 &
```

如果不先训练单模态512版本，也可以将 `--init-ckpt` 指向现有的 `runs/ubc_single_256_v2/best.pt`。它只是兼容参数的热启动，不要求检查点窗口尺寸相同。

## 多模态验证

```bash
python scripts/tools/train_ubc_roof_instance.py eval \
  --data-root /root/autodl-tmp/UBC_v2.0 \
  --weights /root/autodl-tmp/olmoearth_pretrain/weights/OlmoEarth-v1_2-Base \
  --task multimodal \
  --ckpt runs/ubc_multimodal_512_repeat/best.pt \
  --split val \
  --out-json runs/ubc_multimodal_512_repeat/val_predictions.json \
  --tile-size 512 \
  --patch-size 4 \
  --s2-rgb-mode repeat \
  --stride 256 \
  --tile-batch-size 1
```

## 多模态测试

```bash
python scripts/tools/train_ubc_roof_instance.py eval \
  --data-root /root/autodl-tmp/UBC_v2.0 \
  --weights /root/autodl-tmp/olmoearth_pretrain/weights/OlmoEarth-v1_2-Base \
  --task multimodal \
  --ckpt runs/ubc_multimodal_512_repeat/best.pt \
  --split test \
  --out-json runs/ubc_multimodal_512_repeat/test_predictions.json \
  --tile-size 512 \
  --patch-size 4 \
  --s2-rgb-mode repeat \
  --stride 256 \
  --tile-batch-size 1
```
