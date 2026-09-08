# 光学油罐检测与储量反演

当前光学流程分为两个独立阶段：

1. `optical_tank_pipeline.py`：冻结 OlmoEarth 骨干，只训练检测头，用于从大图检出并裁出单罐。
2. `optical_tank_volume.py`：直接处理已经组织好的单罐光学数据，不依赖 OlmoEarth 权重，也不需要训练。

## 储量数据目录

程序直接支持 `ADD_with_metadata` 的实际目录结构：

```text
DATA_ROOT/
  ADD_with_metadata/
    YYYY_MM_DD/
      no_shadow_split_tank/cut10_1.tif
      shadow_split_tank/cut10_1_L.tif
      more_info/*.xml
      more_info/extend_info.txt
  with_metadata/
    YYYY_MM_DD/...
```

`L/N/R` 都是阴影图；它们分别表示左重叠、无重叠和右重叠。程序会去掉该后缀，与同名无阴影图配对。若同一油罐存在多个后缀候选，只保留几何质量最高的结果。

`no_metadata` 默认不处理，因为缺少太阳/卫星角度，无法把像素阴影长度换算为高度和体积。

## 已移植的计算逻辑

- CLAHE、Canny 和 Hough 圆检测；
- 根据 `extend_info.txt` 将圆心从无阴影图映射到阴影图；
- 圆内/圆外阴影分离、固定阈值掩膜、Sobel 边界扫描；
- L/N/R 对应的同名圆弧点选择；
- 边界扫描失败时使用原程序的圆弧匹配回退；
- 读取 XML 的太阳高度角、太阳方位角、卫星高度角、卫星方位角和 GSD；
- 按论文公式计算罐高、浮顶下降深度、油高和圆柱体储量。

所有掩膜和边缘图都只在内存中计算，不生成旧程序的大量中间目录。最终只写一个 CSV。与原程序一致，物理换算使用 `ImageRowGSD`；CSV 同时保留 `ImageColumnGSD` 便于复核。

## Linux 储量计算命令

先用少量样本检查环境和路径：

```bash
uv run python -m scripts.tools.optical_tank_volume \
  --data-root /data/ADD_with_metadata \
  --out-csv runs/optical_smoke.csv \
  --limit 100
```

确认无误后处理所有带元数据的日期：

```bash
uv run python -m scripts.tools.optical_tank_volume \
  --data-root /data/ADD_with_metadata \
  --branches ADD_with_metadata with_metadata \
  --out-csv runs/optical_tank_volumes.csv
```

只处理一个分支：

```bash
uv run python -m scripts.tools.optical_tank_volume \
  --data-root /data/ADD_with_metadata \
  --branches ADD_with_metadata \
  --out-csv runs/optical_tank_volumes_add.csv
```

结果中应筛选 `geometry_valid=True`。失败项不会被静默写成 0，原因记录在 `geometry_status`。合法的零储量表示 `Lex_px == Lin_px`，即内外阴影推导出的油高确实为零。

## Linux 大图检测命令

训练检测头（OlmoEarth 骨干冻结）：

```bash
uv run python -m scripts.tools.optical_tank_pipeline train-det \
  --data-root /data/Object_Detection_train_split \
  --weights /data/OlmoEarth-v1_2-Base \
  --out-dir runs/optical_tank_det \
  --size 1024 --batch-size 2 --epochs 30
```

滑窗检测并保存无损 TIFF 裁片：

```bash
uv run python -m scripts.tools.optical_tank_pipeline detect \
  --weights /data/OlmoEarth-v1_2-Base \
  --ckpt runs/optical_tank_det/best.pt \
  --image-dir /data/optical_scenes \
  --out-dir runs/optical_detect \
  --crop-format tif
```
