# GF-3 HH 储油量训练与评估

入口：`scripts/tools/train_hh_tank_geom.py`。
本次参考本地 `油罐论文.pdf` 第 16 页图 5、第 17–18 页 SAR 方法和公式 (3)–(5)。
这是 OlmoEarth 的适配实现，不是论文 MTUNet 的完整复现，也尚未实测优于旧 checkpoint。

## 物理量与旧结果的区别

论文定义：

```text
d12 = ||O1-O2||，d13 = ||O1-O3||，r = 原始图像像素坐标中的屋顶半径
R = Sr*r/sin(delta)
H = Sr*d12/cos(delta)
h = Sr*(d13-2*r)/cos(delta)       # 浮顶下降高度，论文公式 (5)
oil_height = H-h                 # 根据下降高度定义推导储油高度
V = pi*R²*oil_height
```

旧脚本减的是 `r`，并将该结果直接作为储油高度。旧 CSV 的 `V_m3` 不能直接
充当新公式下的对照基准；应对两组几何预测使用同一公式后再比较。
`--formula legacy` 保留旧公式用于诊断；该模式的 `h_m` 仍表示旧版截断后的储油高度。
默认 `paper` 模式的 `h_m` 表示下降高度，且 `0 <= h <= H`、`R > 0`、`H > 0` 才有效。
不满足时保留 `oil_height_raw_m`、`V_raw_m3`，将 `V_m3` 留空（NaN），标记
`geometry_valid=False`、`geometry_status=invalid_geometry`。不能把留空视为空罐，
也不能在汇总库存时静默忽略这些罐。

Sr、入射角沿用 `meta.xlsx` 中的 `pixel_resolution`、`incidenceangle`。
论文假定距离向与方位向已经做过分辨率对齐；新数据必须核对这一前提。
仅对当前论文/标注定义实现反演，不自动将任意 GF-3 原始数据转换到该几何体系。

## 实现变化

- 用正方形填充和等比例缩放代替中心裁剪；不裁掉大罐。掩膜填充区域为忽略标签，
  不镜像复制屋顶。关键点和半径计算还原到原始像素坐标；导出的掩膜也恢复原图尺寸。
- 关键点使用归一化高斯目标的空间交叉熵，叠加坐标与 O1–O2/O1–O3 距离监督。
  缺失关键点不参与损失和关键点误差统计，仍使用该样本的分割标注。
- 新模型默认使用峰值周围 5×5 局部 soft-argmax；`--decode soft/argmax` 用于对比。
  不默认按横坐标交换 O1/O2/O3 的语义。
- 保留 OlmoEarth 冻结权重，移除阻断输入适配层梯度的 `no_grad()`。
  这会增加训练显存；显存不足先减小 batch size。推理仍禁用梯度。
- 增加原分辨率卷积细节分支和 CBAM，融合到双任务解码器。
  这是借鉴论文注意力和细节保留思想的工程实现；`--no-detail-branch` 可作消融。
- 默认半径由最大连通屋顶、孔洞填充和迭代剔除离群点的圆拟合获得。
  它是论文 Hough 检测器的确定性替代方案，不根据关键点距离强行缩小半径。
  `--radius-mode area/fit/auto` 保留对照；`gt` 仅用于定位误差来源，不能作为部署性能。
- 从 `train` 内部划出固定验证集，保存 `split.json`；不使用 `test` 选最佳权重。
  默认按样本随机划分，不能证明跨场景泛化。同一罐/同一场景应放在同一子集，
  可用 `--val-list validation_filenames.txt` 指定来自 train 的验证文件名，每行一个。
- 梯度裁剪、余弦学习率；checkpoint 保存结构、预处理、解码、损失和划分配置。
  最佳权重依据验证 mIoU、关键点、半径和 d13 误差联合选择。

## Linux 训练

沿用已安装环境；数据需要在 Linux 可读路径，不能直接用 Windows 盘符。
将下面两个路径改成服务器实际位置，模型目录应包含 OlmoEarth 配置和权重。

```bash
cd /root/autodl-tmp/olmoearth_pretrain
source .venv/bin/activate

TANK_DATA=/root/autodl-tmp/SAR_Oil_DataSet/Single_Tank_Oil_Estimation_train_split
TANK_WEIGHTS=/root/autodl-tmp/OlmoEarth-v1_2-Base

python scripts/tools/train_hh_tank_geom.py train \
  --data-root "$TANK_DATA" --weights "$TANK_WEIGHTS" \
  --out-dir runs/tank_geom_v2 \
  --size 128 --patch-size 4 --batch-size 2 --workers 4 \
  --epochs 80 --lr 0.0003 --seed 42 --val-fraction 0.15

python scripts/tools/train_hh_tank_geom.py eval \
  --data-root "$TANK_DATA" --weights "$TANK_WEIGHTS" \
  --ckpt runs/tank_geom_v2/best.pt --split test \
  --meta "$TANK_DATA/test/meta.xlsx" \
  --radius-mode robust --formula paper --decode local \
  --save-preds pred_test_v2
```

使用新的输出目录；脚本拒绝覆盖已有 checkpoint 或 `volumes.csv`。
本次新结构和损失需要重新训练；旧 checkpoint 会自动加载旧结构、旧预处理，
不能仅替换推理脚本就获得新训练分支的能力。旧 checkpoint 默认采用其原来的 soft 解码。
可在单独输出目录尝试 `--decode local`，但其质量必须验证。
旧 checkpoint 的评估 loss 使用新版损失，不能和旧训练日志的 loss 直接比较。

## 如何判断提升

`metrics.json` 包含 mIoU、原图像素单位的 O1/O2/O3 误差、半径 MAE、d13 MAE、
论文公式无效几何比例、GT 无效比例和不完整标注数量。
提供 meta 且导出预测时，还包含有效预测/GT 配对数、有效配对上的体积 MAE/RMSE。
这些体积 GT 来自人工几何标注和同一公式，不是现场独立测得的实际油量。

必须同时报告体积误差与有效覆盖率；仅在少量成功样本上误差低不能说明整体表现好。
同一份测试集、相同公式、相同半径模式下比较；不要在测试集上反复挑选超参数。
缺失 metadata 与几何失败单独记录，不填成 0。

已检查本地数据：train 913 张，其中 7 张缺少 O1 或 O3；test 234 张且关键点完整。
旧中心裁剪会影响 66 张测试图（至少一边大于 128）。测试人工标注全部满足论文几何有效条件。
训练集中另有 18 张完整标注不满足该几何条件，因此没有把强制正体积当作训练目标。

对 `E:/pred_test` 已保存的 234 张预测掩膜做只读对照：原 CSV 半径与标注圆半径的
平均绝对误差为 **14.2455 px**；先按旧中心裁剪/填充规则还原到原图，再使用新版
掩膜清理和稳健圆拟合，误差为 **1.2873 px**。比较使用相同样本及原图像素单位，
未重新训练、未改变关键点，也未使用真实半径指导拟合。该结果验证了半径后处理
确有改进，不代表完整储油量精度或新网络已经验证提升。旧填充区包含镜像屋顶，
将它也用于圆拟合是半径膨胀的重要来源。

## 本地验证

```bash
python -m unittest discover -s tests/unit -p test_hh_tank_geom.py -v
```

回归测试覆盖几何公式、无效值处理、缩放还原、掩膜清理、峰值解码、忽略填充、
适配层梯度、旧结构加载及一轮训练/验证/checkpoint 流程。
使用 CPU 小型替代编码器检查流程，不下载 OlmoEarth 权重。
完整 GPU 训练、真实 OlmoEarth 前向/反向和性能提升仍需在服务器验证。
