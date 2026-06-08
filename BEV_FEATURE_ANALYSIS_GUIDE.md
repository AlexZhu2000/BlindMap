# BEV特征分布分析工具 - 使用指南

## 概述

这个工具用于验证**不同大小物体的融合BEV特征是否具有区分性**，以支持或反驳"按物体大小分类Proto"的假设。

## 关键设计决策

### ✅ 使用GT标签而非预测结果

**原因**：
- 排除v3模型的预测误差干扰
- 验证特征的"物理本质"是否具有大小相关的差异
- 这是一个基线实验，回答"理想的数据条件下，特征是否可区分"

**后续计划**：
- 第二阶段可以用v3的预测结果来验证v3的学习效果

### 物体大小的分类方式

```
按物体占BEV特征图的相对面积分类：

size_ratio = (bbox_pixel_x / map_width) * (bbox_pixel_y / map_height)

Class 0 (极小):  size_ratio < 0.01    (< 1%)      # 远处的车
Class 1 (小):    0.01 ≤ ratio < 0.04  (1-4%)     # 中距离小车
Class 2 (中):    0.04 ≤ ratio < 0.12  (4-12%)    # 标准中距离车
Class 3 (大):    0.12 ≤ ratio < 0.30  (12-30%)   # 大车或近的车
Class 4 (超大):  size_ratio ≥ 0.30    (> 30%)    # 非常近的车
```

## 安装依赖

```bash
# 基础依赖应该已经有，额外需要：
pip install umap-learn scikit-learn

# 确保有matplotlib和seaborn
pip install matplotlib seaborn
```

## 使用方法

### 基本用法

```bash
cd /home/zzh/projects/BlindMap

python opencood/tools/analyze_bev_feature_distribution.py \
  --model_dir /path/to/blindmap/model \
  --config_file /path/to/config.yaml \
  --output_dir ./bev_analysis_results
```

### 参数说明

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| --model_dir | str | ✅ | N/A | BlindMap预训练模型目录 (包含 .pth 文件) |
| --config_file | str | ✅ | N/A | 配置YAML文件路径 |
| --output_dir | str | ❌ | ./bev_feature_analysis | 输出目录 |
| --num_samples | int | ❌ | None | 处理的样本数 (None=全部) |
| --device | str | ❌ | cuda | 设备选择 (cuda 或 cpu) |

### 实际示例

```bash
# 快速测试 (20个样本)
python opencood/tools/analyze_bev_feature_distribution.py \
  --model_dir /home/zzh/projects/BlindMap/opencood/logs/BlindMap_opv2v_m1m2_2025_12_23_19_23_52_thre_0.01_use_history \
  --config_file /home/zzh/projects/BlindMap/opencood/logs/BlindMap_opv2v_m1m2_2025_12_23_19_23_52_thre_0.01_use_history/config.yaml \
  --output_dir ./analysis_quick_test \
  --num_samples 20

# 完整分析 (全部数据)
python opencood/tools/analyze_bev_feature_distribution.py \
  --model_dir /home/zzh/projects/BlindMap/opencood/logs/BlindMap_opv2v_m1m2_2025_12_23_19_23_52_thre_0.01_use_history \
  --config_file /home/zzh/projects/BlindMap/opencood/logs/BlindMap_opv2v_m1m2_2025_12_23_19_23_52_thre_0.01_use_history/config.yaml \
  --output_dir ./analysis_full
```

## 输出文件说明

运行完成后，在 `output_dir` 中会生成：

```
output_dir/
├─ analysis_report.json          # 完整的数值报告
├─ umap_visualization.png        # UMAP可视化 (主要用看这个)
├─ metrics_visualization.png     # 聚类指标柱状图
├─ features_raw.npy              # 原始特征矩阵 (N, C)
└─ labels.npy                    # 大小类别标签 (N,)
```

### 关键输出文件详解

#### 1. umap_visualization.png (最重要)

这个图显示了所有提取的特征在2D空间中的分布，用5种颜色表示5个大小类别：

**理想情况 (✅ 支持方案)**：
- 5个颜色的点形成5个分离的区域
- 同色的点紧密聚集
- 不同颜色的点之间有明显间隙

**问题情况 (❌ 需要调整)**：
- 颜色完全混杂，看不出分离
- 某些颜色的点很分散
- 多个颜色的点重叠

#### 2. analysis_report.json (参考)

示例输出：
```json
{
  "timestamp": "2026-03-26 14:30:00",
  "config": {
    "num_total_features": 1250,
    "bev_size": [205, 205]
  },
  "feature_distribution": {
    "极小": 180,
    "小": 320,
    "中": 450,
    "大": 230,
    "超大": 70
  },
  "metrics": {
    "silhouette_score_global": 0.3456,
    "davies_bouldin_index": 1.8234,
    "calinski_harabasz_index": 125.6,
    "silhouette_score_per_class": {
      "0": 0.32,
      "1": 0.28,
      "2": 0.35,
      "3": 0.38,
      "4": 0.42
    }
  },
  "conclusions": [
    "✅ 聚类效果优秀: 不同大小类别特征分离明显",
    "✅ Proto大小分类方案可行"
  ]
}
```

## 解读指标

### Silhouette Score (轮廓系数)

**范围**: [-1, 1]  
**含义**: 1 表示完美聚类，-1 表示完全错误分类

| 值 | 解释 | 结论 |
|-----|------|------|
| > 0.5 | 强聚类 | ✅ 类别明确分离 |
| 0.3-0.5 | 中等聚类 | ⚠️ 有一定分离但重叠 |
| 0.1-0.3 | 弱聚类 | ❌ 类别混淆 |
| < 0.1 | 无聚类结构 | ❌ 工作方案不可行 |

**脚本输出**:
- `silhouette_score_global`: 所有特征的平均得分
- `silhouette_score_per_class`: 每个大小类别的得分

### Davies-Bouldin Index

**范围**: [0, ∞)，**越低越好**

| 值 | 解释 | 结论 |
|-----|------|------|
| < 1.0 | 优秀 | ✅ 类别分离度高 |
| 1.0-1.5 | 良好 | ✅ 类别可区分 |
| 1.5-2.0 | 一般 | ⚠️ 类别有重叠 |
| > 2.0 | 差 | ❌ 类别混淆 |

### 决策矩阵

| Silhouette | Davies-Bouldin | 解释 | 行动 |
|------------|------------------|------|--------|
| > 0.4 | < 1.5 | ✅ 完美 | 继续Proto方案 |
| 0.2-0.4 | 1.5-2.0 | ⚠️ 一般 | 调整阈值后重试 |
| < 0.2 | > 2.0 | ❌ 差 | 重新设计方案 |

## 常见问题

### Q1: 运行时出现 "RuntimeError: CUDA out of memory"

**解决**:
```bash
# 减少处理样本数
python analyze_bev_feature_distribution.py ... --num_samples 50

# 或使用CPU (较慢但不吃显存)
python analyze_bev_feature_distribution.py ... --device cpu
```

### Q2: 找不到融合特征 "⚠ 样本0: 未找到融合特征"

**原因**: BlindMap的输出中可能没有直接包含融合特征

**解决方案**: 
- 选项A: 修改 `analyze_bev_feature_distribution.py` 中的特征提取逻辑
- 选项B: Hook模型的中间层获取特征（高级用法）

**临时方案** (快速修复):
```python
# 在 BEVFeatureAnalyzer.process_batch() 中修改：
# 可以从output_dict中可用的特征检索
# 如 cls_preds 可以反推特征维度
# 但这不是最优方案，建议咨询模型架构
```

### Q3: "ValueError: x和y的样本数不匹配"

**原因**: 数据加载或特征提取出现不一致

**解决**: 
- 检查数据集是否完整
- 查看控制台的警告信息
- 确保GT标签格式正确

### Q4: UMAP图看不清楚，点太多重叠了

**调整**:
```python
# 在 visualize_with_umap 中修改参数
reducer = umap.UMAP(
    n_neighbors=30,      # 从15改大，保留更多全局结构
    min_dist=0.3,        # 从0.1改大，点之间距离更远
    metric='euclidean'   # 试试欧氏距离
)
```

然后重新运行。

## 实验工作流程

### 第1步：快速验证 (5分钟)

```bash
# 先用20个样本检查流程
python analyze_bev_feature_distribution.py \
  --model_dir /path/to/model \
  --config_file /path/to/config.yaml \
  --output_dir ./test_output \
  --num_samples 20
```

检查清单：
- ✓ 是否有错误？
- ✓ 特征数是否 > 0？
- ✓ 图片是否生成？

### 第2步：小规模评估 (15分钟)

```bash
# 用100个样本做初步评估
python analyze_bev_feature_distribution.py \
  --model_dir /path/to/model \
  --config_file /path/to/config.yaml \
  --output_dir ./small_scale_output \
  --num_samples 100
```

查看关键指标：
- Silhouette Score 是多少？
- 大小类别分布是否均衡？

### 第3步：完整分析 (30-60分钟)

```bash
# 用全部数据做最终评估
python analyze_bev_feature_distribution.py \
  --model_dir /path/to/model \
  --config_file /path/to/config.yaml \
  --output_dir ./full_analysis_output
```

生成最终报告和决策。

## 输出结果的三种可能情况

### Scenario A: ✅ 支持Proto大小分类方案

```
指标:
  - Silhouette: 0.42
  - Davies-Bouldin: 1.23
  
图像:
  - UMAP中5个颜色明显分离
  - 同色点聚集紧密
  
结论: ✅ 可以按大小分类Proto
      继续实现方案第二阶段
```

### Scenario B: ⚠️ 部分支持，需调整

```
指标:
  - Silhouette: 0.28
  - Davies-Bouldin: 1.89
  
图像:
  - 大类别(Class 3,4)分离
  - 小类别(Class 0,1)混淆
  
结论: ⚠️ 需要调整方案:
      - 重新定义大小阈值
      - 或合并某些类别(如0+1为"小")
      - 或添加其他维度(距离)
```

### Scenario C: ❌ 不支持大小分类

```
指标:
  - Silhouette: 0.05
  - Davies-Bouldin: 3.12
  
图像:
  - 5个颜色完全混杂
  - 无明显聚类结构
  
结论: ❌ 大小分类不可行
      需要重新考虑:
      - 其他分类维度?
      - 改为样本级检索?
      - 或放弃Proto方案?
```

## 后续步骤

### 如果得到✅结果：

1. 记录成功参数（阈值等）
2. 开始实现Proto提取脚本 (第二阶段)
3. 集成到v3训练流程

### 如果得到⚠️结果：

1. 尝试调整大小阈值
2. 再次运行分析
3. 或添加其他分类维度（如距离）
4. 重复直到满足要求

### 如果得到❌结果：

1. 反思"按大小分类Proto"的假设
2. 考虑替代方案：
   - 样本级特征检索 (Bank of features)
   - 多维度分类 (大小+距离+角度)
   - 放弃Proto，回到原始监督
3. 与指导者讨论方向

---

**文档完成日期**: 2026-03-26  
**状态**: Ready for experiment
