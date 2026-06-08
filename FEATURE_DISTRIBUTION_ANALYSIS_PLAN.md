# BEV特征分布分析实验设计

## 一、问题陈述

**核心问题**：不同大小物体所覆盖区域的BEV融合特征是否有明显分布差异？

这个问题决定了Proto方案的可行性：
- ✅ 如果有差异 → 按大小分类Proto有效
- ❌ 如果没有差异 → Proto方案需要重新设计

---

## 二、实验设计的关键决策

### 2.1 **使用GT标签 vs 预测结果？**

#### 两种方案的对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **使用GT标签** | 排除预测误差的干扰，验证特征本身的性质 | 不能验证v3的学习效果 | ✅ **选择这个** |
| **使用预测结果** | 接近真实推理场景 | 混淆"特征差异"和"预测误差"两个问题 | 后续验证 |

#### 为什么选GT标签？

```
实验目标：验证"物理性质"而非"模型性能"

问题分解：
  问题A: 不同大小物体的特征是否天然不同？(物理层面)
  问题B: v3模型是否能学到这种区分？(学习层面)

当前状态：
  - BlindMap已验证有效（融合特征质量有保证）
  - v3还未训练（学习能力未知）

因此：
  现阶段应该回答问题A，即用GT标签
  后续再用预测结果回答问题B
```

### 2.2 **特征分组的定义**

```
基于GT标签的分组方式：

对于每个GT框：
  ├─ bbox_size_x, bbox_size_y (世界坐标系)
  ├─ 转换到BEV特征图坐标系
  └─ 计算 size_ratio = (size_x / map_width) * (size_y / map_height)
  
分组阈值（基于大小百分比）：
  Class 0 (极小): size_ratio < 0.01    (< 1%)
  Class 1 (小):   0.01 ≤ size_ratio < 0.04   (1-4%)
  Class 2 (中):   0.04 ≤ size_ratio < 0.12   (4-12%)
  Class 3 (大):   0.12 ≤ size_ratio < 0.30   (12-30%)
  Class 4 (超大): size_ratio ≥ 0.30   (> 30%)
```

### 2.3 **特征提取的范围**

```
问题：对于某个物体，应该提取哪些特征？

选项A：仅提取物体中心点的特征
  └─ 特征维度: (1, C) 或仅仅 C
  └─ 问题: 丢失物体覆盖区域的信息
  └─ 不利于观察"大车小车的差异"
  
选项B：提取物体bbox覆盖的整个区域的特征
  └─ 特征维度: (num_pos_pixels, C)
  └─ 优点: 保留空间分布信息
  └─ 问题: 特征数量不同，难以比较
  
选项C：聚合和规范化
  └─ 提取整个区域，然后聚合: Mean/Max/PCA
  └─ 特征维度: (C,) 或保留部分维度
  └─ ✅ 推荐

选择：选项C
  理由：
    - 保留物体覆盖区域的统计特性
    - 不同大小物体可比较
    - 便于可视化
```

---

## 三、技术实现方案

### 3.1 特征提取流程

```python
对于每个验证样本：
  
1. 加载数据和GT标签
   └─ object_bbx_center: GT框中心 (max_objs, 3)
   └─ object_bbx_mask: 有效标记 (max_objs,)
   └─ 融合特征: fused_feature (1, C, H, W)

2. 前向推理获取pos_equal_one
   └─ BlindMap(data) → output_dict
   └─ 提取 pos_equal_one (1, H, W, anchor_num)

3. 对每个有效的GT框 i:
   
   3.1 确定大小类别
       ├─ 计算 bbox 在BEV图上的像素大小
       ├─ size_ratio = (pixel_x / W) * (pixel_y / H)
       └─ size_class = discretize(size_ratio)
   
   3.2 获取物体对应的pos位置
       ├─ 对GT框做透视变换到BEV坐标
       ├─ 得到物体在特征图上的范围
       └─ mask_i = BEV_bbox(物体i的范围)
   
   3.3 使用pos_equal_one提取特征
       ├─ pos_mask_i = pos_equal_one * mask_i
       ├─ valid_pos = pos_mask_i > 0 (WHERE pos标记为1)
       └─ 获取valid_pos对应的融合特征像素
   
   3.4 聚合特征
       ├─ features_raw = fused_feature[valid_pos] 
       │                 shape: (num_pos_pixels, C)
       ├─ 多种聚合方式：
       │  ├─ 方式1: Mean pooling
       │  │  └─ feature_agg = features_raw.mean(dim=0)  
       │  │     shape: (C,)
       │  └─ 方式2: PCA降维（可选，保留前K个分量）
       │     └─ feature_agg = pca.transform(features_raw)
       │        shape: (num_samples, K)
       │
       └─ 结果: feature_i_agg ∈ R^C

4. 收集所有特征及其元信息
   └─ features_dict = {
        'feature': feature_i_agg,
        'size_class': size_class,
        'bbox_size': (pixel_x, pixel_y),
        'num_pos_pixels': num_pos_pixels,
        'gt_box_id': i
      }

5. 按照size_class分组
   └─ grouped_features = {
        0: [feature_agg, ...],
        1: [feature_agg, ...],
        2: [feature_agg, ...],
        3: [feature_agg, ...],
        4: [feature_agg, ...]
      }
```

### 3.2 可视化方案

```python
降维方法对比（选择最佳的）：

1. t-SNE
   └─ 非线性降维，保留局部结构
   └─ 适合看簇的分离程度
   └─ 缺点: 耗时，结果依赖随机种子

2. UMAP (推荐)
   └─ 非线性降维，保留全局+局部结构
   └─ 比t-SNE快很多
   └─ 结果更稳定
   └─ 推荐使用

3. PCA
   └─ 线性降维，计算快
   └─ 保留主要方差方向
   └─ 适合作为基线对比

可视化输出：
  ├─ 图1: 所有特征按size_class着色 (5种颜色)
  │  └─ 观察: 5个类别是否形成分离的区域？
  │
  ├─ 图2: 每个size_class的特征单独可视化
  │  └─ 观察: 同类特征内是否聚集？
  │
  └─ 图3: 统计指标
     ├─ 每类的Silhouette Score
     ├─ Davies-Bouldin Index (类间距/类内距)
     ├─ 类内方差
     └─ 类间距离
```

---

## 四、定量评估指标

### 4.1 聚类质量指标

```python
对每个size_class，计算：

1. Silhouette Score (单个样本级别)
   └─ 范围: [-1, 1]
   └─ s(i) = (b_i - a_i) / max(a_i, b_i)
   └─ 其中 a_i = 样本i到同类其他样本的平均距离
   └─        b_i = 样本i到最近异类的平均距离
   └─ 解释: > 0.5 表示该样本被正确聚类

2. Davies-Bouldin Index (整体水平)
   └─ 范围: [0, ∞)，越低越好
   └─ DB = (1/k) * Σ max_j(R_ij)
   └─ R_ij = (σ_i + σ_j) / d(c_i, c_j)
   └─ 其中 σ_i = 类i的平均半径
   └─        d(c_i, c_j) = 类i和j中心的距离
   └─ 解释: < 1.5 表示聚类较好

3. 类内方差
   └─ for each class k:
   │   variance_k = mean ||f - mean(class_k)|| ^ 2
   └─ 低方差表示该类内特征相似

4. 类间距离
   └─ for each pair (k1, k2):
   │   distance = ||center_k1 - center_k2||
   └─ 高距离表示类的分离程度好
```

### 4.2 解读标准

```
理想情况（支持Proto方案）：
  ✓ Silhouette Score (每类平均) > 0.4
  ✓ Davies-Bouldin Index < 1.5
  ✓ 类间距离 > 2 * 类内方差
  ✓ 5个类别在UMAP平面上形成5个不同的区域

问题情况（需要调整方案）：
  ✗ Silhouette Score < 0.2
    → 特征没有明显的类别结构
  ✗ Davies-Bouldin Index > 2.0
    → 类别间重叠太多
  ✗ 类间距离 < 1 * 类内方差
    → 类别间距离太近，不可区分
```

---

## 五、实现的代码文件

应该创建：`/home/zzh/projects/BlindMap/opencood/tools/analyze_bev_feature_distribution.py`

**功能模块**：
```
1. 数据加载模块
   └─ 加载BlindMap模型、config、数据集

2. 特征提取模块
   ├─ 前向推理获得fused_feature和pos_equal_one
   ├─ 根据GT计算size_class
   ├─ 提取pos区域的融合特征
   └─ 聚合特征

3. 分析模块
   ├─ 按size_class分组
   ├─ 计算聚类指标
   └─ 生成统计报告

4. 可视化模块
   ├─ UMAP/t-SNE降维
   ├─ 生成颜色编码的2D散点图
   ├─ 生成每类的直方图/分布图
   └─ 保存可视化结果

5. 报告生成模块
   └─ 生成summary JSON和可视化总结
```

---

## 六、实验流程

```
Step 1: 准备
  □ 加载BlindMap模型 (epoch37)
  □ 加载OPV2V验证集
  □ 准备输出目录

Step 2: 特征提取
  □ 遍历验证集样本
  □ 对每个样本：
    ├─ 前向推理 → fused_feature, pos_equal_one
    ├─ 识别有效GT框
    ├─ 按GT提取pos区域特征
    ├─ 计算size_class
    ├─ 聚合特征到(C,)维度
    └─ 保存结果

Step 3: 分析
  □ 按size_class分组所有特征
  □ 计算聚类指标 (Silhouette, DB, etc.)
  □ 生成统计报告

Step 4: 可视化
  □ UMAP降维到2D
  □ 绘制：
    ├─ 全局散点图 (5个颜色)
    ├─ 每类单独散点图
    └─ 统计柱状图

Step 5: 结论
  □ 是否支持"按大小分类Proto"的假设？
  □ 如果是: 继续实现Proto方案
  □ 如果否: 需要重新设计
```

---

## 七、预期结果的几种情况

### Case 1: 理想情况（支持方案）
```
可视化结果：
  - UMAP平面上5个类别形成明显分离的簇
  - 同类样本紧密聚集
  - 不同类别间有明显间隙
  
指标：
  - Silhouette Score (avg) > 0.4
  - Davies-Bouldin < 1.5
  
结论：✅ Proto方案可行
```

### Case 2: 部分分离（需要调整）
```
可视化结果：
  - 某些类别清晰分离，某些类别混淆
  - 例：大车(Class 3,4)分离，小车(Class 0,1)混淆
  
指标：
  - Silhouette Score (avg) 0.2-0.4
  - Davies-Bouldin 1.5-2.5
  
结论：⚠️ 需要调整：
  - 重新定义大小阈值
  - 或者考虑其他分类维度（如距离）
```

### Case 3: 完全混淆（不支持方案）
```
可视化结果：
  - UMAP平面上5个类别完全混在一起
  - 没有明显的聚类结构
  - 颜色完全混杂
  
指标：
  - Silhouette Score (avg) < 0.2
  - Davies-Bouldin > 2.5
  
结论：❌ Proto大小分类方案不行
  需要重新思考：
  - 是否需要多维度分类？
  - 是否特征本身就没有区分性？
  - 是否应该改为"样本级检索"而非"聚类Proto"？
```

---

## 八、关键参数和实现细节

### 8.1 大小阈值的确定

```
目前的阈值是初步的经验值：
  Class 0: < 0.01
  Class 1: 0.01-0.04
  Class 2: 0.04-0.12
  Class 3: 0.12-0.30
  Class 4: > 0.30

可以通过直方图调整：
  1. 首先计算所有GT框的size_ratio分布
  2. 绘制直方图看是否自然聚集
  3. 按聚集的"谷值"来划分阈值
  （而不是均匀划分）
```

### 8.2 特征聚合方式

```
推荐使用Mean pooling，因为：
  - 简单，易于理解
  - 计算快
  - 能保留物体覆盖区域的统计特性
  
可选方案：
  1. Max pooling
     └─ 更关注"最强信号"
     
  2. 加权平均 (按IOU加权)
     └─ 更关注"高质量"的pos位置
     
  3. PCA聚合
     └─ 保留主要方差
```

### 8.3 降维参数

```
UMAP配置：
  n_neighbors = 15      # 局部邻域大小
  min_dist = 0.1        # 最小分散距离
  metric = 'cosine'     # 使用余弦距离
  
t-SNE配置 (如果使用)：
  perplexity = 30
  n_iter = 1000
  metric = 'cosine'
```

---

## 九、输出文件结构

```
output_dir/
├─ metadata.json                      # 运行配置和参数
├─ statistics.json                    # 聚类指标
├─ size_distribution.json             # 大小类别的样本统计
├─ features_raw.npy                   # 原始特征 (N, C)
├─ features_metadata.json             # 每个特征的元信息
├─ features_umap_2d.npy               # UMAP投影 (N, 2)
│
├─ visualizations/
│  ├─ umap_global.png                 # 全局UMAP (5个颜色)
│  ├─ umap_class_0.png                # 第0类放大图
│  ├─ umap_class_1.png
│  ├─ umap_class_2.png
│  ├─ umap_class_3.png
│  ├─ umap_class_4.png
│  ├─ silhouette_scores.png           # Silhouette分数分布
│  ├─ size_distribution.png           # 大小分布直方图
│  └─ class_statistics.png            # 聚类质量指标柱状图
│
└─ analysis_report.md                 # 综合分析报告
```

---

## 总结

**使用GT标签的理由**：
- 排除v3模型的预测误差干扰
- 验证特征的"物理本质"是否具有大小相关的差异
- 为Proto方案的可行性提供基线支持

**关键假设**：
- BlindMap的融合特征质量有保证 (已验证的模型)
- 不同大小物体的BEV特征应该不同 (物理直觉)
- 按大小分类可以捕捉这种差异 (待验证)

**实验的意义**：
- 如果✅支持方案 → 可以继续实现Proto分类方案
- 如果⚠️需调整 → 需要重新定义分类维度或调整阈值
- 如果❌不支持 → 需要考虑完全不同的架构设计
