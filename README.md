# SME Guard —— 小微商户智能客服 RAG 系统

> 基于 **RAG（Retrieval-Augmented Generation）+ Multi-Query + RRF + Reranker + 本地大模型** 构建的小微商户智能客服系统。
> 面向健身、美容、餐饮、工作室等小微商户，将商户自己的业务知识库转化为可查询、可追溯的智能客服。

---
## before
*启动方式：
1.clone到本地
2.运行gradio_app.py,然后在浏览器打开对应http：//localhost端口
3.上传资料库，用MarkDown文件格式，可以让豆包生成一份模拟体验

## 1. 项目简介

SME Guard 是一个面向小微商户场景的本地化智能客服系统。

传统客服系统通常需要人工维护大量固定问答，面对用户表达方式变化时容易出现：

* 用户换一种问法就无法匹配
* FAQ 数量越来越多，维护成本高
* 大模型容易产生知识库之外的内容
* 相似问题可能召回错误文档
* 套餐价格、有效期、使用规则等业务信息容易混淆
* 知识库规模增加后，检索准确率和响应速度难以兼顾

因此，本项目采用 **RAG + 多路检索 + 重排序 + 严格回答约束** 的方式，将：

```text
商户业务知识
      ↓
Markdown / TXT / PDF
      ↓
知识库构建
      ↓
语义向量检索
      ↓
Multi-Query
      ↓
RRF 融合
      ↓
Dedup 去重
      ↓
Reranker 重排序
      ↓
Top-K 上下文
      ↓
本地大模型
      ↓
带知识来源的客服回答
```

最终实现一个可以由商户自行维护知识库的智能客服系统。

---

# 2. 核心特点

### 2.1 多商户知识库隔离

系统按照 `merchant_id` 为不同商户创建独立的向量集合：

```text
merchant_demo_merchant_001
merchant_xxx
merchant_yyy
```

不同商户之间的知识不会混淆。

---

### 2.2 支持商户自主上传知识

目前支持：

* Markdown
* TXT
* PDF

例如：

```text
营业时间.md
套餐介绍.md
预约流程.md
会员规则.md
退款规则.md
注意事项.md
```

上传之后自动进行解析、切分、Embedding，并写入向量数据库。

---

### 2.3 Multi-Query 查询扩展

用户提出的问题可能非常口语化。

例如：

```text
周一几点开门？
```

系统会生成语义相近的查询：

```text
周一几点开门？
周一开门时间是多少？
```

然后分别进行检索，提高召回能力。

---

### 2.4 RRF 多路结果融合

不同 Query 得到的检索结果并不完全相同。

系统使用 **Reciprocal Rank Fusion（RRF）** 对多个 Query 的结果进行融合。

核心思想：

```text
Query 1 → 检索结果
Query 2 → 检索结果
Query 3 → 检索结果
       ↓
      RRF
       ↓
统一候选结果
```

这样可以降低单个查询表达方式对最终结果的影响。

---

### 2.5 Reranker 精排

向量检索主要负责：

> **尽可能把相关内容召回。**

但向量相似度并不一定能够准确判断：

> 哪一段内容最适合回答当前问题。

因此系统采用：

```text
Embedding Recall
        ↓
Top 20
        ↓
Reranker
        ↓
Top 5
```

当前使用：

```text
BAAI/bge-reranker-v2-m3
```

用于对候选文档进行二次语义相关性判断。

---

### 2.6 严格知识库约束

系统不会让大模型自由发挥。

核心回答原则：

1. 只使用知识库提供的信息
2. 不允许编造知识库之外的事实
3. 知识库无法支持回答时，明确拒答
4. 知识库存在冲突时，不自行判断哪个正确
5. 回答关键事实时附带文档来源和检索分数

例如：

```text
我们的常规营业时间为 09:00-21:00。

[doc: 营业时间.md | score: 0.7046]
```

如果知识库中没有相关内容：

```text
暂时无法回答这个问题，请联系商户工作人员确认。
```

---

# 3. 系统整体架构

```text
                         ┌────────────────────┐
                         │      用户问题       │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │   Query Expansion  │
                         │    Multi-Query     │
                         └─────────┬──────────┘
                                   │
                     ┌─────────────┴─────────────┐
                     ▼                           ▼
              Query 1 检索                  Query 2 检索
                     │                           │
                     └─────────────┬─────────────┘
                                   ▼
                         ┌────────────────────┐
                         │        RRF         │
                         │    多路结果融合     │
                         └─────────┬──────────┘
                                   ▼
                         ┌────────────────────┐
                         │       Dedup        │
                         │       去重         │
                         └─────────┬──────────┘
                                   ▼
                         ┌────────────────────┐
                         │      Reranker      │
                         │ BGE Reranker M3    │
                         └─────────┬──────────┘
                                   ▼
                            Top-K Relevant
                                   │
                                   ▼
                         ┌────────────────────┐
                         │      Context       │
                         │   上下文构建        │
                         └─────────┬──────────┘
                                   ▼
                         ┌────────────────────┐
                         │    Local LLM       │
                         │      Ollama        │
                         └─────────┬──────────┘
                                   ▼
                         ┌────────────────────┐
                         │    客服最终回答     │
                         └────────────────────┘
```

---

# 4. 技术栈

| 模块        | 技术                      |
| --------- | ----------------------- |
| 开发语言      | Python                  |
| Web UI    | Gradio                  |
| LLM       | Ollama 本地大模型            |
| Embedding | Qwen3-Embedding         |
| Reranker  | BAAI/bge-reranker-v2-m3 |
| Vector DB | ChromaDB                |
| 检索方式      | Cosine Similarity       |
| Query 扩展  | Multi-Query             |
| 多路融合      | RRF                     |
| 精排        | Reranker                |
| 文档格式      | Markdown / TXT / PDF    |
| 运行方式      | 本地部署                    |

---

# 5. 项目结构

```text
intelligent customer service/
│
├── sme_guard/
│   │
│   ├── llm_client.py
│   │
│   ├── rag/
│   │   ├── vector_db.py
│   │   ├── ingest.py
│   │   ├── retrieve.py
│   │   └── mqe.py
│   │
│   └── ui/
│       └── gradio_app.py
│
├── scripts/
│   └── interactive_chat.py
│
├── chroma_db/
│
├── requirements.txt
│
└── README.md
```

---

# 6. 环境要求

推荐环境：

```text
Python >= 3.10
Ollama
CUDA GPU（推荐，但不是必须）
```

如果使用 GPU 运行 Reranker，建议：

```text
NVIDIA GPU
CUDA
```

---

# 7. 安装

## 7.1 创建 Python 环境

例如使用 Conda：

```bash
conda create -n agenttest python=3.10
conda activate agenttest
```

---

## 7.2 安装依赖

```bash
pip install -r requirements.txt
```

---

# 8. 安装 Ollama 模型

本项目使用 Ollama 运行本地模型。

启动 Ollama 后，根据项目配置拉取对应模型。

例如：

```bash
ollama pull qwen3-embedding:4b
```

Embedding 模型用于：

```text
文档 → 向量
用户问题 → 向量
```

然后通过向量相似度完成初步召回。

项目中的 LLM 模型可以根据本机硬件进行替换。

---

# 9. 启动项目

## 9.1 命令行版本

运行：

```bash
python scripts/interactive_chat.py
```

启动之后：

```text
======================================================================
小微商户智能客服 RAG
输入 exit / quit 退出
======================================================================

用户：
```

例如：

```text
用户：周一几点开门？
```

系统会：

```text
用户问题
 ↓
Multi-Query
 ↓
向量召回
 ↓
RRF
 ↓
Dedup
 ↓
Reranker
 ↓
Top-K
 ↓
LLM
 ↓
最终回答
```

---

# 10. Gradio Web 界面

启动：

```bash
python sme_guard/ui/gradio_app.py
```

默认访问：

```text
http://127.0.0.1:7860
```

Web UI 主要包含以下功能。

---

## 10.1 商户管理

可以创建商户：

```text
商户 ID
商户名称
行业
```

例如：

```text
merchant_id:
demo_merchant_001

商户名称：
悦动生活综合工作室

行业：
健身
```

---

## 10.2 知识库上传

可以上传：

```text
.md
.markdown
.txt
.pdf
```

系统会自动：

```text
文件
 ↓
文本解析
 ↓
语义分块
 ↓
Embedding
 ↓
ChromaDB
```

---

## 10.3 智能客服

用户可以直接输入问题：

```text
有哪些私教套餐？
```

系统自动从商户自己的知识库中检索相关内容并生成回答。

---

## 10.4 未命中问题

系统会记录知识库无法有效回答的问题。

例如：

```text
用户问题
↓
检索
↓
相关度不足
↓
记录 Unknown Question
```

后续商户可以根据这些问题补充知识库。

这形成：

```text
用户提问
   ↓
未命中
   ↓
问题记录
   ↓
补充知识库
   ↓
重新 Embedding
   ↓
客服能力提升
```

---

# 11. 知识库设计

RAG 系统的效果很大程度上取决于知识库质量。

因此本项目没有简单地把所有文字按照固定长度切分。

例如不推荐：

```text
每 500 字强制切一块
```

因为商业信息具有明显的结构：

```text
套餐名称
价格
课时
有效期
适用人群
使用规则
退款规则
```

如果简单按照字符长度切分，很容易出现：

```text
套餐价格
        ↓
被切到 Chunk A

有效期
        ↓
被切到 Chunk B

使用限制
        ↓
被切到 Chunk C
```

最终模型虽然找到了“套餐”，却没有同时拿到完整的价格和有效期。

---

# 12. Markdown 语义分块

因此项目采用基于 Markdown 标题层级的语义分块方式。

例如：

```markdown
# 健身私教套餐

## A体验私教课

- 课时：1节，60分钟
- 价格：199元
- 有效期：30天
- 新客户限购1次

## B基础套餐

- 课时：12节
- 价格：1680元
- 有效期：6个月
```

系统优先按照业务语义进行组织：

```text
健身私教套餐
 ├── A体验私教课
 │     ├── 课时
 │     ├── 价格
 │     ├── 有效期
 │     └── 使用规则
 │
 └── B基础套餐
       ├── 课时
       ├── 价格
       ├── 有效期
       └── 使用规则
```

这样能够尽量保证一个 Chunk 内的信息具有完整业务语义。

---

# 13. 项目开发过程中遇到的问题

这个项目并不是一开始就采用最终架构，而是经过多轮测试和优化。

整个演进过程可以概括为：

```text
基础 RAG
   ↓
发现回答不稳定
   ↓
分析是检索问题还是生成问题
   ↓
优化知识库结构
   ↓
优化 Chunk
   ↓
优化 Embedding
   ↓
提高召回 Top-K
   ↓
Multi-Query
   ↓
RRF
   ↓
Reranker
   ↓
Context 精简
   ↓
Prompt 约束
   ↓
性能分析
```

---

# 14. 问题一：固定长度切分导致业务信息被拆散

## 现象

例如用户问：

```text
A套餐多少钱？有效期多久？
```

虽然知识库中存在相关内容，但是检索结果可能只包含：

```text
A体验私教课
价格：199元
```

却没有：

```text
有效期：30天
```

---

## 原因

最初采用简单的文本切分策略。

固定长度切分无法理解：

```text
套餐
价格
课时
有效期
限制
```

这些业务字段之间的语义关系。

---

## 解决方案

改为：

> **基于 Markdown 标题和业务语义进行 Chunk 切分。**

让一个 Chunk 尽可能保持完整业务语义。

---

# 15. 问题二：单次向量检索召回不稳定

## 现象

用户输入：

```text
周一几点开门？
```

而知识库写的是：

```text
常规营业时间
周一至周日 09:00-21:00
```

虽然语义相近，但不同表达方式可能导致向量检索结果出现波动。

---

## 原因

单个 Query 的表达方式会影响 Embedding。

例如：

```text
周一几点开门？

周一营业时间是多少？

星期一什么时候营业？

星期一几点开始营业？
```

这些问题实际上表达的是同一个意图。

但单次向量检索只使用其中一种表达。

---

## 解决方案：Multi-Query

系统让一个问题生成多个语义表达：

```text
原始 Query
      ↓
Query Expansion
      ↓
Query 1
Query 2
Query 3
...
```

分别检索后再进行融合。

---

# 16. 问题三：Multi-Query 会产生重复结果

多个 Query 检索之后：

```text
Query 1 → A B C D
Query 2 → A B C E
Query 3 → A B C F
```

会产生大量重复文档。

因此增加：

```text
Dedup
```

根据文档 ID、标题和内容进行去重。

最终得到：

```text
A B C D E F
```

---

# 17. 问题四：向量相似度不等于真正相关

这是 RAG 中比较关键的问题。

例如：

```text
用户：
A套餐多少钱？

召回：

A套餐介绍       similarity = 0.71
B套餐介绍       similarity = 0.69
退款规则        similarity = 0.66
预约流程        similarity = 0.64
营业时间        similarity = 0.61
```

这些内容在向量空间中可能都比较接近。

但真正回答：

```text
A套餐多少钱？
```

最重要的是：

```text
A套餐介绍
```

---

# 18. 解决方案：Recall + Precision 两阶段检索

因此没有让向量搜索直接决定最终结果。

而是采用：

```text
第一阶段：

Embedding
    ↓
Top 20
```

目标：

> 尽可能保证答案不要漏掉。

然后：

```text
第二阶段：

Reranker
    ↓
Top 5
```

目标：

> 从召回结果中找出真正最相关的内容。

也就是：

```text
Vector Search
     ↓
负责 Recall
     ↓
Reranker
     ↓
负责 Precision
```

---

# 19. 问题五：为什么最终选择 BGE Reranker

项目早期曾考虑：

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

但测试发现该模型主要面向英文 MS MARCO 检索任务。

对于中文商业客服场景并不是理想选择。

因此最终使用：

```text
BAAI/bge-reranker-v2-m3
```

进行中文、多语言语义重排序。

在测试中可以看到类似：

```text
cosine_score: 0.7046
rerank_score: 0.9268
```

这里将：

```text
cosine similarity
```

和：

```text
rerank score
```

分开记录。

Cosine 用于初步召回，Reranker 用于最终排序。

---

# 20. 问题六：检索结果很多，但上下文越多不一定越好

如果把 Top 20 全部交给大模型：

```text
Top 20
 ↓
全部 Context
 ↓
LLM
```

可能产生：

* Context 过长
* 无关信息增加
* 模型注意力被分散
* 推理速度下降
* 错误信息进入上下文

因此最终采用：

```text
Top 20 Recall
      ↓
Reranker
      ↓
Top 5
      ↓
LLM
```

只把最相关的内容交给大模型。

---

# 21. 问题七：大模型可能产生知识库之外的信息

即使检索结果正确，大模型仍然可能根据自身预训练知识进行补充。

对于商业客服场景，这存在明显风险。

例如知识库没有说明：

```text
是否支持退款
```

模型却可能根据常识回答：

```text
一般情况下可以退款……
```

这并不符合系统要求。

---

# 22. 解决方案：严格 Prompt 约束

系统要求模型：

```text
只能依据知识库回答。

不能使用知识库之外的信息。

如果知识库无法支持答案：
返回“暂时无法回答”。

如果知识库存在冲突：
明确指出冲突，不自行判断。
```

同时要求关键事实附带：

```text
[doc: 文档标题 | score: xxx]
```

这样可以提高回答的：

* 可追溯性
* 可解释性
* 可审计性

---

# 23. 问题八：无法回答的问题怎么办？

RAG 系统不能只追求：

> “什么问题都回答。”

对于客服系统，更重要的是：

> “不知道的时候不要乱回答。”

因此设置了最低相关度阈值。

当检索结果无法达到要求时：

```text
检索相关度不足
      ↓
拒绝生成
      ↓
暂时无法回答
```

这样可以降低幻觉。

---

# 24. 最终 RAG 策略

目前系统整体流程：

```text
                  用户问题
                     │
                     ▼
              Multi-Query
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
       Query 1               Query 2
          │                     │
          ▼                     ▼
      Vector Search         Vector Search
          │                     │
          └──────────┬──────────┘
                     ▼
                    RRF
                     │
                     ▼
                  Dedup
                     │
                     ▼
              Candidate Top-K
                     │
                     ▼
                 Reranker
                     │
                     ▼
                  Top 5
                     │
                     ▼
                 Context
                     │
                     ▼
                Local LLM
                     │
             ┌───────┴────────┐
             ▼                ▼
         有可靠依据        无可靠依据
             │                │
             ▼                ▼
          正常回答        暂时无法回答
```

---

# 25. 检索参数

目前主要参数：

```python
RETRIEVAL_TOP_K = 20
FINAL_TOP_K = 5
EXPANSION_COUNT = 1
MIN_SCORE = 0.5
RRF_K = 60
```

整体思想：

```text
Top-K Recall
    ↓
扩大候选集

RRF
    ↓
融合多路结果

Reranker
    ↓
提高精度

Top-5
    ↓
控制 Context

LLM
    ↓
生成最终回答
```

这些参数并不是固定不变，而是通过测试集逐步调整。

---

# 26. 测试示例

例如知识库包含：

```text
# 悦动生活综合工作室

## A体验私教课

- 课时：1节，60分钟
- 价格：199元
- 有效期：30天
- 新客户限购1次
```

用户：

```text
A套餐多少钱？
```

系统：

```text
用户问题
↓
Embedding
↓
Multi-Query
↓
Vector Recall
↓
RRF
↓
Reranker
↓
A体验私教课
↓
LLM
```

最终回答：

```text
A体验私教课价格为199元，包含1节60分钟私教课，
有效期为30天，新客户限购1次。

[doc: 悦动生活综合工作室 — 套餐介绍 | score: xxx]
```

---

# 27. 性能分析

在开发过程中发现：

> RAG 系统的耗时并不只有 LLM。

因此增加了不同阶段的监控。

整体可以拆成：

```text
Embedding
    ↓
Vector Retrieval
    ↓
MQE
    ↓
RRF
    ↓
Reranker
    ↓
Context
    ↓
LLM Generation
```

分别记录耗时后，可以定位：

```text
到底是检索慢？
还是 Reranker 慢？
还是大模型生成慢？
```

而不是简单地认为：

> “系统卡顿就是大模型的问题。”

---

# 28. 目前取得的效果

通过多轮测试和架构调整，系统从最初的：

```text
简单向量检索
+
直接交给 LLM
```

逐步演进到：

```text
语义 Chunk
+
Embedding Recall
+
Multi-Query
+
RRF
+
Dedup
+
Reranker
+
Top-K Context
+
严格 Prompt
+
Fallback
+
性能监控
```

主要改善方向包括：

### 检索稳定性

通过 Multi-Query 和 RRF，降低单一 Query 表达方式对召回结果的影响。

### 检索准确性

通过：

```text
Top 20 Recall
→ Reranker
→ Top 5
```

减少无关文档进入最终上下文。

### 知识完整性

通过 Markdown 语义分块，降低价格、有效期、课时、使用规则被拆散的问题。

### 幻觉控制

通过严格 Prompt 和最低相关度判断，让系统在知识库无法支持时拒绝回答。

### 可解释性

最终回答可以追溯到具体文档和检索结果。

### 可维护性

商户可以直接通过 Web UI 上传新的知识文档，而不需要修改代码。

---

# 29. 项目最大的技术思路

这个项目最重要的并不是单独使用了：

```text
Chroma
Embedding
Reranker
LLM
```

而是把 RAG 拆成了两个核心问题：

## 第一阶段：解决“找得到”

目标：

> 不要漏掉可能相关的知识。

使用：

```text
Embedding
+
Multi-Query
+
Top-K Recall
+
RRF
```

---

## 第二阶段：解决“找得准”

目标：

> 从候选知识中找到真正适合回答当前问题的内容。

使用：

```text
Reranker
+
Dedup
+
Top-K Context
```

---

## 第三阶段：解决“答得对”

目标：

> 不要让大模型脱离知识库自由发挥。

使用：

```text
Strict Prompt
+
Source Citation
+
Minimum Score
+
Fallback
```

所以整个系统可以总结为：

```text
找得到
   ↓
找得准
   ↓
答得对
   ↓
可追溯
```

---

# 30. 后续优化方向

后续可以继续从以下方向优化：

### 30.1 Query Router

根据问题类型选择不同检索策略：

```text
价格问题
 → 套餐知识

营业时间
 → 营业规则

退款问题
 → 售后规则

预约问题
 → 预约流程
```

---

### 30.2 混合检索

当前主要依赖向量检索。

后续可以加入：

```text
BM25 / Keyword Search
+
Vector Search
```

形成：

```text
Hybrid Search
```

对于：

```text
套餐编号
价格
手机号
日期
特殊名称
```

等关键词敏感的信息可能更加稳定。

---

### 30.3 RAG 自动评测

建立固定测试集：

```text
问题
标准答案
正确文档
```

然后自动计算：

```text
Recall
Precision
MRR
Hit Rate
Answer Accuracy
```

让每次修改 RAG 策略之后都可以进行量化比较。

---

### 30.4 缓存

对于高频问题：

```text
营业时间
地址
联系电话
常见套餐
```

可以加入缓存：

```text
Query
 ↓
Cache Hit？
 ↓
是 → 直接返回
否 → RAG
```

减少重复 Embedding 和 LLM 推理。

---

# 31. 一句话总结

> **SME Guard 是一个面向小微商户的本地化 RAG 智能客服系统，通过语义知识库、Multi-Query、RRF、Reranker 和严格生成约束，将“知识召回”和“答案生成”解耦，在保证回答可追溯性的同时降低大模型幻觉，并支持商户通过 Web UI 自主维护知识库。**

---

# 32. 项目演进总结

整个项目的核心演进可以浓缩成下面这张图：

```text
                    初始方案
                       │
                       ▼
              基础 Vector RAG
                       │
                       ▼
             发现回答不稳定
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
       检索问题                 生成问题
            │                     │
            ▼                     ▼
       优化 Chunk              Prompt 约束
            │                     │
            ▼                     ▼
       Multi-Query             Fallback
            │
            ▼
           RRF
            │
            ▼
         Top-K Recall
            │
            ▼
        BGE Reranker
            │
            ▼
        Top-K Context
            │
            ▼
        Local LLM
            │
            ▼
       可追溯客服回答
```

最终形成：

> **知识库 → 检索 → 重排 → 上下文 → 生成 → 追溯 → 监控**

的一套完整智能客服 RAG pipeline。
