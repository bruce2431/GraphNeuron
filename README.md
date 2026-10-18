# MemGraph

> 版本 **v1.0.0**。仓库：[bruce2431/GraphNeuron](https://github.com/bruce2431/GraphNeuron)（MemGraph 实现已合并入该公开仓库）。

认知社群记忆图原型（Pj11 SubPj9）。核心命题：**认知不是因为语义相似而聚合，而是因为使用（检索次数）而聚合。**

把神经元的三层结构（`l3.raw` → `l2.mem` → `l1.cog`）扩展成一张**带证据的图**，在图上做社群检测与多粒度检索，再用真实检索反馈反哺记忆结构。

## 架构

```
l2.mem/mem.db ──blocks──▶ 记忆节点
      │                      │
      │                ┌─────┴─────┐
      │                │  五类边   │
      │                └─────┬─────┘
      │      reference / entity / semantic / temporal / usage
      │                      │
      └──precog.db──▶ usage ─┘
                             ▼
                    Leiden 社群检测
                             ▼
                社群文本（LLM 可直接阅读）
                             ▼
        recall: mem / graph / cog / community
                             ▼
                        usage 信号
                             ▼
                       反哺循环（三级）
```

- **节点**：每条 mem 一个节点，文本 = `blocks` 拼接。
- **边**：`revelant` 引用、共享实体、语义相似、时间邻近、共召回使用；每条边带 `weight` 与 `evidence`。
- **社群**：多分辨率 Leiden；社群文本包含核心记忆、内部关系、相邻社群。
- **recall 四粒度**：
  - `mem`：块级检索（baseline / dense / bm25 / rrf / mix / hybrid / hybrid_mmr），是唯一写 precog 的粒度；
  - `graph`：种子 + Personalized PageRank；
  - `cog`：社群概念匹配后在社群内排序；
  - `community`：只返回社群文本。

## 反哺循环

原则：**先改权重和结构，最后才动文本；原块永不改写。**

| 级别 | 触发 | 动作 |
|------|------|------|
| `usage_weight` | 自动 | precog 的 accuracy 分级加权（`true=1` / `revelant=0.5` / `false=0`）+ 180 天半衰期 → `usage` 边 |
| `structure_diagnose` | 自动 | 块粒度诊断：多话题 mem、噪锚点、孤立被用、重复块、跨社群桥 |
| `derived_rewrite` | 门控 | 只对「有 usage 信号且结构命中」的候选调用宿主 LLM，写 `l1.cog/block_derived.json`（append-only，`pending → accepted/rejected`） |

关键判据：

- 反馈单位是 **block**，不是整条 mem；
- 没有 usage 信号的 mem 不进入 LLM 改写队列（避免盲改）；
- 多话题用 complete-linkage 聚类，避免链式合并掩盖话题；
- 重复块只在被使用过的块之间比对；
- derived 只追加，原文与历史版本始终可回溯。

## 借用宿主接口

本仓库不持有模型密钥，也不重复实现版本系统：

- **LLM**（`src/host.ts`）：读神经元 `config.yaml` 的 `llm.provider/model/temperature/thinking`，密钥取全局 `credentials.json` 的 `providers.<provider>`，协议适配走宿主的 `resolveProtocolFetch`。宿主按工作区相对位置查找（逐级向上找 `Pj16-CodeAgent构建/Floria/src`）；找不到时抛错，调用方跳过 LLM 阶段，`build / eval / recall` 仍可正常运行。
- **版本**（`src/version.ts`）：与宿主同构的 `l1.cog/versions/vN + manifest.json`，支持 list / snapshot / restore。快照文件：`cog.json`、`cog_graph.json`、`community.json`、`cog2.json`。

## 命令

```bash
bun install

bun run src/cli.ts build   --dataset ljj   # 建图 + 社群
bun run src/cli.ts eval    --dataset ljj   # 检索评测
bun run src/cli.ts recall mem "查询" --dataset ljj
bun run src/cli.ts recall graph|cog|community "查询" --dataset ljj

bun run src/cli.ts feedback --dataset ljj              # 结构诊断 → data/ljj/feedback.json
bun run src/cli.ts feedback rewrite --dataset ljj --limit 5 [--dry-run]

bun run src/cli.ts version list|snapshot|restore --dataset ljj
bun test
```

## 数据

数据目录**不在本仓库内**，需自备神经元：

```
data/ljj-neuron/          # 或任意兼容 l2.mem/mem.db 的神经元
├── l2.mem/mem.db         # memory_id / blocks / revelant / source
├── l2.mem/embeddings.npy
├── l1.cog/               # precog.db / cog / community / versions
└── config.yaml
```

构建产物写入 `data/ljj/`（`graph.json`、`community.json`、`embeddings.json` 缓存、`feedback.json`）。

评测用查询集 `src/evalQueriesLjj.ts` 含个人信息，**只保留在本地、不进入公开仓库**；缺失时 `eval` 的 LJJ 集退化为空，其余功能不受影响。

## 参考指标

李京瑾库（4391 条 mem，3676 条有 blocks，13385 块）：

- 图：3676 节点 / 34491 边（usage 306）/ 14 社群 / modularity 0.606
- 检索（mix，100 条查询）：hit@1 78.3% / hit@3 83.3% / hit@5 86.7% / MRR 0.811
- 诊断：多话题 49 / 噪锚点 9 / 孤立被用 1 / 重复块 1 / 跨社群桥 243

## 实现说明

- TypeScript + Bun（`bun:sqlite`），无外部运行时依赖。
- Leiden 实现内嵌于 `src/leiden.ts`。
- 嵌入模型：`BAAI/bge-small-zh-v1.5`（512 维），向量缓存按 `Bun.hash(text)` 复用。
