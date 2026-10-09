# GraphNeuron

事件图记忆引擎 —— **记忆 = 事件的图，检索 = 图上蔓延**。
Pj11 (LoRAMoE) / SubPj9 产出。TS + Bun，零外部依赖。

> 设计定案见 `../[20261009130500]-GraphNeuron设计-v0.2.md`。

## 核心命题

认知不是因为语义相似而聚合，而是因为**使用（检索次数）**而聚合。
因此图上的 `usage` 边在每次共召回时累加（Hebbian），社群即多张记忆图交叠后的完整记录。

## 事件 schema（五维）

```
m = ⟨ Time , Chara₁ , Move , Chara₂ , Content ⟩
```

- `Move` = 单个动作原子，**只取肯定式谓词**（喜欢 / 拒绝 / 告知…），不带否定/时态
- `Content` = 内容槽：字面量 **或嵌套事件**（携带 `m₁(m₂)`，连 `carries` 边）
- 正交限定符：`pol`（极性/否定）、`mod`（情态/时体）、`chara_mode`（集合读法）
- **否定不折进 Move**：否定是作用于事件的算子且有多层辖域，故 `pol` 为事件节点独立字段，嵌套结构天然给出辖域
- `trace` = 溯源痕迹（该事件来自原始记录的哪一层级/片段，管线填充）

## 检索

```
Query(事件模式) → 种子打分(结构指纹+文本向量) → Personalized PageRank 蔓延
             → 诱导子图（top-k + 主路径） → usage 边 +δ（闭环）
```

时空邻近由 `temporal`（权重 `exp(−Δt/τ)`）与 `spatial` 边在蔓延中自然生效。

## 运行

```bash
bun run src/api.ts        # 启动 HTTP API（默认 :8787，PORT 可覆盖）
bun run src/cli.ts demo   # 内置演示全链路
bun test                  # API 调用测试
```

## HTTP API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 存活 / 规模 |
| POST | `/ingest` | `{ events: EventInput[], rebuild?: bool }` 批量入库并重建结构边 |
| POST | `/search` | `{ query: QueryPattern, k?, alpha?, hops?, bumpUsage?, ... }` |
| POST | `/rebuild` | 重建结构边 |
| GET | `/events` `/events/:id` | 事件 |
| GET | `/graph` | 全量快照 |
| POST | `/community` | `{ resolutions?, minSize?, edgeTypes? }` Leiden 社群 |
| POST | `/save` `/load` | `{ path }` JSON 持久化 |
| POST | `/reset` | 清空 |

### 例

```bash
curl -s localhost:8787/ingest -H 'content-type: application/json' -d '{
  "events":[{
    "chara1":"A","move":"告诉","chara2":"B","source":"s1",
    "content":{"chara1":"C","move":"表白","chara2":"D","mod":"曾经"}
  }]}'

curl -s localhost:8787/search -H 'content-type: application/json' \
  -d '{"query":{"move":"告诉","chara1":"A"},"k":5,"hops":3}'
```

## 参数

- 种子权重 `seedWeights`：`move/chara/time/text/pol`（默认 `1.0/0.8/0.6/0.7/0.2`）
- 蔓延阻尼 `alpha` 默认 `0.6`；最大跳数 `hops` 默认 `3`
- 边权重：`carries/causes/cooc/char/temporal/spatial` 可在 `GNOptions` 覆盖
- Embedder 可插拔（默认本地 feature-hash；实现 `Embedder` 接口即可换 BGE / 外部 API）

## 依赖

Bun ≥ 1.3（`bun.lock` 无第三方运行时依赖）。
