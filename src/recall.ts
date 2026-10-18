/**
 * 分粒度 recall：
 *   mem       记忆检索（唯一会写 precog 的模式）
 *   graph     图检索（种子 + Personalized PageRank + 子图/路径）
 *   cog       认知检索（社群概念匹配 → 社群内成员记忆）
 *   community 社群检索（只返回高质量社群文本）
 */

import { Database } from 'bun:sqlite'
import { mkdirSync } from 'node:fs'
import { join } from 'node:path'
import {
  loadMemories, loadLjjMemories, loadPrecog, usageCounts, defaultNeuronPath, defaultLjjNeuronPath,
} from './load.ts'
import { buildUsageSignal, type UsageSignal } from './usage.ts'
import { buildBlockVectors, embedQuery } from './embed.ts'
import { buildMemGraph, loadGraph, saveGraph } from './graph.ts'
import { detectCommunities, loadCommunities, saveCommunities, type CommunityResult } from './community.ts'
import { rank, tokenize, MIX_WEIGHTS_BY_DATASET, type RankContext, type Variant } from './retrieval.ts'
import type { MemEntry, MemGraph, RetrievedMemory } from './types.ts'

export type RecallMode = 'mem' | 'graph' | 'cog' | 'community'

/** pj11 = 项目记忆库；ljj = 李京瑾神经元（data/ljj-neuron/l2.mem/，唯一测试库）。 */
export type Dataset = 'pj11' | 'ljj'

export interface Runtime {
  dataset: Dataset
  neuronPath: string
  dataDir: string
  entries: MemEntry[]
  vectors: Map<string, Float32Array[]>
  usage: Map<string, number>
  /** accuracy 加权 + 衰减后的使用信号（反哺循环输入） */
  usageSignal: UsageSignal
  graph: MemGraph
  communities: CommunityResult
}

export interface RecallOptions {
  topK?: number
  variant?: Variant
  /** 写入 precog 的 db 路径；缺省 = 原型 data/precog.db，仅在 mem 模式写入 */
  precogDb?: string
  seedK?: number
  alpha?: number
  resolution?: number
}

export interface RecallResult {
  mode: RecallMode
  query: string
  results: RetrievedMemory[]
  concepts?: Array<{ name: string; description: string; cos: number; members: string[] }>
  communities?: Array<{ id: string; title: string; cos: number; text: string }>
  subgraph?: {
    nodes: Array<{ id: string; activation: number; title: string }>
    edges: Array<{ source: string; target: string; type: string; weight: number; evidence: string }>
  }
  path?: Array<{ id: string; title: string }>
  precog?: { record_id: string; db: string } | null
  note?: string
}

export function defaultDataDir(dataset: Dataset = 'pj11'): string {
  return join(import.meta.dir, '..', 'data', ...(dataset === 'ljj' ? ['ljj'] : []))
}

/** 建库：向量 + 记忆图 + 社群。已有缓存也会重新建图。 */
export async function buildRuntime(
  opts: { dataset?: Dataset; neuronPath?: string; dataDir?: string; resolution?: number } = {},
): Promise<{ runtime: Runtime; stats: Record<string, unknown> }> {
  const dataset = opts.dataset ?? 'pj11'
  const neuronPath = opts.neuronPath ?? (dataset === 'pj11' ? defaultNeuronPath() : defaultLjjNeuronPath())
  const dataDir = opts.dataDir ?? defaultDataDir(dataset)
  mkdirSync(dataDir, { recursive: true })
  const entries = dataset === 'ljj' ? loadLjjMemories() : loadMemories(neuronPath)
  const { vectors, encoded, fromCache } = await buildBlockVectors(entries, dataDir)
  const precog = loadPrecog(neuronPath)
  const usage = usageCounts(precog)
  const usageSignal = buildUsageSignal(precog)
  const graph = buildMemGraph(entries, vectors, precog, { temporalAll: dataset === 'ljj' })
  const communities = detectCommunities(graph, entries, { resolution: opts.resolution ?? 1.0 })
  saveGraph(dataDir, graph)
  saveCommunities(dataDir, communities)
  const runtime: Runtime = { dataset, neuronPath, dataDir, entries, vectors, usage, usageSignal, graph, communities }
  return {
    runtime,
    stats: {
      dataset,
      memories: entries.length,
      active: entries.filter((e) => !e.deprecatedBy).length,
      blocks: entries.reduce((s, e) => s + e.blocks.length, 0),
      vectorsEncoded: encoded,
      vectorCacheHit: fromCache,
      graphNodes: graph.nodes.length,
      graphEdges: graph.edges.length,
      edgeTypes: countBy(graph.edges.map((e) => e.type)),
      communities: communities.communities.length,
      communityEdges: communities.edges.length,
      modularity: communities.modularity,
      precogRecords: precog.length,
    },
  }
}

export function loadRuntime(opts: { dataset?: Dataset; neuronPath?: string; dataDir?: string } = {}): Runtime {
  const dataset = opts.dataset ?? 'pj11'
  const neuronPath = opts.neuronPath ?? (dataset === 'pj11' ? defaultNeuronPath() : defaultLjjNeuronPath())
  const dataDir = opts.dataDir ?? defaultDataDir(dataset)
  const entries = dataset === 'ljj' ? loadLjjMemories() : loadMemories(neuronPath)
  const graph = loadGraph(dataDir)
  const communities = loadCommunities(dataDir)
  if (!graph || !communities) throw new Error(`图/社群未构建，先运行 build（dataDir=${dataDir}）`)
  const precog = loadPrecog(neuronPath)
  const usage = usageCounts(precog)
  const usageSignal = buildUsageSignal(precog)
  // 载入缓存向量（buildRuntime 已写盘；这里只读缓存，缺块给零向量会失真，故报错）
  const vectors = new Map<string, Float32Array[]>()
  for (const e of entries) {
    if (e.deprecatedBy) continue
    vectors.set(e.id, [])
  }
  return { dataset, neuronPath, dataDir, entries, vectors, usage, usageSignal, graph, communities }
}

function countBy(xs: string[]): Record<string, number> {
  const m: Record<string, number> = {}
  for (const x of xs) m[x] = (m[x] ?? 0) + 1
  return m
}

// ── precog（仅 mem 模式） ──

function ensurePrecogDb(path: string): void {
  mkdirSync(join(path, '..'), { recursive: true })
  const db = new Database(path)
  db.run(`CREATE TABLE IF NOT EXISTS precog (
    record_id TEXT PRIMARY KEY, status TEXT, query TEXT, keywords TEXT,
    source TEXT, top_k INTEGER, results TEXT, description TEXT)`)
  db.close()
}

export function writePrecog(
  dbPath: string,
  query: string,
  results: RetrievedMemory[],
  topK: number,
): string {
  ensurePrecogDb(dbPath)
  const now = new Date()
  const p = (n: number) => String(n).padStart(2, '0')
  const stamp = `${now.getFullYear()}-${p(now.getMonth() + 1)}-${p(now.getDate())}-${p(now.getHours())}:${p(now.getMinutes())}:${p(now.getSeconds())}`
  const recordId = `PC_PJ11PROTO_${stamp}`
  const db = new Database(dbPath)
  try {
    db.query('INSERT OR REPLACE INTO precog VALUES (?,?,?,?,?,?,?,?)').run(
      recordId,
      'pre',
      query,
      JSON.stringify(tokenize(query)),
      'memgraph-prototype',
      topK,
      JSON.stringify(results.map((r) => ({ id: r.id, accuracy: '', summary: r.blocks.join(' ').slice(0, 60) }))),
      '',
    )
  } finally {
    db.close()
  }
  return recordId
}

// ── 检索 ──

async function makeContext(rt: Runtime, query: string, dataDir: string): Promise<RankContext> {
  // loadRuntime 不带向量；build 路径有。若为空则从缓存补齐。
  if (!rt.vectors.size || [...rt.vectors.values()].every((v) => v.length === 0)) {
    const { vectors } = await buildBlockVectors(rt.entries, dataDir)
    rt.vectors = vectors
  }
  const qVec = await embedQuery(query)
  return {
    entries: rt.entries,
    vectors: rt.vectors,
    qVec,
    usage: rt.usage,
    mixWeights: MIX_WEIGHTS_BY_DATASET[rt.dataset],
  }
}

export async function recall(
  mode: RecallMode,
  query: string,
  rt: Runtime,
  opts: RecallOptions = {},
): Promise<RecallResult> {
  const k = opts.topK ?? 5
  const topK = k

  if (mode === 'mem') {
    const ctx = await makeContext(rt, query, rt.dataDir)
    const results = rank(ctx, query, opts.variant ?? 'mix', topK)
    const db = opts.precogDb ?? join(rt.dataDir, 'precog.db')
    const recordId = writePrecog(db, query, results, topK)
    return {
      mode,
      query,
      results,
      precog: { record_id: recordId, db },
      note: '仅 mem 模式写 precog（accuracy 待标注）',
    }
  }

  if (mode === 'community') {
    const qVec = await embedQuery(query)
    const scored = await scoreCommunities(rt, qVec)
    return {
      mode,
      query,
      results: [],
      communities: scored.slice(0, topK).map((c) => ({
        id: c.community.id,
        title: c.community.title,
        cos: Number(c.cos.toFixed(4)),
        text: c.community.text,
      })),
      precog: null,
      note: 'community 模式只返回社群文本，不写 precog',
    }
  }

  if (mode === 'cog') {
    const qVec = await embedQuery(query)
    const matched = (await scoreCommunities(rt, qVec)).filter((c) => c.cos >= 0.35).slice(0, 3)
    const ctx = await makeContext(rt, query, rt.dataDir)
    const memberSet = new Set(matched.flatMap((m) => m.community.members))
    const inCommunity = ctx.entries.filter((e) => memberSet.has(e.id))
    const results = rank({ ...ctx, entries: inCommunity }, query, opts.variant ?? 'mix', topK)
    return {
      mode,
      query,
      results,
      concepts: matched.map((c) => ({
        name: c.community.title,
        description: c.community.text.split('\n').slice(1, 4).join(' '),
        cos: Number(c.cos.toFixed(4)),
        members: c.community.members,
      })),
      communities: matched.map((c) => ({
        id: c.community.id,
        title: c.community.title,
        cos: Number(c.cos.toFixed(4)),
        text: c.community.text,
      })),
      precog: null,
      note: 'cog 模式 = 社群概念匹配 + 社群内记忆排序，不写 precog',
    }
  }

  // graph：种子 + PPR
  const ctx = await makeContext(rt, query, rt.dataDir)
  const seedK = opts.seedK ?? 8
  const seeds = rank(ctx, query, opts.variant ?? 'mix', seedK)
  return graphRecall(rt, query, seeds, k, opts.alpha ?? 0.6)
}

async function scoreCommunities(rt: Runtime, qVec: Float32Array): Promise<Array<{ community: CommunityResult['communities'][number]; cos: number }>> {
  const out: Array<{ community: CommunityResult['communities'][number]; cos: number }> = []
  for (const c of rt.communities.communities) {
    const v = await embedQuery(c.text)
    let s = 0
    for (let i = 0; i < Math.min(v.length, qVec.length); i++) s += v[i]! * qVec[i]!
    out.push({ community: c, cos: s })
  }
  out.sort((a, b) => b.cos - a.cos)
  return out
}

function graphRecall(
  rt: Runtime,
  query: string,
  seeds: RetrievedMemory[],
  k: number,
  alpha: number,
): RecallResult {
  const adj = new Map<string, Array<{ id: string; weight: number; type: string; evidence: string }>>()
  for (const e of rt.graph.edges) {
    if (!adj.has(e.source)) adj.set(e.source, [])
    if (!adj.has(e.target)) adj.set(e.target, [])
    adj.get(e.source)!.push({ id: e.target, weight: e.weight, type: e.type, evidence: e.evidence })
    adj.get(e.target)!.push({ id: e.source, weight: e.weight, type: e.type, evidence: e.evidence })
  }
  const ids = rt.graph.nodes.map((n) => n.id)
  const restart = new Map<string, number>()
  for (const s of seeds) restart.set(s.id, s.score)
  let total = 0
  for (const v of restart.values()) total += v
  if (total > 0) for (const [id, v] of restart) restart.set(id, v / total)

  let p = new Map(ids.map((id) => [id, restart.get(id) ?? 0]))
  for (let it = 0; it < 80; it++) {
    const np = new Map<string, number>()
    for (const id of ids) np.set(id, (1 - alpha) * (restart.get(id) ?? 0))
    for (const id of ids) {
      const pv = p.get(id) ?? 0
      if (!pv) continue
      const list = adj.get(id) ?? []
      let deg = 0
      for (const x of list) deg += x.weight
      if (!deg) continue
      for (const x of list) np.set(x.id, (np.get(x.id) ?? 0) + alpha * pv * (x.weight / deg))
    }
    let delta = 0
    for (const id of ids) delta += Math.abs((np.get(id) ?? 0) - (p.get(id) ?? 0))
    p = np
    if (delta < 1e-9) break
  }

  const titleOf = new Map(rt.graph.nodes.map((n) => [n.id, n.title]))
  const seedIds = new Set(seeds.map((s) => s.id))
  const ranked = [...p.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, k)
  const resultIds = new Set(ranked.map(([id]) => id))

  const subEdges = rt.graph.edges
    .filter((e) => resultIds.has(e.source) && resultIds.has(e.target))
    .sort((a, b) => b.weight - a.weight)

  // 主路径：从激活最高节点沿最强边（不回头）
  const path: Array<{ id: string; title: string }> = []
  let cur = ranked[0]?.[0]
  const seen = new Set<string>()
  while (cur && path.length < 6 && !seen.has(cur)) {
    path.push({ id: cur, title: titleOf.get(cur) ?? cur })
    seen.add(cur)
    const next = rt.graph.edges
      .filter((e) => e.source === cur || e.target === cur)
      .filter((e) => resultIds.has(e.source === cur ? e.target : e.source))
      .filter((e) => { const o = e.source === cur ? e.target : e.source; return !seen.has(o) })
      .sort((a, b) => b.weight - a.weight)[0]
    cur = next ? (next.source === cur ? next.target : next.source) : undefined
  }

  const results: RetrievedMemory[] = ranked.map(([id, act]) => {
    const e = rt.entries.find((x) => x.id === id)!
    return {
      id,
      score: Number(act.toFixed(6)),
      cos: 0,
      bm25: 0,
      kw: 0,
      usage: 0,
      recency: 0,
      title: e?.blocks[0]?.slice(0, 60) ?? titleOf.get(id) ?? id,
      blocks: e?.blocks ?? [],
    }
  })

  return {
    mode: 'graph',
    query,
    results,
    subgraph: {
      nodes: ranked.map(([id, act]) => ({ id, activation: Number(act.toFixed(6)), title: titleOf.get(id) ?? id })),
      edges: subEdges.slice(0, 20).map((e) => ({
        source: e.source,
        target: e.target,
        type: e.type,
        weight: Number(e.weight.toFixed(3)),
        evidence: e.evidence,
      })),
    },
    path,
    precog: null,
    note: `graph 模式 = ${seedIds.size} 个种子 + PPR(α=${alpha})，不写 precog`,
  }
}
