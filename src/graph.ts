/**
 * 记忆图构建。
 *
 * 节点 = 一条记忆；边全部带 type + evidence，保证 LLM 能读出「为什么相连」：
 *   reference  显式引用（revelant 指向另一条记忆）
 *   entity     共享实体/路径/领域词（证据列出共享项）
 *   semantic   语义近邻（BGE cos）
 *   temporal   时序相邻（LOG 连续事件）
 *   usage      共召回（precog 真实命中）
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { buildUsageSignal, type UsageOptions } from './usage.ts'
import type { EdgeType, MemEntry, MemGraph, MemGraphEdge, PrecogRecord } from './types.ts'

const EN_STOP = new Set([
  'this', 'that', 'with', 'from', 'have', 'has', 'were', 'will', 'would', 'could', 'should',
  'http', 'https', 'html', 'true', 'false', 'null', 'none', 'json', 'type', 'name', 'path',
  'index', 'src', 'api', 'app', 'data', 'test', 'docs', 'node', 'bun', 'run', 'use', 'used',
  'file', 'files', 'code', 'the', 'and', 'for', 'not', 'but', 'all', 'any', 'into', 'open',
])

const CJK_STOP = new Set([
  '这个', '那个', '一个', '我们', '你们', '他们', '可以', '需要', '进行', '通过', '因为',
  '所以', '但是', '然后', '已经', '没有', '不是', '就是', '还是', '以及', '其中', '目前',
  '问题', '情况', '结果', '方式', '内容', '时候', '现在', '之前', '之后', '一样', '两个',
])

export interface EntityIndex {
  /** entity → 出现它的 memory id 集合 */
  postings: Map<string, Set<string>>
  idf: Map<string, number>
  perEntry: Map<string, Set<string>>
}

function latinEntities(text: string): string[] {
  const out: string[] = []
  for (const m of text.match(/[A-Za-z][A-Za-z0-9_.-]{3,}/g) ?? []) {
    if (EN_STOP.has(m.toLowerCase())) continue
    out.push(m)
  }
  for (const m of text.match(/(?:[\w\u4e00-\u9fff.-]+\/)+[\w\u4e00-\u9fff.-]+/g) ?? []) {
    out.push(m)
  }
  return out
}

function cjkNgrams(text: string): string[] {
  const out: string[] = []
  for (const run of text.match(/[\u4e00-\u9fff]{2,}/g) ?? []) {
    const chars = [...run]
    for (let n = 2; n <= 6; n++) {
      for (let i = 0; i + n <= chars.length; i++) {
        const g = chars.slice(i, i + n).join('')
        if (!CJK_STOP.has(g)) out.push(g)
      }
    }
  }
  return out
}

/** 构建实体索引：拉丁标识 + 路径 + 高频中文领域词（df 过滤 + 去子串）。 */
export function buildEntityIndex(entries: MemEntry[]): EntityIndex {
  const perEntry = new Map<string, Set<string>>()
  const raw = new Map<string, Set<string>>()
  const cjkDf = new Map<string, number>()

  for (const e of entries) {
    if (e.deprecatedBy) continue
    const set = new Set<string>()
    for (const x of latinEntities(e.text)) set.add(x)
    const seenCjk = new Set<string>()
    for (const g of cjkNgrams(e.text)) {
      if (seenCjk.has(g)) continue
      seenCjk.add(g)
      cjkDf.set(g, (cjkDf.get(g) ?? 0) + 1)
    }
    // 先存拉丁实体；中文词在全局 df 算完后再补
    perEntry.set(e.id, set)
    for (const x of set) {
      if (!raw.has(x)) raw.set(x, new Set())
      raw.get(x)!.add(e.id)
    }
  }

  const nEntries = perEntry.size
  const candidates = [...cjkDf.entries()]
    .filter(([g, df]) => g.length >= 2 && df >= 4 && df <= nEntries * 0.3)
    .map(([g, df]) => ({ g, df, score: df * Math.log(g.length + 1) }))
    .sort((a, b) => b.score - a.score || b.g.length - a.g.length)

  const kept: string[] = []
  for (const c of candidates) {
    if (kept.some((k) => k.includes(c.g))) continue
    kept.push(c.g)
    if (kept.length >= 150) break
  }

  // 回填中文实体 postings
  for (const e of entries) {
    if (e.deprecatedBy) continue
    const set = perEntry.get(e.id)!
    for (const g of kept) {
      if (e.text.includes(g)) set.add(g)
    }
    for (const x of set) {
      if (!raw.has(x)) raw.set(x, new Set())
      raw.get(x)!.add(e.id)
    }
  }

  const idf = new Map<string, number>()
  for (const [x, ids] of raw) idf.set(x, Math.log(1 + (nEntries - ids.size + 0.5) / (ids.size + 0.5)))
  return { postings: raw, idf, perEntry }
}

function meanNorm(vs: Float32Array[] | undefined): Float32Array | null {
  if (!vs?.length) return null
  const dim = vs[0]!.length
  const out = new Float32Array(dim)
  for (const v of vs) for (let i = 0; i < dim; i++) out[i]! += v[i]!
  let n = 0
  for (const x of out) n += x * x
  n = Math.sqrt(n) || 1
  for (let i = 0; i < dim; i++) out[i]! /= n
  return out
}

function dot(a: Float32Array, b: Float32Array): number {
  let s = 0
  for (let i = 0; i < Math.min(a.length, b.length); i++) s += a[i]! * b[i]!
  return s
}

function edgeKey(a: string, b: string, t: EdgeType): string {
  return `${a < b ? a : b}\u0000${a < b ? b : a}\u0000${t}`
}

function addEdge(map: Map<string, MemGraphEdge>, e: MemGraphEdge): void {
  const k = edgeKey(e.source, e.target, e.type)
  const old = map.get(k)
  if (!old) {
    map.set(k, e)
    return
  }
  old.weight += e.weight
  if (!old.evidence.includes(e.evidence)) old.evidence = `${old.evidence}；${e.evidence}`
}

/** 语义边是「同一对邻居」而非多条证据：只保留较大的 cos，避免双向命中把权重叠成 >1。 */
function addSemanticEdge(map: Map<string, MemGraphEdge>, e: MemGraphEdge): void {
  const k = edgeKey(e.source, e.target, e.type)
  const old = map.get(k)
  if (!old || e.weight > old.weight) map.set(k, e)
}

export interface BuildGraphOptions {
  semanticThreshold?: number
  semanticTopK?: number
  temporalWindowMs?: number
  entityWeightFloor?: number
  /** 时序边是否覆盖所有来源（李京瑾聊天库）；默认仅 LOG 来源 */
  temporalAll?: boolean
  /** 使用信号：accuracy 权重 / 衰减 / 共召回窗口 */
  usage?: UsageOptions
}

export function buildMemGraph(
  entries: MemEntry[],
  vectors: Map<string, Float32Array[]>,
  precog: PrecogRecord[] = [],
  opts: BuildGraphOptions = {},
): MemGraph {
  const threshold = opts.semanticThreshold ?? 0.62
  const topK = opts.semanticTopK ?? 3
  const windowMs = opts.temporalWindowMs ?? 6 * 3600_000
  const entityFloor = opts.entityWeightFloor ?? 3.0

  const active = entries.filter((e) => !e.deprecatedBy && e.blocks.length)
  const byId = new Map(active.map((e) => [e.id, e]))
  const index = buildEntityIndex(active)
  const edges = new Map<string, MemGraphEdge>()

  // 1) 共享实体：走倒排索引累加（O(Σ df²)），避免大库全量两两求交。
  //    df 过大的词（如人名）会把所有人连成一团，直接跳过。
  const MAX_ENTITY_DF = 200
  const pairShared = new Map<string, { a: string; b: string; shared: Array<{ x: string; idf: number }> }>()
  for (const [x, ids] of index.postings) {
    const idf = index.idf.get(x) ?? 0
    if (idf < 0.7) continue
    const arr = [...ids].filter((id) => byId.has(id))
    if (arr.length < 2 || arr.length > MAX_ENTITY_DF) continue
    for (let i = 0; i < arr.length; i++) {
      for (let j = i + 1; j < arr.length; j++) {
        const p = arr[i]!
        const q = arr[j]!
        const [a, b] = p < q ? [p, q] : [q, p]
        const k = `${a}\u0000${b}`
        let rec = pairShared.get(k)
        if (!rec) {
          rec = { a, b, shared: [] }
          pairShared.set(k, rec)
        }
        rec.shared.push({ x, idf })
      }
    }
  }
  const entityCand: MemGraphEdge[] = []
  for (const rec of pairShared.values()) {
    rec.shared.sort((p, q) => q.idf - p.idf)
    const top = rec.shared.slice(0, 4)
    const w = top.reduce((s, p) => s + p.idf, 0)
    if (w < entityFloor) continue
    entityCand.push({
      source: rec.a,
      target: rec.b,
      type: 'entity',
      weight: Number(w.toFixed(4)),
      evidence: `共享实体: ${top.map((p) => p.x).join('、')}`,
    })
  }
  const keepPerNode = new Map<string, number>()
  const kept = new Set<MemGraphEdge>()
  for (const e of [...entityCand].sort((p, q) => q.weight - p.weight)) {
    const ca = keepPerNode.get(e.source) ?? 0
    const cb = keepPerNode.get(e.target) ?? 0
    if (ca >= 10 && cb >= 10) continue
    keepPerNode.set(e.source, ca + 1)
    keepPerNode.set(e.target, cb + 1)
    kept.add(e)
  }
  for (const e of kept) addEdge(edges, e)

  // 2) 语义近邻
  const reps = new Map<string, Float32Array | null>()
  for (const e of active) reps.set(e.id, meanNorm(vectors.get(e.id)))
  for (let i = 0; i < active.length; i++) {
    const a = active[i]!
    const va = reps.get(a.id)
    if (!va) continue
    const sims: Array<{ id: string; cos: number }> = []
    for (let j = 0; j < active.length; j++) {
      if (i === j) continue
      const b = active[j]!
      const vb = reps.get(b.id)
      if (!vb) continue
      const cos = dot(va, vb)
      if (cos >= threshold) sims.push({ id: b.id, cos })
    }
    sims.sort((x, y) => y.cos - x.cos)
    for (const s of sims.slice(0, topK)) {
      addSemanticEdge(edges, {
        source: a.id,
        target: s.id,
        type: 'semantic',
        weight: Number(s.cos.toFixed(4)),
        evidence: `语义相似 cos=${s.cos.toFixed(3)}`,
      })
    }
  }

  // 3) 显式引用（revelant 里出现另一条记忆 id）
  for (const e of active) {
    for (const r of e.revelant) {
      if (!byId.has(r) || r === e.id) continue
      addEdge(edges, {
        source: e.id,
        target: r,
        type: 'reference',
        weight: 3,
        evidence: `显式引用: revelant → ${r}`,
      })
    }
  }

  // 4) 时序相邻（默认仅 LOG 来源；聊天库可放开到全部来源）
  const logEntries = active
    .filter((e) => (opts.temporalAll || e.source === 'LOG') && e.time !== null)
    .sort((a, b) => a.time! - b.time!)
  for (let i = 1; i < logEntries.length; i++) {
    const a = logEntries[i - 1]!
    const b = logEntries[i]!
    const dt = b.time! - a.time!
    if (dt > windowMs) continue
    addEdge(edges, {
      source: a.id,
      target: b.id,
      type: 'temporal',
      weight: 0.25,
      evidence: `时序相邻: 间隔 ${(dt / 60000).toFixed(0)} 分钟`,
    })
  }

  // 5) 共召回（precog → 使用信号）：accuracy 分级 + 时间衰减 + top-k 窗口
  const usageSignal = buildUsageSignal(precog, opts.usage)
  for (const u of usageSignal.edges) {
    if (!byId.has(u.source) || !byId.has(u.target)) continue
    addEdge(edges, {
      source: u.source,
      target: u.target,
      type: 'usage',
      weight: u.weight,
      evidence: u.evidence,
    })
  }

  return {
    nodes: active.map((e) => ({
      id: e.id,
      title: e.blocks[0]?.slice(0, 60) ?? e.id,
      time: e.time,
    })),
    edges: [...edges.values()],
  }
}

export function saveGraph(dataDir: string, graph: MemGraph): string {
  mkdirSync(dataDir, { recursive: true })
  const p = join(dataDir, 'memgraph.json')
  writeFileSync(p, JSON.stringify(graph))
  return p
}

export function loadGraph(dataDir: string): MemGraph | null {
  const p = join(dataDir, 'memgraph.json')
  if (!existsSync(p)) return null
  try {
    return JSON.parse(readFileSync(p, 'utf-8')) as MemGraph
  } catch {
    return null
  }
}
