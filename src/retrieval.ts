/** mem 检索：baseline(cos+kw) / BM25 / 混合 RRF / usage+近因 / MMR */

import type { MemEntry, RetrievedMemory } from './types.ts'

const segmenter = new Intl.Segmenter('zh', { granularity: 'word' })

const STOP = new Set([
  '的', '了', '是', '在', '不', '和', '就', '都', '与', '及', '或', '而', '被', '把', '对',
  '我', '你', '他', '她', '它', '用户', '这个', '那个', '一个', '什么', '怎么', '如何',
  '可以', '需要', '进行', '通过', '使用', '以及', '没有', '不是', '就是', '还是', '但是',
])

/** 词级切分（与活引擎 splitQuery 同口径：只 ICU 分词，不加 bigram） */
export function tokenizeWords(text: string): string[] {
  const out: string[] = []
  const lower = text.toLowerCase()
  for (const seg of segmenter.segment(lower)) {
    if (!seg.isWordLike) continue
    const w = seg.segment.trim()
    if (!w || STOP.has(w)) continue
    out.push(w)
  }
  return out
}

/** BM25 用：词 + 中文 bigram（弥补 ICU 对专名/复合词的切分歧义） */
export function tokenize(text: string): string[] {
  const lower = text.toLowerCase()
  const out = tokenizeWords(text)
  for (const run of lower.match(/[\u4e00-\u9fff]{2,}/g) ?? []) {
    const chars = [...run]
    for (let i = 0; i + 1 < chars.length; i++) out.push(chars[i]! + chars[i + 1]!)
  }
  return out
}

export type Variant = 'baseline' | 'dense' | 'bm25' | 'rrf' | 'mix' | 'hybrid' | 'hybrid_mmr'

export const VARIANTS: Variant[] = ['baseline', 'dense', 'bm25', 'rrf', 'mix', 'hybrid', 'hybrid_mmr']

export interface MixWeights {
  dense: number
  bm25: number
  kw: number
}

/** mix 默认权重（Pj11 项目记忆 38 条查询网格搜索最优）。 */
export const MIX_WEIGHTS: MixWeights = { dense: 0.6, bm25: 0.1, kw: 0.3 }

/** 分数据集权重：李京瑾 60 条查询（30 精确 + 30 改写）网格搜索最优。 */
export const MIX_WEIGHTS_BY_DATASET: Record<'pj11' | 'ljj', MixWeights> = {
  pj11: MIX_WEIGHTS,
  ljj: { dense: 0.7, bm25: 0.2, kw: 0.1 },
}

export interface RankContext {
  entries: MemEntry[]
  vectors: Map<string, Float32Array[]>
  qVec: Float32Array
  usage: Map<string, number>
  /** mix 变体的权重；缺省用 MIX_WEIGHTS */
  mixWeights?: MixWeights
  now?: number
}

function dot(a: Float32Array, b: Float32Array): number {
  let s = 0
  const n = Math.min(a.length, b.length)
  for (let i = 0; i < n; i++) s += a[i]! * b[i]!
  return s
}

/** 条目代表向量 = 块向量归一化均值（仅 MMR 用） */
function meanVector(vs: Float32Array[] | undefined): Float32Array | null {
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

interface Row {
  entry: MemEntry
  cos: number
  kw: number
  bm25: number
  usage: number
  recency: number
}

interface Bm25Index {
  entries: MemEntry[]
  docs: Map<string, number>[]
  lengths: number[]
  df: Map<string, number>
  N: number
  avgdl: number
}

const BM_CACHE = new WeakMap<MemEntry[], Bm25Index>()

/** 分词索引按 entries 数组身份缓存：大库（李京瑾 3675 条）不能每条查询重切一遍。 */
function prepareBm25(entries: MemEntry[]): Bm25Index {
  const cached = BM_CACHE.get(entries)
  if (cached) return cached
  const active = entries.filter((e) => !e.deprecatedBy && e.blocks.length)
  const docs: Map<string, number>[] = []
  const lengths: number[] = []
  const df = new Map<string, number>()
  let total = 0
  for (const e of active) {
    const tf = new Map<string, number>()
    let len = 0
    for (const t of tokenize(e.text)) {
      tf.set(t, (tf.get(t) ?? 0) + 1)
      len++
    }
    docs.push(tf)
    lengths.push(len)
    total += len
    for (const t of tf.keys()) df.set(t, (df.get(t) ?? 0) + 1)
  }
  const index = { entries: active, docs, lengths, df, N: active.length, avgdl: total / Math.max(1, active.length) }
  BM_CACHE.set(entries, index)
  return index
}

function bm25Score(idx: number, qTokens: string[], bm: Bm25Index): number {
  const tf = bm.docs[idx]!
  const dl = bm.lengths[idx]!
  const k1 = 1.2
  const b = 0.75
  let s = 0
  for (const t of new Set(qTokens)) {
    const f = tf.get(t) ?? 0
    if (!f) continue
    const df = bm.df.get(t) ?? 0
    const idf = Math.log(1 + (bm.N - df + 0.5) / (df + 0.5))
    s += idf * ((f * (k1 + 1)) / (f + k1 * (1 - b + (b * dl) / (bm.avgdl || 1))))
  }
  return s
}

function normalizeMax<T>(rows: T[], get: (r: T) => number): Map<T, number> {
  let max = 0
  for (const r of rows) max = Math.max(max, get(r))
  const m = new Map<T, number>()
  for (const r of rows) m.set(r, max > 0 ? get(r) / max : 0)
  return m
}

/** 全量打分（返回 top 候选，未排序前的候选池） */
export function scoreAll(ctx: RankContext, query: string): Row[] {
  const qTokens = tokenize(query)
  const qSet = [...new Set(tokenizeWords(query))]
  const now = ctx.now ?? Date.now()
  const bm = prepareBm25(ctx.entries)
  const active = bm.entries

  return active.map((e, idx) => {
    const vs = ctx.vectors.get(e.id) ?? []
    let cos = 0
    for (const v of vs) cos = Math.max(cos, dot(v, ctx.qVec))
    const lower = e.text.toLowerCase()
    const hits = qSet.filter((t) => lower.includes(t)).length
    const kw = qSet.length ? hits / qSet.length : 0
    const ageDays = e.time ? Math.max(0, (now - e.time) / 86_400_000) : 365
    return {
      entry: e,
      cos: Math.max(0, cos),
      kw,
      bm25: bm25Score(idx, qTokens, bm),
      usage: ctx.usage.get(e.id) ?? 0,
      recency: Math.exp(-ageDays / 90),
    }
  })
}

function toResult(r: Row, score: number): RetrievedMemory {
  return {
    id: r.entry.id,
    score: Number(score.toFixed(6)),
    cos: Number(r.cos.toFixed(4)),
    bm25: Number(r.bm25.toFixed(4)),
    kw: Number(r.kw.toFixed(4)),
    usage: r.usage,
    recency: Number(r.recency.toFixed(4)),
    title: r.entry.blocks[0]?.slice(0, 60) ?? '',
    blocks: r.entry.blocks,
  }
}

function rrfRank(rows: Row[], by: (r: Row) => number, k = 60): Map<Row, number> {
  const sorted = [...rows].sort((a, b) => by(b) - by(a))
  const m = new Map<Row, number>()
  sorted.forEach((r, i) => m.set(r, 1 / (k + i + 1)))
  return m
}

function mmrSelect(rows: Row[], scoreOf: Map<Row, number>, ctx: RankContext, k: number, lambda = 0.7): Row[] {
  const pool = [...rows].sort((a, b) => (scoreOf.get(b) ?? 0) - (scoreOf.get(a) ?? 0)).slice(0, 60)
  const rep = new Map<Row, Float32Array | null>()
  for (const r of pool) rep.set(r, meanVector(ctx.vectors.get(r.entry.id)))
  const rel = normalizeMax(pool, (r) => scoreOf.get(r) ?? 0)
  const picked: Row[] = []
  while (picked.length < k && pool.length) {
    let best: Row | null = null
    let bestVal = -Infinity
    for (const r of pool) {
      if (picked.includes(r)) continue
      let maxSim = 0
      const rv = rep.get(r)
      if (rv) {
        for (const p of picked) {
          const pv = rep.get(p)
          if (pv) maxSim = Math.max(maxSim, dot(rv, pv))
        }
      }
      const val = lambda * (rel.get(r) ?? 0) - (1 - lambda) * maxSim
      if (val > bestVal) {
        bestVal = val
        best = r
      }
    }
    if (!best) break
    picked.push(best)
  }
  return picked
}

export function rank(ctx: RankContext, query: string, variant: Variant = 'hybrid', k = 10): RetrievedMemory[] {
  const rows = scoreAll(ctx, query)
  if (!rows.length) return []

  if (variant === 'baseline') {
    return rows
      .map((r) => ({ r, s: 0.3 * r.cos + 0.7 * r.kw }))
      .sort((a, b) => b.s - a.s)
      .slice(0, k)
      .map(({ r, s }) => toResult(r, s))
  }
  if (variant === 'dense') {
    return rows
      .sort((a, b) => b.cos - a.cos)
      .slice(0, k)
      .map((r) => toResult(r, r.cos))
  }
  if (variant === 'bm25') {
    const norm = normalizeMax(rows, (r) => r.bm25)
    return rows
      .sort((a, b) => b.bm25 - a.bm25)
      .slice(0, k)
      .map((r) => toResult(r, norm.get(r) ?? 0))
  }

  if (variant === 'mix') {
    const w = ctx.mixWeights ?? MIX_WEIGHTS
    const cosN = normalizeMax(rows, (r) => r.cos)
    const bmN = normalizeMax(rows, (r) => r.bm25)
    const kwN = normalizeMax(rows, (r) => r.kw)
    const score = new Map<Row, number>()
    for (const r of rows) {
      score.set(
        r,
        w.dense * (cosN.get(r) ?? 0) +
          w.bm25 * (bmN.get(r) ?? 0) +
          w.kw * (kwN.get(r) ?? 0),
      )
    }
    return rows
      .sort((a, b) => (score.get(b) ?? 0) - (score.get(a) ?? 0))
      .slice(0, k)
      .map((r) => toResult(r, score.get(r) ?? 0))
  }

  const rDense = rrfRank(rows, (r) => r.cos)
  const rBm25 = rrfRank(rows, (r) => r.bm25)

  if (variant === 'rrf') {
    const score = new Map<Row, number>()
    for (const r of rows) score.set(r, (rDense.get(r) ?? 0) + (rBm25.get(r) ?? 0))
    const norm = normalizeMax(rows, (r) => score.get(r) ?? 0)
    return rows
      .sort((a, b) => (score.get(b) ?? 0) - (score.get(a) ?? 0))
      .slice(0, k)
      .map((r) => toResult(r, norm.get(r) ?? 0))
  }

  // hybrid = RRF 0.75 + usage 0.15 + recency 0.10（各项先归一化到 [0,1]）
  const rrfRaw = new Map<Row, number>()
  for (const r of rows) {
    rrfRaw.set(r, (rDense.get(r) ?? 0) + (rBm25.get(r) ?? 0))
  }
  const rrfNorm = normalizeMax(rows, (r) => rrfRaw.get(r) ?? 0)
  const maxUsage = Math.max(0, ...rows.map((r) => r.usage))
  const score = new Map<Row, number>()
  for (const r of rows) {
    const usageN = maxUsage > 0 ? Math.log1p(r.usage) / Math.log1p(maxUsage) : 0
    score.set(r, 0.75 * (rrfNorm.get(r) ?? 0) + 0.15 * usageN + 0.10 * r.recency)
  }

  if (variant === 'hybrid_mmr') {
    const picked = mmrSelect(rows, score, ctx, k)
    const norm = normalizeMax(rows, (r) => score.get(r) ?? 0)
    return picked.map((r) => toResult(r, norm.get(r) ?? 0))
  }

  const norm = normalizeMax(rows, (r) => score.get(r) ?? 0)
  return rows
    .sort((a, b) => (score.get(b) ?? 0) - (score.get(a) ?? 0))
    .slice(0, k)
    .map((r) => toResult(r, norm.get(r) ?? 0))
}
