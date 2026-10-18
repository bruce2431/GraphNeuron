/**
 * 使用信号：把 precog（真实检索 + accuracy 标注）折叠成反哺循环的一等输入。
 *
 * 设计要点（相对旧实现的修正）：
 *   - accuracy 分级：true=1 / revelant=0.5 / 未标注=0 / false=0（false 是负样本，
 *     不能像旧实现那样与 revelant 同权进边）。
 *   - 时间衰减：同一次共召回按半衰期衰减，旧用法不会永久固化。
 *   - 共召回只在 top-k 前缀内配对（co_hit_window），避免长尾结果两两成边。
 *   - 证据里写查询原句，社群文本/LLM 能读出「为什么相连」。
 */

import type { PrecogRecord } from './types.ts'

export interface AccuracyWeights {
  true: number
  revelant: number
  unlabeled: number
  false: number
}

export const DEFAULT_ACCURACY_WEIGHTS: AccuracyWeights = {
  true: 1,
  revelant: 0.5,
  unlabeled: 0,
  false: 0,
}

export interface UsageStat {
  id: string
  trueHits: number
  relevantHits: number
  falseHits: number
  unlabeledHits: number
  /** 衰减加权的命中强度（true=1 / revelant=0.5） */
  weighted: number
  /** 去重后的查询句（最多保留 8 条） */
  queries: string[]
  lastTime: number | null
}

export interface UsageSignalEdge {
  source: string
  target: string
  weight: number
  evidence: string
  /** 共召回次数（未衰减） */
  hits: number
}

export interface UsageSignal {
  stats: Map<string, UsageStat>
  edges: UsageSignalEdge[]
  records: number
  /** 有正向命中的记忆数 */
  usedMemories: number
}

export interface UsageOptions {
  weights?: AccuracyWeights
  /** 共召回半衰期（天） */
  decayDays?: number
  /** 只在前 window 个结果内配对 */
  coHitWindow?: number
  now?: number
}

const RECORD_TIME = /(\d{4})-(\d{2})-(\d{2})-(\d{2}):(\d{2}):(\d{2})/

export function parseRecordTime(recordId: string): number | null {
  const m = RECORD_TIME.exec(recordId)
  if (!m) return null
  return new Date(
    Number(m[1]), Number(m[2]) - 1, Number(m[3]),
    Number(m[4]), Number(m[5]), Number(m[6]),
  ).getTime()
}

function accWeight(accuracy: string, w: AccuracyWeights): number {
  switch ((accuracy ?? '').toLowerCase()) {
    case 'true': return w.true
    case 'revelant':
    case 'relevant': return w.revelant
    case 'false': return w.false
    default: return w.unlabeled
  }
}

function emptyStat(id: string): UsageStat {
  return {
    id,
    trueHits: 0,
    relevantHits: 0,
    falseHits: 0,
    unlabeledHits: 0,
    weighted: 0,
    queries: [],
    lastTime: null,
  }
}

export function buildUsageSignal(records: PrecogRecord[], opts: UsageOptions = {}): UsageSignal {
  const w = opts.weights ?? DEFAULT_ACCURACY_WEIGHTS
  const halfLifeDays = opts.decayDays ?? 180
  const window = opts.coHitWindow ?? 5
  const now = opts.now ?? Date.now()
  const stats = new Map<string, UsageStat>()
  const pairs = new Map<string, { a: string; b: string; weight: number; hits: number; queries: string[] }>()

  for (const r of records) {
    const t = parseRecordTime(r.record_id) ?? now
    const ageDays = Math.max(0, (now - t) / 86_400_000)
    const decay = Math.pow(0.5, ageDays / Math.max(1, halfLifeDays))

    // 同一次检索里同一 id 只计一次
    const seen = new Set<string>()
    const positive: Array<{ id: string; weight: number }> = []
    let rank = 0
    for (const x of r.results) {
      if (!x.id || seen.has(x.id)) continue
      seen.add(x.id)
      const s = stats.get(x.id) ?? emptyStat(x.id)
      const a = (x.accuracy ?? '').toLowerCase()
      if (a === 'true') s.trueHits++
      else if (a === 'revelant' || a === 'relevant') s.relevantHits++
      else if (a === 'false') s.falseHits++
      else s.unlabeledHits++
      const aw = accWeight(a, w)
      if (aw > 0) {
        s.weighted += aw * decay
        if (s.queries.length < 8 && !s.queries.includes(r.query)) s.queries.push(r.query)
        s.lastTime = Math.max(s.lastTime ?? 0, t)
      }
      stats.set(x.id, s)
      rank++
      if (rank > window) continue
      if (aw > 0) positive.push({ id: x.id, weight: aw })
    }

    // 共召回边：只连同一次成功检索里 top-k 窗口内的正向结果
    for (let i = 0; i < positive.length; i++) {
      for (let j = i + 1; j < positive.length; j++) {
        const p = positive[i]!
        const q = positive[j]!
        const [a, b] = p.id < q.id ? [p.id, q.id] : [q.id, p.id]
        const k = `${a}\u0000${b}`
        let rec = pairs.get(k)
        if (!rec) {
          rec = { a, b, weight: 0, hits: 0, queries: [] }
          pairs.set(k, rec)
        }
        rec.weight += p.weight * q.weight * decay
        rec.hits++
        if (rec.queries.length < 3 && !rec.queries.includes(r.query)) rec.queries.push(r.query)
      }
    }
  }

  const edges: UsageSignalEdge[] = [...pairs.values()].map((p) => ({
    source: p.a,
    target: p.b,
    weight: Number(p.weight.toFixed(4)),
    hits: p.hits,
    evidence: `共召回 ${p.hits} 次：${p.queries.join('；')}`,
  }))
  edges.sort((a, b) => b.weight - a.weight || b.hits - a.hits)

  return {
    stats,
    edges,
    records: records.length,
    usedMemories: [...stats.values()].filter((s) => s.trueHits + s.relevantHits > 0).length,
  }
}

export function usageSummary(signal: UsageSignal): Record<string, unknown> {
  let t = 0
  let r = 0
  let f = 0
  let u = 0
  let weighted = 0
  for (const s of signal.stats.values()) {
    t += s.trueHits
    r += s.relevantHits
    f += s.falseHits
    u += s.unlabeledHits
    weighted += s.weighted
  }
  return {
    records: signal.records,
    usedMemories: signal.usedMemories,
    trueHits: t,
    relevantHits: r,
    falseHits: f,
    unlabeledHits: u,
    weightedHits: Number(weighted.toFixed(3)),
    coHitEdges: signal.edges.length,
  }
}
