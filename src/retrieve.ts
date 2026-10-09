/**
 * GraphNeuron — 图检索（种子匹配 + 蔓延 + 子图诱导）
 *
 * 检索单元是「图」：先按事件模式打分得到种子，再以种子为重启分布做
 * Personalized PageRank（带阻尼的蔓延激活），最后把激活最高的事件诱导成子图，
 * 并抽出主路径。时空邻近通过 temporal/spatial 边在蔓延中自然生效。
 */

import { type GEvent, type Polarity, asArray, parseTime, verbalize } from './schema.ts'
import { type Embedder, cosine, hashEmbed } from './embed.ts'
import { type GEdge, type EdgeType } from './schema.ts'
import type { GraphNeuron } from './store.ts'

export interface QueryPattern {
  time?: number | string
  chara1?: string[] | string
  move?: string
  chara2?: string[] | string
  pol?: Polarity
  mod?: string
  location?: string
  /** 自然语句（补充语义种子） */
  text?: string
}

export interface SearchOptions {
  /** 返回事件数 */
  k?: number
  /** 阻尼系数：越大蔓延越远（默认 0.6） */
  alpha?: number
  /** 最大跳数（限制诱导子图半径，默认 3） */
  hops?: number
  /** 激活阈值 */
  minActivation?: number
  /** 是否回写 usage 边（默认 true） */
  bumpUsage?: boolean
  /** 时间容差（ms），默认 24h */
  timeToleranceMs?: number
  seedWeights?: Partial<SeedWeights>
}

export interface SeedWeights {
  move: number
  chara: number
  time: number
  text: number
  pol: number
}

const SEED_DEFAULTS: SeedWeights = { move: 1.0, chara: 0.8, time: 0.6, text: 0.7, pol: 0.2 }

export interface ScoredEvent {
  id: string
  activation: number
  seedScore: number
  hop: number
  event: GEvent
  verbal: string
}

export interface SubgraphEdge {
  source: string
  target: string
  type: EdgeType
  weight: number
}

export interface SearchResult {
  query: QueryPattern
  k: number
  alpha: number
  hops: number
  seeds: number
  results: ScoredEvent[]
  subgraph: { nodes: { id: string; activation: number; move: string; verbal: string }[]; edges: SubgraphEdge[] }
  path: string[]
  usage_bumped: number
}

function norm(s: string): string {
  return String(s).trim().toLowerCase()
}

/** 打分：结构字段 + 文本向量（返回未归一化的原始分） */
function seedScore(ev: GEvent, q: QueryPattern, w: SeedWeights, qVec: number[] | null, gn: GraphNeuron, tolMs: number): number {
  let s = 0
  if (q.move) {
    const a = norm(q.move)
    const b = norm(ev.move)
    if (a === b) s += w.move
    else if (b.includes(a) || a.includes(b)) s += w.move * 0.5
  }
  const qa = new Set([...asArray(q.chara1), ...asArray(q.chara2)].map(norm))
  if (qa.size > 0) {
    const ea = new Set([...ev.chara1, ...ev.chara2].map(norm))
    let inter = 0
    for (const c of qa) if (ea.has(c)) inter++
    if (inter > 0) s += w.chara * (inter / qa.size)
  }
  if (q.location && ev.location && norm(q.location) === norm(ev.location)) s += w.chara * 0.5
  if (q.time !== undefined && q.time !== null && ev.time != null) {
    const qt = parseTime(q.time)
    if (qt !== null) {
      const dt = Math.abs(qt - ev.time)
      s += w.time * Math.exp(-dt / tolMs)
    }
  }
  if (q.pol && ev.pol !== q.pol) {
    // 极性不符：轻度惩罚（肯定/否定仍同源，只降权不排除）
    s -= w.pol * 0.5
  } else if (q.pol && ev.pol === q.pol) {
    s += w.pol
  }
  if (qVec) {
    const ev = gn.vectorOf(ev.id)
    if (ev) s += w.text * Math.max(0, cosine(qVec, ev))
  }
  return s
}

/** BFS 求从种子集出发的最短跳数（用于限制诱导半径） */
function hopDistances(gn: GraphNeuron, seedIds: string[], maxHops: number): Map<string, number> {
  const adj = gn.adjacency()
  const dist = new Map<string, number>()
  const q: string[] = []
  for (const id of seedIds) {
    dist.set(id, 0)
    q.push(id)
  }
  let head = 0
  while (head < q.length) {
    const cur = q[head++]
    const d = dist.get(cur)!
    if (d >= maxHops) continue
    for (const { nbr } of adj.get(cur) ?? []) {
      if (!dist.has(nbr)) {
        dist.set(nbr, d + 1)
        q.push(nbr)
      }
    }
  }
  return dist
}

/** Personalized PageRank：以 seeds 为重启分布做带阻尼蔓延 */
function ppr(gn: GraphNeuron, restart: Map<string, number>, alpha: number, maxIter = 120): Map<string, number> {
  const adj = gn.adjacency()
  const ids = [...gn.events.keys()]
  let p = new Map<string, number>()
  for (const id of ids) p.set(id, restart.get(id) ?? 0)
  const deg = new Map<string, number>()
  for (const [id, list] of adj) {
    let d = 0
    for (const e of list) d += e.weight
    deg.set(id, d)
  }
  for (let it = 0; it < maxIter; it++) {
    const np = new Map<string, number>()
    for (const id of ids) np.set(id, (1 - alpha) * (restart.get(id) ?? 0))
    for (const id of ids) {
      const pv = p.get(id)!
      if (pv === 0) continue
      const d = deg.get(id)!
      if (d <= 0) continue
      for (const { nbr, weight } of adj.get(id) ?? []) {
        np.set(nbr, np.get(nbr)! + alpha * pv * (weight / d))
      }
    }
    let delta = 0
    for (const id of ids) delta += Math.abs(np.get(id)! - p.get(id)!)
    p = np
    if (delta < 1e-9) break
  }
  return p
}

/** 从最高激活种子出发，沿最强边抽出主路径（避免回头） */
function extractPath(gn: GraphNeuron, top: ScoredEvent[], maxLen: number): string[] {
  if (top.length === 0) return []
  const included = new Set(top.map((t) => t.id))
  const model = new Map(top.map((t) => [t.id, t]))
  const path: string[] = []
  let cur = top[0].id
  const seen = new Set<string>()
  while (cur && path.length < maxLen && !seen.has(cur)) {
    path.push(cur)
    seen.add(cur)
    let best: { id: string; w: number } | null = null
    for (const e of gn.edgesOf(cur)) {
      if (e.type === 'usage' && best?.w && e.weight < best.w) continue
      const other = e.source === cur ? e.target : e.source
      if (!included.has(other) || seen.has(other)) continue
      if (!best || e.weight > best.w) best = { id: other, w: e.weight }
    }
    cur = best?.id ?? ''
  }
  // 若路径太短，用激活次高者续接（弱化约束）
  if (path.length < 2 && top.length > 1) {
    for (const t of top.slice(1)) {
      if (!path.includes(t.id)) {
        path.push(t.id)
        break
      }
    }
  }
  return path
}

export function search(gn: GraphNeuron, q: QueryPattern, opts: SearchOptions = {}): SearchResult {
  const k = Math.max(1, opts.k ?? 5)
  const alpha = Math.min(0.95, Math.max(0.05, opts.alpha ?? 0.6))
  const hops = Math.max(1, opts.hops ?? 3)
  const minActivation = opts.minActivation ?? 1e-4
  const w: SeedWeights = { ...SEED_DEFAULTS, ...(opts.seedWeights ?? {}) }
  const tolMs = opts.timeToleranceMs ?? 24 * 3600 * 1000

  const qVec = q.text ? hashEmbed(q.text, gn.embedder.dim) : null

  // 1) 种子打分
  const raw = new Map<string, number>()
  let total = 0
  for (const ev of gn.events.values()) {
    const s = seedScore(ev, q, w, qVec, gn, tolMs)
    if (s > 0) {
      raw.set(ev.id, s)
      total += s
    }
  }
  if (total <= 0) {
    return { query: q, k, alpha, hops, seeds: 0, results: [], subgraph: { nodes: [], edges: [] }, path: [], usage_bumped: 0 }
  }
  const restart = new Map<string, number>()
  for (const [id, s] of raw) restart.set(id, s / total)

  // 2) 蔓延
  const p = ppr(gn, restart, alpha)

  // 3) 跳数限制
  const dist = hopDistances(gn, [...restart.keys()], hops)

  // 4) 取 top-k
  const ranked = [...p.entries()]
    .filter(([id, a]) => a >= minActivation && dist.has(id))
    .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
    .slice(0, k)

  const results: ScoredEvent[] = ranked.map(([id, activation]) => {
    const event = gn.events.get(id)!
    return {
      id,
      activation: Number(activation.toFixed(6)),
      seedScore: Number((raw.get(id) ?? 0).toFixed(6)),
      hop: dist.get(id) ?? -1,
      event,
      verbal: verbalize(event, (cid) => gn.events.get(cid)),
    }
  })

  // 5) 诱导子图
  const nodeSet = new Set(ranked.map(([id]) => id))
  const edges: SubgraphEdge[] = []
  for (const e of gn.edges.values()) {
    if (nodeSet.has(e.source) && nodeSet.has(e.target)) {
      edges.push({ source: e.source, target: e.target, type: e.type, weight: Number(e.weight.toFixed(4)) })
    }
  }
  edges.sort((a, b) => b.weight - a.weight || (a.source < b.source ? -1 : 1))

  const path = extractPath(gn, results, Math.max(2, hops + 1))

  // 6) 闭环：共召回 → usage 边
  let usage_bumped = 0
  if (opts.bumpUsage !== false && results.length > 1) {
    usage_bumped = gn.bumpUsage(results.map((r) => r.id))
  }

  return {
    query: q,
    k,
    alpha,
    hops,
    seeds: restart.size,
    results,
    subgraph: {
      nodes: results.map((r) => ({ id: r.id, activation: r.activation, move: r.event.move, verbal: r.verbal })),
      edges,
    },
    path,
    usage_bumped,
  }
}
