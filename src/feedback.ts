/**
 * 反哺循环：使用信号 → 结构诊断 → 派生动作计划（只诊断，不改原文）。
 *
 * 三级循环：
 *   1. 权重层（自动）  usage 边 / salience —— 由 usage.ts + graph.ts 完成
 *   2. 结构层（自动）  block 粒度诊断 —— 本文件，找出「多话题 mem / 噪锚点 /
 *      孤立被用 mem / 重复块 / 跨社群桥」
 *   3. 文本层（门控）  只对结构层判定的候选，用宿主 LLM 生成 derived 块，
 *      写 block_derived（append-only），A/B 校验不通过则回滚
 *
 * 关键判据：反馈的单位是 block 而不是整条 mem；没有 usage 信号的 mem 不进入
 * LLM 改写队列（避免盲改）。
 */

import { tokenizeWords } from './retrieval.ts'
import type { UsageSignal } from './usage.ts'
import type { Community, MemEntry, MemGraph } from './types.ts'

export type CandidateKind =
  | 'multi_topic'
  | 'noisy_anchor'
  | 'orphan_used'
  | 'duplicate_block'
  | 'bridge'

export interface FeedbackOptions {
  dataset?: string
  /** 正向命中（true+revelant）达到此数才进入结构诊断 */
  minHits?: number
  /** 块间 cos 低于此值 → 视为不同话题 */
  multiTopicCos?: number
  /** 跨 mem 块 cos 高于此值 → 重复块 */
  duplicateCos?: number
  /** false 命中占比阈值（falseHits ≥ 2 才判） */
  noisyFalseRatio?: number
  /** 图中度数 ≤ 此值且被使用 → 孤立 */
  orphanDegree?: number
  /** 每个类别最多输出多少条 */
  topPerKind?: number
}

export interface FeedbackCandidate {
  kind: CandidateKind
  memoryId: string
  score: number
  action: 'derived_split' | 'anchor_rewrite' | 'edge_attach' | 'merge_duplicate' | 'keep_bridge'
  reason: string
  evidence: Record<string, unknown>
}

export interface FeedbackReport {
  generatedAt: string
  dataset: string
  stats: {
    memories: number
    blocks: number
    usedMemories: number
    usage: Record<string, unknown>
    multiTopic: number
    noisyAnchor: number
    orphanUsed: number
    duplicateBlockPairs: number
    bridges: number
  }
  candidates: FeedbackCandidate[]
  plan: Array<{ action: FeedbackCandidate['action']; count: number; note: string }>
}

function dot(a: Float32Array, b: Float32Array): number {
  let s = 0
  const n = Math.min(a.length, b.length)
  for (let i = 0; i < n; i++) s += a[i]! * b[i]!
  return s
}

function short(s: string, n = 48): string {
  return s.length > n ? `${s.slice(0, n)}…` : s
}

/** 块对查询的词覆盖（块里命中查询词的比例），用于把 mem 级 usage 归因到块。 */
function blockQueryOverlap(block: string, queries: string[]): number {
  const q = new Set<string>()
  for (const x of queries) for (const t of tokenizeWords(x)) q.add(t)
  if (!q.size) return 0
  const b = new Set(tokenizeWords(block))
  let hit = 0
  for (const t of q) if (b.has(t)) hit++
  return hit / q.size
}

/** 贪心聚类（complete linkage）：与组内所有成员 cos ≥ threshold 才并入，避免链式合并掩盖多话题。 */
function clusterBlocks(vectors: Float32Array[], cos: (a: Float32Array, b: Float32Array) => number, threshold: number): number[][] {
  const groups: number[][] = []
  for (let i = 0; i < vectors.length; i++) {
    let placed = false
    for (const g of groups) {
      if (g.every((j) => cos(vectors[i]!, vectors[j]!) >= threshold)) {
        g.push(i)
        placed = true
        break
      }
    }
    if (!placed) groups.push([i])
  }
  return groups
}

function minPairCos(vectors: Float32Array[], cos: (a: Float32Array, b: Float32Array) => number): number {
  let min = 1
  for (let i = 0; i < vectors.length; i++) {
    for (let j = i + 1; j < vectors.length; j++) min = Math.min(min, cos(vectors[i]!, vectors[j]!))
  }
  return min
}

export function diagnoseFeedback(
  entries: MemEntry[],
  vectors: Map<string, Float32Array[]>,
  graph: MemGraph,
  communities: Community[],
  usage: UsageSignal,
  opts: FeedbackOptions = {},
): FeedbackReport {
  const minHits = opts.minHits ?? 2
  const multiTopicCos = opts.multiTopicCos ?? 0.55
  const duplicateCos = opts.duplicateCos ?? 0.92
  const noisyFalseRatio = opts.noisyFalseRatio ?? 0.5
  const orphanDegree = opts.orphanDegree ?? 0
  const topPerKind = opts.topPerKind ?? 20

  const byId = new Map(entries.map((e) => [e.id, e]))
  const degree = new Map<string, number>()
  for (const e of graph.edges) {
    degree.set(e.source, (degree.get(e.source) ?? 0) + 1)
    degree.set(e.target, (degree.get(e.target) ?? 0) + 1)
  }
  const commOf = new Map<string, string>()
  for (const c of communities) for (const m of c.members) commOf.set(m, c.id)

  const multi: FeedbackCandidate[] = []
  const noisy: FeedbackCandidate[] = []
  const orphan: FeedbackCandidate[] = []
  const duplicate: FeedbackCandidate[] = []

  const used = [...usage.stats.values()].filter((s) => s.trueHits + s.relevantHits >= minHits)
  // 重复块只在「被使用过的 mem」之间比对，避免 1.3 万块全量 O(n²)
  const usedBlocks: Array<{ id: string; index: number; text: string; vec: Float32Array; weighted: number }> = []

  for (const s of used) {
    const e = byId.get(s.id)
    if (!e || !e.blocks.length) continue
    const vs = vectors.get(s.id) ?? []
    const deg = degree.get(s.id) ?? 0

    if (deg <= orphanDegree) {
      orphan.push({
        kind: 'orphan_used',
        memoryId: s.id,
        score: Number(s.weighted.toFixed(3)),
        action: 'edge_attach',
        reason: `被检索 ${s.trueHits + s.relevantHits} 次但图中度数为 ${deg}`,
        evidence: { queries: s.queries, anchor: short(e.blocks[0] ?? '', 60) },
      })
    }

    const pos = s.trueHits + s.relevantHits
    const total = pos + s.falseHits
    if (s.falseHits >= 2 && total > 0 && s.falseHits / total >= noisyFalseRatio) {
      noisy.push({
        kind: 'noisy_anchor',
        memoryId: s.id,
        score: Number((s.weighted * (s.falseHits / total)).toFixed(3)),
        action: 'anchor_rewrite',
        reason: `false 命中 ${s.falseHits}/${total}，检索锚点（block[0]）与查询意图不匹配`,
        evidence: { queries: s.queries, anchor: short(e.blocks[0] ?? '', 60) },
      })
    }

    for (let i = 0; i < e.blocks.length; i++) {
      const v = vs[i]
      if (v?.length) usedBlocks.push({ id: s.id, index: i, text: e.blocks[i]!, vec: v, weighted: s.weighted })
    }

    if (e.blocks.length >= 2 && vs.length === e.blocks.length && vs.every((v) => v.length)) {
      const minCos = minPairCos(vs, dot)
      if (minCos < multiTopicCos) {
        const weights = e.blocks.map((b) => blockQueryOverlap(b, s.queries))
        const groups = clusterBlocks(vs, dot, multiTopicCos)
        multi.push({
          kind: 'multi_topic',
          memoryId: s.id,
          score: Number((s.weighted * (1 - minCos) * (1 + 0.1 * (groups.length - 1))).toFixed(3)),
          action: 'derived_split',
          reason: `块间最低 cos=${minCos.toFixed(3)}（<${multiTopicCos}），${groups.length} 个话题簇；命中查询指向块 ${weights.indexOf(Math.max(...weights))}`,
          evidence: {
            queries: s.queries,
            groups: groups.map((g) => g.map((i) => ({ index: i, text: short(e.blocks[i]!, 40), overlap: Number(weights[i]!.toFixed(2)) }))),
          },
        })
      }
    }
  }

  // 跨 mem 的重复块（只查被用过的块）
  for (let i = 0; i < usedBlocks.length; i++) {
    for (let j = i + 1; j < usedBlocks.length; j++) {
      const a = usedBlocks[i]!
      const b = usedBlocks[j]!
      if (a.id === b.id) continue
      const c = dot(a.vec, b.vec)
      if (c < duplicateCos) continue
      duplicate.push({
        kind: 'duplicate_block',
        memoryId: a.id,
        score: Number((c * Math.min(a.weighted, b.weighted)).toFixed(3)),
        action: 'merge_duplicate',
        reason: `块与 ${b.id}#${b.index} cos=${c.toFixed(3)}`,
        evidence: { peer: b.id, peerIndex: b.index, a: short(a.text, 50), b: short(b.text, 50) },
      })
    }
  }

  // 跨社群 usage 桥（正向信号，保留并在社群文本中显式呈现）
  const bridges: FeedbackCandidate[] = []
  for (const u of usage.edges) {
    const ca = commOf.get(u.source)
    const cb = commOf.get(u.target)
    if (!ca || !cb || ca === cb) continue
    bridges.push({
      kind: 'bridge',
      memoryId: u.source,
      score: u.weight,
      action: 'keep_bridge',
      reason: `${ca} ↔ ${cb} 的共召回桥（${u.hits} 次）`,
      evidence: { peer: u.target, communities: [ca, cb], queries: u.evidence },
    })
  }

  const take = <T extends { score: number }>(xs: T[]) =>
    xs.sort((a, b) => b.score - a.score).slice(0, topPerKind)
  const candidates = [
    ...take(multi),
    ...take(noisy),
    ...take(orphan),
    ...take(duplicate),
    ...take(bridges),
  ]

  return {
    generatedAt: new Date().toISOString(),
    dataset: opts.dataset ?? '',
    stats: {
      memories: entries.filter((e) => !e.deprecatedBy).length,
      blocks: entries.reduce((s, e) => s + e.blocks.length, 0),
      usedMemories: usage.usedMemories,
      usage: {
        records: usage.records,
        trueHits: sumStat(usage, 'trueHits'),
        relevantHits: sumStat(usage, 'relevantHits'),
        falseHits: sumStat(usage, 'falseHits'),
        coHitEdges: usage.edges.length,
      },
      multiTopic: multi.length,
      noisyAnchor: noisy.length,
      orphanUsed: orphan.length,
      duplicateBlockPairs: duplicate.length,
      bridges: bridges.length,
    },
    candidates,
    plan: [
      { action: 'derived_split', count: multi.length, note: '多话题 mem → 生成 derived 块（append-only，需 A/B 通过）' },
      { action: 'anchor_rewrite', count: noisy.length, note: '噪锚点 → 重写 block[0] 为查询形自然句（derived）' },
      { action: 'edge_attach', count: orphan.length, note: '被用但孤立 → 检查实体/时序连边' },
      { action: 'merge_duplicate', count: duplicate.length, note: '重复块 → 合并候选（先校验再动）' },
      { action: 'keep_bridge', count: bridges.length, note: '跨社群桥 → 保留 usage 边并写进社群文本' },
    ],
  }
}

function sumStat(usage: UsageSignal, key: 'trueHits' | 'relevantHits' | 'falseHits'): number {
  let n = 0
  for (const s of usage.stats.values()) n += s[key]
  return n
}

export function formatFeedback(r: FeedbackReport): string {
  const lines: string[] = []
  lines.push(`反哺诊断  ${r.generatedAt}`)
  lines.push(`记忆 ${r.stats.memories}（有文本块）| 块 ${r.stats.blocks} | usage：${r.stats.usage.records} 次检索，` +
    `true ${r.stats.usage.trueHits} / revelant ${r.stats.usage.relevantHits} / false ${r.stats.usage.falseHits}，` +
    `共召回边 ${r.stats.usage.coHitEdges}`)
  lines.push(`候选：多话题 ${r.stats.multiTopic} | 噪锚点 ${r.stats.noisyAnchor} | 孤立被用 ${r.stats.orphanUsed} | 重复块 ${r.stats.duplicateBlockPairs} | 跨社群桥 ${r.stats.bridges}`)
  const byKind = new Map<CandidateKind, FeedbackCandidate[]>()
  for (const c of r.candidates) {
    const arr = byKind.get(c.kind) ?? []
    arr.push(c)
    byKind.set(c.kind, arr)
  }
  for (const [kind, xs] of byKind) {
    lines.push('')
    lines.push(`【${kind}】${xs.length}`)
    for (const c of xs.slice(0, 8)) lines.push(`- ${c.memoryId}  score=${c.score}  ${c.reason}`)
  }
  return lines.join('\n')
}
