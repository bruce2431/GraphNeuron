/**
 * 认知社群：Leiden 划分记忆图 → 社群级图 → 高质量文本。
 *
 * 社群文本是给 LLM 直接读的：成员、时间跨度、关键词、核心记忆、内部关系（带关系类型
 * 与证据）、相邻社群与桥接词。不做纯 id 罗列。
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { leidenCommunities, type LeidenEdgeInput } from './leiden.ts'
import { buildEntityIndex } from './graph.ts'
import type { Community, MemEntry, MemGraph } from './types.ts'

export interface CommunityEdge {
  source: string
  target: string
  weight: number
  bridge: string
}

export interface CommunityResult {
  resolution: number
  modularity: number
  communities: Community[]
  edges: CommunityEdge[]
}

function fmtDate(t: number | null): string {
  if (!t) return '未知时间'
  const d = new Date(t)
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`
}

function short(title: string, n = 36): string {
  return title.length > n ? `${title.slice(0, n)}…` : title
}

export function detectCommunities(
  graph: MemGraph,
  entries: MemEntry[],
  opts: { resolution?: number; minSize?: number } = {},
): CommunityResult {
  const minSize = opts.minSize ?? 2
  const resolution = opts.resolution ?? 1.0
  const ids = graph.nodes.map((n) => n.id)
  const idx = new Map(ids.map((id, i) => [id, i]))
  const byId = new Map(entries.map((e) => [e.id, e]))

  // 同对节点的多类型边合并（Leiden 只吃加权无向边）
  const pair = new Map<string, number>()
  for (const e of graph.edges) {
    const a = idx.get(e.source)
    const b = idx.get(e.target)
    if (a === undefined || b === undefined || a === b) continue
    const k = a < b ? `${a}:${b}` : `${b}:${a}`
    pair.set(k, (pair.get(k) ?? 0) + e.weight)
  }
  const edges: LeidenEdgeInput[] = [...pair.entries()].map(([k, weight]) => {
    const [a, b] = k.split(':').map(Number)
    return { source: a!, target: b!, weight }
  })

  const res = leidenCommunities(ids.length, edges, { resolutions: [resolution] })
  const part = res.partitions[0]!
  const groups = new Map<number, string[]>()
  part.membership.forEach((cid, i) => {
    if (!groups.has(cid)) groups.set(cid, [])
    groups.get(cid)!.push(ids[i]!)
  })

  const entityIndex = buildEntityIndex(entries)
  const degree = new Map<string, number>()
  for (const e of graph.edges) {
    degree.set(e.source, (degree.get(e.source) ?? 0) + 1)
    degree.set(e.target, (degree.get(e.target) ?? 0) + 1)
  }

  const communities: Community[] = []
  const commOf = new Map<string, string>()
  let seq = 0
  for (const members of [...groups.values()].sort((a, b) => b.length - a.length)) {
    if (members.length < minSize) continue
    seq++
    const id = `C${String(seq).padStart(2, '0')}`
    for (const m of members) commOf.set(m, id)

    // 关键词：成员内按 idf 加权累计，排除路径类
    const kwScore = new Map<string, number>()
    for (const m of members) {
      for (const x of entityIndex.perEntry.get(m) ?? []) {
        if (x.includes('/') || x.length > 24) continue
        kwScore.set(x, (kwScore.get(x) ?? 0) + (entityIndex.idf.get(x) ?? 0))
      }
    }
    const keywords = [...kwScore.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([x]) => x)
      .filter((x, _i, arr) => !arr.some((y) => y !== x && y.includes(x) && (kwScore.get(y) ?? 0) >= (kwScore.get(x) ?? 0)))
      .slice(0, 5)

    const times = members.map((m) => byId.get(m)?.time).filter((t): t is number => !!t).sort((a, b) => a - b)
    const timeSpan: [string, string] | null = times.length
      ? [fmtDate(times[0]!), fmtDate(times[times.length - 1]!)]
      : null

    // 内部关系：取最强的前 6 条边
    const memberSet = new Set(members)
    const internal = graph.edges
      .filter((e) => memberSet.has(e.source) && memberSet.has(e.target) && e.type !== 'temporal')
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 6)

    const titleOf = (id: string) => short(byId.get(id)?.blocks[0] ?? id, 30)
    const core = [...members]
      .sort((a, b) => (degree.get(b) ?? 0) - (degree.get(a) ?? 0))
      .slice(0, 8)

    const lines: string[] = []
    lines.push(`社群 ${id}「${keywords.slice(0, 3).join(' · ') || '未命名'}」`)
    lines.push(`规模: ${members.length} 条记忆 | 时间: ${timeSpan ? `${timeSpan[0]} → ${timeSpan[1]}` : '未知'} | 模块度: ${part.modularity.toFixed(3)}`)
    lines.push(`关键词: ${keywords.join('、') || '（无）'}`)
    lines.push('')
    lines.push('【核心记忆】')
    for (const m of core) {
      const e = byId.get(m)
      lines.push(`- ${fmtDate(e?.time ?? null)} ${short(e?.blocks[0] ?? m, 60)}`)
    }
    lines.push('')
    lines.push('【内部关系】')
    if (internal.length) {
      for (const r of internal) {
        lines.push(`- 「${titleOf(r.source)}」 --${r.type}（${r.evidence}）--> 「${titleOf(r.target)}」`)
      }
    } else {
      lines.push('- （无强关系）')
    }

    communities.push({
      id,
      title: keywords.slice(0, 3).join(' · ') || `社群 ${id}`,
      size: members.length,
      modularity: Number(part.modularity.toFixed(4)),
      keywords,
      members,
      timeSpan,
      text: lines.join('\n'),
      neighbors: [],
    })
  }

  // 社群级图：跨社群边聚合
  const cross = new Map<string, { weight: number; bridge: Map<string, number>; evidence: string }>()
  for (const e of graph.edges) {
    const ca = commOf.get(e.source)
    const cb = commOf.get(e.target)
    if (!ca || !cb || ca === cb) continue
    const k = ca < cb ? `${ca}|${cb}` : `${cb}|${ca}`
    if (!cross.has(k)) cross.set(k, { weight: 0, bridge: new Map(), evidence: e.evidence })
    const c = cross.get(k)!
    c.weight += e.weight
    if (e.type === 'entity') {
      for (const w of e.evidence.replace('共享实体: ', '').split('、')) {
        c.bridge.set(w, (c.bridge.get(w) ?? 0) + 1)
      }
    }
  }

  const communityEdges: CommunityEdge[] = []
  for (const [k, c] of cross) {
    const [a, b] = k.split('|')
    const bridge = [...c.bridge.entries()].sort((x, y) => y[1] - x[1]).slice(0, 3).map(([w]) => w).join('、')
    communityEdges.push({
      source: a!,
      target: b!,
      weight: Number(c.weight.toFixed(3)),
      bridge: bridge || c.evidence,
    })
  }

  const byComm = new Map(communities.map((c) => [c.id, c]))
  for (const e of communityEdges) {
    byComm.get(e.source)?.neighbors.push({ id: e.target, weight: e.weight, bridge: e.bridge })
    byComm.get(e.target)?.neighbors.push({ id: e.source, weight: e.weight, bridge: e.bridge })
  }
  for (const c of communities) {
    if (!c.neighbors.length) continue
    c.text += '\n\n【相邻社群】'
    for (const n of c.neighbors.sort((a, b) => b.weight - a.weight).slice(0, 4)) {
      c.text += `\n- ${n.id}「${byComm.get(n.id)?.title ?? ''}」桥接: ${n.bridge}（权重 ${n.weight}）`
    }
  }

  return { resolution, modularity: Number(part.modularity.toFixed(4)), communities, edges: communityEdges }
}

export function saveCommunities(dataDir: string, result: CommunityResult): string {
  mkdirSync(dataDir, { recursive: true })
  const p = join(dataDir, 'communities.json')
  writeFileSync(p, JSON.stringify(result, null, 2))
  return p
}

export function loadCommunities(dataDir: string): CommunityResult | null {
  const p = join(dataDir, 'communities.json')
  if (!existsSync(p)) return null
  try {
    return JSON.parse(readFileSync(p, 'utf-8')) as CommunityResult
  } catch {
    return null
  }
}
