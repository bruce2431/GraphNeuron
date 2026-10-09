/**
 * GraphNeuron — 社群检测（Leiden 多分辨率）
 *
 * 在事件事件图上跑 Leiden：社群 = 多张记忆图交叠后的「完整记录」，对应 cog2 概念层。
 * 节点 = 事件 id；边 = 全部事件间边（可加权，usage 边按权重计入）。
 */

import { type GEdge, verbalize } from './schema.ts'
import type { GraphNeuron } from './store.ts'
import { type LeidenEdgeInput, leidenCommunities } from './leiden.ts'

export interface CommunityMember {
  id: string
  move: string
  verbal: string
  degree: number
}

export interface Community {
  id: string
  size: number
  density: number
  members: CommunityMember[]
}

export interface CommunityResult {
  resolution: number
  nCommunities: number
  modularity: number
  communities: Community[]
}

export interface CommunityOptions {
  resolutions?: number[]
  seed?: number
  /** 只统计规模 ≥ 此值的社群进入输出（默认 2） */
  minSize?: number
  /** 边类型过滤：默认全部 */
  edgeTypes?: GEdge['type'][]
}

export function detectCommunities(gn: GraphNeuron, opts: CommunityOptions = {}): CommunityResult[] {
  const ids = [...gn.events.keys()].sort()
  const idx = new Map(ids.map((id, i) => [id, i]))
  const allow = opts.edgeTypes ? new Set(opts.edgeTypes) : null

  const edges: LeidenEdgeInput[] = []
  for (const e of gn.edges.values()) {
    if (allow && !allow.has(e.type)) continue
    const a = idx.get(e.source)
    const b = idx.get(e.target)
    if (a === undefined || b === undefined) continue
    edges.push({ source: a, target: b, weight: e.weight })
  }

  const res = leidenCommunities(ids.length, edges, {
    resolutions: opts.resolutions,
    seed: opts.seed,
  })

  const minSize = opts.minSize ?? 2
  return res.partitions.map((p) => {
    const groups = new Map<number, number[]>()
    for (let i = 0; i < p.membership.length; i++) {
      const c = p.membership[i]
      const arr = groups.get(c) ?? []
      arr.push(i)
      groups.set(c, arr)
    }
    const communities: Community[] = []
    for (const [c, members] of groups) {
      if (members.length < minSize) continue
      const set = new Set(members)
      const pairs = new Set<number>()
      const n = ids.length
      for (const e of edges) {
        if (set.has(e.source) && set.has(e.target)) {
          const a = Math.min(e.source, e.target)
          const b = Math.max(e.source, e.target)
          pairs.add(a * n + b)
        }
      }
      const internalEdges = pairs.size
      const maxEdges = (members.length * (members.length - 1)) / 2
      const membersOut: CommunityMember[] = members.map((i) => {
        const ev = gn.events.get(ids[i])!
        const degree = gn.edgesOf(ev.id).length
        return { id: ev.id, move: ev.move, verbal: verbalize(ev, (cid) => gn.events.get(cid)), degree }
      })
      communities.push({
        id: `COMM_${c}`,
        size: members.length,
        density: maxEdges > 0 ? Number((internalEdges / maxEdges).toFixed(4)) : 0,
        members: membersOut,
      })
    }
    communities.sort((a, b) => b.size - a.size || (a.id < b.id ? -1 : 1))
    return {
      resolution: p.resolution,
      nCommunities: communities.length,
      modularity: Number(p.modularity.toFixed(4)),
      communities,
    }
  })
}
