/**
 * GraphNeuron — 图存储与结构边构建
 *
 * 记忆 = 事件的图：事件节点 + 类型化边（carries/causes/cooc/char/temporal/spatial/usage）。
 * usage 边由检索共召回累加（Hebbian），是「认知因使用而聚合」的落地。
 */

import { readFileSync, writeFileSync } from 'node:fs'
import {
  type Actor,
  type EdgeType,
  type EventInput,
  type GEdge,
  type GEvent,
  makeEventId,
  normalizeEvent,
  parseTime,
} from './schema.ts'
import { type Embedder, LocalEmbedder, eventVector } from './embed.ts'

export interface GNOptions {
  /** 时间邻近窗口（ms），默认 6 小时 */
  temporalWindowMs?: number
  /** 时间衰减常数 τ（ms），默认 30 分钟 */
  temporalTauMs?: number
  /** 各边基础权重 */
  coocWeight?: number
  charWeight?: number
  carriesWeight?: number
  causesWeight?: number
  temporalWeight?: number
  spatialWeight?: number
  /** 单个角色/地点最多连接的同伴数（防 O(n²) 爆炸），默认 50 */
  groupCap?: number
  /** 每次共召回 usage 增量 */
  usageDelta?: number
  /** usage 边每次重建时的衰减系数（0~1，1=不衰减） */
  usageDecay?: number
}

const DEFAULTS: Required<GNOptions> = {
  temporalWindowMs: 6 * 3600 * 1000,
  temporalTauMs: 30 * 60 * 1000,
  coocWeight: 0.6,
  charWeight: 1.0,
  carriesWeight: 1.0,
  causesWeight: 1.2,
  temporalWeight: 0.5,
  spatialWeight: 0.8,
  groupCap: 50,
  usageDelta: 0.15,
  usageDecay: 1.0,
}

function edgeKey(type: EdgeType, a: string, b: string): string {
  return a < b ? `${type}|${a}|${b}` : `${type}|${b}|${a}`
}

/** 从 m_YYYYMMDDHHMMSS_xxx 形式的 id 中解出 epoch ms */
export function timeFromId(id: string): number | null {
  const m = /m_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})/.exec(id)
  if (!m) return null
  const ms = new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]).getTime()
  return Number.isFinite(ms) ? ms : null
}

export class GraphNeuron {
  events = new Map<string, GEvent>()
  edges = new Map<string, GEdge>()
  actors = new Map<string, Actor>()
  private vectors = new Map<string, number[]>()
  embedder: Embedder
  opts: Required<GNOptions>

  constructor(opts: GNOptions = {}, embedder: Embedder = new LocalEmbedder()) {
    this.opts = { ...DEFAULTS, ...opts }
    this.embedder = embedder
  }

  get size(): number {
    return this.events.size
  }

  /** 添加事件（content 为对象时递归建子事件并连 carries 边）。返回父事件。 */
  addEvent(input: EventInput): GEvent {
    if (!input || typeof input.move !== 'string' || !input.move.trim()) {
      throw new Error('addEvent: move 必填')
    }
    const id = input.id ? String(input.id) : undefined
    let time = parseTime(input.time)
    if (time === null && id) time = timeFromId(id)
    if (time === null) time = Date.now()
    const finalId = id ?? makeEventId(time)
    const ev = normalizeEvent(input, finalId, time)

    // 递归处理嵌套 content
    if (input.content && typeof input.content === 'object') {
      const child = this.addEvent(input.content as EventInput)
      ev.content_event = child.id
      ev.content_literal = null
    }
    this.events.set(ev.id, ev)
    this.vectors.delete(ev.id)

    // 登记角色/地点
    for (const c of [...ev.chara1, ...ev.chara2]) this.ensureActor(c)
    if (ev.location) this.ensureActor(ev.location, 'place')
    return ev
  }

  private ensureActor(label: string, kind: Actor['kind'] = 'person'): Actor {
    const id = `actor:${label}`
    let a = this.actors.get(id)
    if (!a) {
      a = { id, kind, label }
      this.actors.set(id, a)
    }
    return a
  }

  getEvent(id: string): GEvent | undefined {
    return this.events.get(id)
  }

  vectorOf(id: string): number[] | undefined {
    const ev = this.events.get(id)
    if (!ev) return undefined
    let v = this.vectors.get(id)
    if (!v) {
      v = eventVector(ev, this.embedder.dim, (cid) => this.events.get(cid))
      this.vectors.set(id, v)
    }
    return v
  }

  edgesOf(id: string): GEdge[] {
    const out: GEdge[] = []
    for (const e of this.edges.values()) {
      if (e.source === id || e.target === id) out.push(e)
    }
    return out
  }

  /** usage 边保留，结构边全量重建（派生，确定性） */
  buildEdges(): Record<string, number> {
    const o = this.opts
    const keptUsage = new Map<string, GEdge>()
    for (const [k, e] of this.edges) {
      if (e.type === 'usage') keptUsage.set(k, e)
    }
    this.edges = new Map()

    const put = (a: string, b: string, type: EdgeType, w: number) => {
      if (a === b || w <= 0 || !this.events.has(a) || !this.events.has(b)) return
      const k = edgeKey(type, a, b)
      const prev = this.edges.get(k)
      if (prev) prev.weight += w
      else this.edges.set(k, { source: a < b ? a : b, target: a < b ? b : a, type, weight: w })
    }

    const counts: Record<string, number> = { carries: 0, causes: 0, cooc: 0, char: 0, temporal: 0, spatial: 0 }

    // carries / causes
    for (const ev of this.events.values()) {
      if (ev.content_event) {
        put(ev.id, ev.content_event, 'carries', o.carriesWeight)
        counts.carries++
      }
      for (const cid of ev.causes) {
        put(ev.id, cid, 'causes', o.causesWeight)
        counts.causes++
      }
    }

    // cooc：同源内按时间相邻连边（避免同源全连接爆炸）
    const bySource = new Map<string, GEvent[]>()
    for (const ev of this.events.values()) {
      if (!ev.source) continue
      const arr = bySource.get(ev.source) ?? []
      arr.push(ev)
      bySource.set(ev.source, arr)
    }
    for (const arr of bySource.values()) {
      arr.sort((a, b) => (a.time ?? 0) - (b.time ?? 0) || (a.id < b.id ? -1 : 1))
      for (let i = 1; i < arr.length; i++) {
        put(arr[i - 1].id, arr[i].id, 'cooc', o.coocWeight)
        counts.cooc++
      }
    }

    // char：共享角色
    const byChar = new Map<string, string[]>()
    for (const ev of this.events.values()) {
      for (const c of new Set([...ev.chara1, ...ev.chara2])) {
        const arr = byChar.get(c) ?? []
        arr.push(ev.id)
        byChar.set(c, arr)
      }
    }
    for (const ids of byChar.values()) {
      if (ids.length < 2) continue
      const take = ids.length > o.groupCap ? ids.slice(0, o.groupCap) : ids
      for (let i = 0; i < take.length; i++) {
        for (let j = i + 1; j < take.length; j++) {
          put(take[i], take[j], 'char', o.charWeight)
          counts.char++
        }
      }
    }

    // temporal：全表按时间排序，窗口内相邻连边，权重 exp(−Δt/τ)
    const timed = [...this.events.values()].filter((e) => e.time != null).sort((a, b) => a.time! - b.time!)
    for (let i = 1; i < timed.length; i++) {
      const dt = timed[i].time! - timed[i - 1].time!
      if (dt <= o.temporalWindowMs) {
        const w = o.temporalWeight * Math.exp(-dt / o.temporalTauMs)
        put(timed[i - 1].id, timed[i].id, 'temporal', w)
        counts.temporal++
      }
    }

    // spatial：同地点
    const byLoc = new Map<string, string[]>()
    for (const ev of this.events.values()) {
      if (!ev.location) continue
      const arr = byLoc.get(ev.location) ?? []
      arr.push(ev.id)
      byLoc.set(ev.location, arr)
    }
    for (const ids of byLoc.values()) {
      const take = ids.length > o.groupCap ? ids.slice(0, o.groupCap) : ids
      for (let i = 0; i < take.length; i++) {
        for (let j = i + 1; j < take.length; j++) {
          put(take[i], take[j], 'spatial', o.spatialWeight)
          counts.spatial++
        }
      }
    }

    // 回填 usage（保序）
    for (const [k, e] of keptUsage) {
      const w = e.weight * o.usageDecay
      if (w > 0 && this.events.has(e.source) && this.events.has(e.target)) {
        this.edges.set(k, { ...e, weight: w })
      }
    }
    counts.usage = [...this.edges.values()].filter((e) => e.type === 'usage').length
    return counts
  }

  /** 共召回 → usage 边累加（Hebbian 闭环） */
  bumpUsage(ids: string[], delta?: number): number {
    const d = delta ?? this.opts.usageDelta
    if (ids.length < 2) return 0
    let n = 0
    for (let i = 0; i < ids.length; i++) {
      for (let j = i + 1; j < ids.length; j++) {
        const k = edgeKey('usage', ids[i], ids[j])
        const prev = this.edges.get(k)
        if (prev) prev.weight += d
        else {
          this.edges.set(k, {
            source: ids[i] < ids[j] ? ids[i] : ids[j],
            target: ids[i] < ids[j] ? ids[j] : ids[i],
            type: 'usage',
            weight: d,
          })
        }
        n++
      }
    }
    return n
  }

  /** 邻接表：id → [{nbr, weight, type}]（无向） */
  adjacency(): Map<string, { nbr: string; weight: number; type: EdgeType }[]> {
    const adj = new Map<string, { nbr: string; weight: number; type: EdgeType }[]>()
    for (const id of this.events.keys()) adj.set(id, [])
    for (const e of this.edges.values()) {
      if (!adj.has(e.source) || !adj.has(e.target)) continue
      adj.get(e.source)!.push({ nbr: e.target, weight: e.weight, type: e.type })
      adj.get(e.target)!.push({ nbr: e.source, weight: e.weight, type: e.type })
    }
    return adj
  }

  snapshot() {
    const byType: Record<string, number> = {}
    for (const e of this.edges.values()) byType[e.type] = (byType[e.type] ?? 0) + 1
    return {
      stats: {
        events: this.events.size,
        actors: this.actors.size,
        edges: this.edges.size,
        edgesByType: byType,
      },
      events: [...this.events.values()],
      edges: [...this.edges.values()],
      actors: [...this.actors.values()],
    }
  }

  save(path: string): void {
    writeFileSync(path, JSON.stringify(this.snapshot(), null, 2), 'utf8')
  }

  static load(path: string, opts: GNOptions = {}, embedder?: Embedder): GraphNeuron {
    const raw = JSON.parse(readFileSync(path, 'utf8'))
    const gn = new GraphNeuron(opts, embedder)
    for (const ev of raw.events ?? []) gn.events.set(ev.id, ev)
    for (const a of raw.actors ?? []) gn.actors.set(a.id, a)
    for (const e of raw.edges ?? []) gn.edges.set(edgeKey(e.type, e.source, e.target), e)
    for (const ev of gn.events.values()) {
      for (const c of [...ev.chara1, ...ev.chara2]) gn.ensureActor(c)
      if (ev.location) gn.ensureActor(ev.location, 'place')
    }
    return gn
  }
}
