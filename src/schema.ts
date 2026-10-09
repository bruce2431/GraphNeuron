/**
 * GraphNeuron — 事件模板（schema）
 *
 * 事件 m = ⟨ Time , Chara₁ , Move , Chara₂ , Content ⟩  （五维）
 *   附加正交限定符：pol（极性/否定）、mod（情态/时体）、chara_mode（集合读法）
 *
 * 设计要点（见 SubPj9 [20261009130500]-GraphNeuron设计-v0.2.md）：
 *   - Move 是「单个动作原子」，**不带否定/时态**；否定是事件的算子，记在 pol 上。
 *   - Content 是内容槽：字面量（payload）或嵌套事件（携带 m₁(m₂)）。
 *   - 时间由 id 编码（m_YYYYMMDDHHMMSS…），time 字段可空。
 */

export type Polarity = '+' | '-'
export type CharaMode = 'unspecified' | 'collective' | 'distributive'

/** 外部输入的事件（宽松；content 可嵌套事件） */
export interface EventInput {
  id?: string
  /** epoch ms 或 "YYYYMMDDHHMMSS" 或 ISO 串；缺省则由 id / now 派生 */
  time?: number | string
  chara1?: string[] | string
  move: string
  chara2?: string[] | string
  /** 内容槽：字面量串，或嵌套事件（携带） */
  content?: string | EventInput
  /** 极性：默认 '+'；否定写 '-' */
  pol?: Polarity
  /** 情态/时体：realis / irrealis / past / habitual …（自由标签，可为空） */
  mod?: string
  chara_mode?: CharaMode
  /** 来源记忆/会话 id */
  source?: string
  location?: string
  /** 显式因果：本事件导致的事件 id 列表 */
  causes?: string[]
  /** 溯源痕迹（管线填充）：本事件来自原始记录的哪一层级/片段 */
  trace?: TraceStep[]
  meta?: Record<string, unknown>
}

/** 内部规范化后的事件节点 */
export interface GEvent {
  id: string
  time: number | null
  chara1: string[]
  move: string
  chara2: string[]
  pol: Polarity
  mod: string | null
  chara_mode: CharaMode
  /** 内容槽的两种取值之一 */
  content_literal: string | null
  content_event: string | null
  source: string | null
  location: string | null
  causes: string[]
  /** 溯源痕迹：该事件在其原始记录里的层级/片段路径（管线填充，可空） */
  trace: TraceStep[]
  meta: Record<string, unknown>
}

/** 溯源片段：一条记录被切分后，本事件来自哪一层哪一片 */
export interface TraceStep {
  /** 层级（0=最外层事件；嵌套每深一层 +1） */
  level?: number
  /** 片段序号/角色标签（如 chunk 索引、说话人轮次） */
  segment?: string
  /** 该片段的原文文本 */
  text?: string
  /** 自由标签 */
  tag?: string
}

/** 图边类型 */
export type EdgeType = 'carries' | 'causes' | 'cooc' | 'char' | 'temporal' | 'spatial' | 'usage'

export interface GEdge {
  source: string
  target: string
  type: EdgeType
  weight: number
}

/** 人/地点等角色节点（用于 char/spatial 边的物化，不参与蔓延） */
export interface Actor {
  id: string
  kind: 'person' | 'place' | 'org' | 'other'
  label: string
}

let seq = 0

function pad(n: number, w = 2): string {
  return String(n).padStart(w, '0')
}

/** 由时间生成 id：m_YYYYMMDDHHMMSS_<seq> */
export function makeEventId(time: number | null): string {
  const d = time == null ? new Date() : new Date(time)
  const stamp =
    `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}` +
    `${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`
  seq = (seq + 1) % 1000
  return `m_${stamp}_${pad(seq, 3)}`
}

/** 解析时间：number(epoch ms) | "YYYYMMDDHHMMSS" | ISO | 缺省 now */
export function parseTime(t: number | string | undefined | null): number | null {
  if (t === undefined || t === null) return null
  if (typeof t === 'number') return Number.isFinite(t) ? t : null
  const s = String(t).trim()
  if (/^\d{14}$/.test(s)) {
    const y = +s.slice(0, 4), mo = +s.slice(4, 6), d = +s.slice(6, 8)
    const h = +s.slice(8, 10), mi = +s.slice(10, 12), se = +s.slice(12, 14)
    const ms = new Date(y, mo - 1, d, h, mi, se).getTime()
    return Number.isFinite(ms) ? ms : null
  }
  if (/^\d+$/.test(s)) return Number(s)
  const ms = Date.parse(s)
  return Number.isFinite(ms) ? ms : null
}

export function asArray(v: string[] | string | undefined | null): string[] {
  if (v === undefined || v === null) return []
  if (Array.isArray(v)) return v.map((x) => String(x).trim()).filter(Boolean)
  return String(v)
    .split(/[,，、;；\s]+/)
    .map((x) => x.trim())
    .filter(Boolean)
}

/** EventInput → GEvent（嵌套 content 事件由 store 递归先建后引用；此处只处理字面量） */
export function normalizeEvent(input: EventInput, id: string, time: number | null): GEvent {
  const contentIsLiteral = typeof input.content === 'string' ? input.content : null
  return {
    id,
    time,
    chara1: asArray(input.chara1),
    move: String(input.move ?? '').trim(),
    chara2: asArray(input.chara2),
    pol: input.pol === '-' ? '-' : '+',
    mod: input.mod ? String(input.mod) : null,
    chara_mode: input.chara_mode ?? 'unspecified',
    content_literal: contentIsLiteral,
    content_event: null,
    source: input.source ? String(input.source) : null,
    location: input.location ? String(input.location) : null,
    causes: Array.isArray(input.causes) ? [...input.causes] : [],
    trace: Array.isArray(input.trace) ? input.trace.map((t) => ({ ...t })) : [],
    meta: input.meta && typeof input.meta === 'object' ? { ...input.meta } : {},
  }
}

/**
 * 事件的口语化文本（用于 embedding / 展示）。
 * 否定以「没」前缀呈现；嵌套以「说/携带」体现（简化：{{{child}}} 占位）。
 */
export function verbalize(e: GEvent, resolveChild?: (id: string) => GEvent | undefined): string {
  const a1 = e.chara1.join('')
  const a2 = e.chara2.join('')
  const neg = e.pol === '-' ? '没' : ''
  const mod = e.mod ? `${e.mod}` : ''
  let core = `${a1}${neg}${mod}${e.move}${a2}`
  if (e.location) core += `@${e.location}`
  if (e.content_literal) core += `：${e.content_literal}`
  if (e.content_event) {
    const child = resolveChild?.(e.content_event)
    core += `（携带 ${child ? verbalize(child, resolveChild) : e.content_event}）`
  }
  return core
}
