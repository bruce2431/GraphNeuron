/** MemGraph 原型 — 公共类型 */

export interface CoreFileItem {
  name: string
  path?: string
  content?: string
}

/** 一条记忆（mem.db 行 + 派生字段） */
export interface MemEntry {
  id: string
  blocks: string[]
  text: string
  source: string
  revelant: string[]
  coreFiles: CoreFileItem[]
  deprecatedBy: string | null
  /** 从 memory_id 解析出的毫秒时间；解析失败为 null */
  time: number | null
}

export type EdgeType = 'reference' | 'entity' | 'semantic' | 'temporal' | 'usage'

/** 记忆图边。evidence 是给 LLM 读的关系说明，必须非空。 */
export interface MemGraphEdge {
  source: string
  target: string
  type: EdgeType
  weight: number
  evidence: string
}

export interface MemGraph {
  nodes: Array<{ id: string; title: string; time: number | null }>
  edges: MemGraphEdge[]
}

export interface RetrievedMemory {
  id: string
  score: number
  cos: number
  bm25: number
  kw: number
  usage: number
  recency: number
  title: string
  blocks: string[]
}

export interface RankedResult {
  memory_id: string
  score: number
  cos: number
  bm25: number
  kw: number
  usage: number
  recency: number
  title: string
  blocks: string[]
}

/** 社群（Leiden 划分 + LLM 可读的高质量文本） */
export interface Community {
  id: string
  title: string
  size: number
  modularity: number
  keywords: string[]
  members: string[]
  timeSpan: [string, string] | null
  /** 给 LLM 直接读的完整文本 */
  text: string
  /** 相邻社群（社群级图的边） */
  neighbors: Array<{ id: string; weight: number; bridge: string }>
}

export interface PrecogRecord {
  record_id: string
  query: string
  keywords: string[]
  top_k: number
  results: Array<{ id: string; accuracy: string; summary: string }>
  description: string
}
