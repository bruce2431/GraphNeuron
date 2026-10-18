/** 读取 Pj11 neuron：l2.mem/mem.db + l1.cog/precog.db */

import { Database } from 'bun:sqlite'
import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import type { CoreFileItem, MemEntry, PrecogRecord } from './types.ts'

const MEM_ID_TIME = /^[A-Z]+\d*_MEM_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/
const VERIFY_ID_TIME = /_MEM_VERIFY_DB(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/

export function parseMemTime(id: string): number | null {
  const m = MEM_ID_TIME.exec(id) ?? VERIFY_ID_TIME.exec(id)
  if (!m) return null
  return new Date(
    Number(m[1]), Number(m[2]) - 1, Number(m[3]),
    Number(m[4]), Number(m[5]), Number(m[6]),
  ).getTime()
}

function jsonArr<T>(v: unknown): T[] {
  if (Array.isArray(v)) return v as T[]
  if (typeof v !== 'string' || !v) return []
  try {
    const x = JSON.parse(v)
    return Array.isArray(x) ? (x as T[]) : []
  } catch {
    return []
  }
}

/** 默认神经元路径：Pj11 项目根下（从本文件位置向上找）。 */
export function defaultNeuronPath(): string {
  let dir = import.meta.dir
  for (let i = 0; i < 8; i++) {
    const p = join(dir, '.claude', 'neturon', 'neurons', 'Neuron-Pj11-LoRAMoE的混合专家架构')
    if (existsSync(join(p, 'l2.mem', 'mem.db'))) return p
    dir = join(dir, '..')
  }
  throw new Error('找不到 Pj11 neuron（l2.mem/mem.db）')
}

export function loadMemories(neuronPath = defaultNeuronPath()): MemEntry[] {
  const db = new Database(join(neuronPath, 'l2.mem', 'mem.db'), { readonly: true })
  try {
    const rows = db.query(
      'SELECT memory_id, revelant, blocks, source, core_file, deprecated_by FROM memories ORDER BY memory_id',
    ).all() as Array<Record<string, unknown>>
    return rows.map((r) => {
      const blocks = jsonArr<string>(r.blocks)
      const revelant = jsonArr<string>(r.revelant)
      return {
        id: String(r.memory_id ?? ''),
        blocks,
        text: blocks.join(' '),
        source: String(r.source ?? ''),
        revelant,
        coreFiles: jsonArr<CoreFileItem>(r.core_file),
        deprecatedBy: r.deprecated_by ? String(r.deprecated_by) : null,
        time: parseMemTime(String(r.memory_id ?? '')),
      }
    })
  } finally {
    db.close()
  }
}

/** mem.db 中已有的块级向量（新写入条目），供 embedding 缓存缺失时兜底。 */
export function loadStoredVectors(neuronPath: string): Map<string, Float32Array> {
  const out = new Map<string, Float32Array>()
  const p = join(neuronPath, 'l2.mem', 'mem.db')
  if (!existsSync(p)) return out
  const db = new Database(p, { readonly: true })
  try {
    const rows = db.query('SELECT memory_id, block_vectors FROM memories').all() as Array<{
      memory_id: string
      block_vectors: Uint8Array | null
    }>
    for (const r of rows) {
      if (!r.block_vectors || r.block_vectors.byteLength === 0) continue
      const buf = r.block_vectors
      const copy = new Float32Array(buf.byteLength / 4)
      new Uint8Array(copy.buffer).set(buf)
      out.set(r.memory_id, copy)
    }
  } finally {
    db.close()
  }
  return out
}

/**
 * 李京瑾神经元（唯一测试库）：从 SubPj7 归档解出的 Neturon-李京瑾。
 * 权威源是 l2.mem/mem.db（memory_id / blocks / revelant / source）。
 */
export function defaultLjjNeuronPath(): string {
  let dir = import.meta.dir
  for (let i = 0; i < 8; i++) {
    const p = join(dir, 'data', 'ljj-neuron')
    if (existsSync(join(p, 'l2.mem', 'mem.db'))) return p
    dir = join(dir, '..')
  }
  throw new Error('找不到李京瑾神经元（data/ljj-neuron）')
}

export function defaultLjjPath(): string {
  return join(defaultLjjNeuronPath(), 'l2.mem', 'mem.db')
}

interface LjjRaw {
  memory_id: string
  men?: {
    content?: string
    source?: string
    revelant?: string[]
    core_file?: unknown[]
    time?: string
  }
  sem?: { summary?: string; blocks?: string[] }
}

function parseLjjTime(s: string | undefined): number | null {
  if (!s) return null
  const m = /^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})/.exec(s)
  if (!m) return null
  return new Date(
    Number(m[1]), Number(m[2]) - 1, Number(m[3]),
    Number(m[4]), Number(m[5]), Number(m[6]),
  ).getTime()
}

/**
 * 李京瑾记忆：只取 sem.blocks 作为检索文本；空块条目保留但不可检索。
 * 重复 memory_id 合并 blocks（原始数据有 7 组重复 id）。
 */
export function loadLjjMemories(file = defaultLjjPath()): MemEntry[] {
  if (file.endsWith('.db')) return loadLjjFromDb(file)
  const raw = JSON.parse(readFileSync(file, 'utf-8')) as LjjRaw[]
  const byId = new Map<string, MemEntry>()
  for (const r of raw) {
    const id = String(r.memory_id ?? '')
    if (!id) continue
    const blocks = (r.sem?.blocks ?? []).map((b) => String(b).trim()).filter(Boolean)
    const prev = byId.get(id)
    if (prev) {
      for (const b of blocks) if (!prev.blocks.includes(b)) prev.blocks.push(b)
      prev.text = prev.blocks.join(' ')
      continue
    }
    byId.set(id, {
      id,
      blocks,
      text: blocks.join(' '),
      source: String(r.men?.source ?? ''),
      revelant: (r.men?.revelant ?? []).map(String),
      coreFiles: [],
      deprecatedBy: null,
      time: parseLjjTime(r.men?.time),
    })
  }
  return [...byId.values()].sort((a, b) => a.id.localeCompare(b.id))
}

/** 直接读神经元权威库 mem.db（无 time 列，时间从 memory_id 解析）。 */
function loadLjjFromDb(file: string): MemEntry[] {
  const db = new Database(file, { readonly: true })
  try {
    const rows = db.query(
      'SELECT memory_id, revelant, blocks, source FROM memories ORDER BY memory_id',
    ).all() as Array<Record<string, unknown>>
    return rows.map((r) => {
      const id = String(r.memory_id ?? '')
      const blocks = jsonArr<string>(r.blocks).map((b) => String(b).trim()).filter(Boolean)
      return {
        id,
        blocks,
        text: blocks.join(' '),
        source: String(r.source ?? ''),
        revelant: jsonArr<string>(r.revelant).map(String),
        coreFiles: [],
        deprecatedBy: null,
        time: parseMemTime(id),
      }
    })
  } finally {
    db.close()
  }
}

/** 读 precog（只读；用于 usage 统计与认知标注） */
export function loadPrecog(neuronPath: string): PrecogRecord[] {
  const p = join(neuronPath, 'l1.cog', 'precog.db')
  if (!existsSync(p)) return []
  const db = new Database(p, { readonly: true })
  try {
    const rows = db.query('SELECT * FROM precog').all() as Array<Record<string, unknown>>
    return rows.map((r) => ({
      record_id: String(r.record_id ?? ''),
      query: String(r.query ?? ''),
      keywords: jsonArr<string>(r.keywords),
      top_k: Number(r.top_k ?? 0),
      results: jsonArr<{ id: string; accuracy: string; summary: string }>(r.results),
      description: String(r.description ?? ''),
    }))
  } finally {
    db.close()
  }
}

/** usage：每条记忆被真实检索（accuracy != false）命中的次数。 */
export function usageCounts(records: PrecogRecord[]): Map<string, number> {
  const m = new Map<string, number>()
  for (const r of records) {
    for (const x of r.results) {
      if (!x.id) continue
      if ((x.accuracy ?? '').toLowerCase() === 'false') continue
      m.set(x.id, (m.get(x.id) ?? 0) + 1)
    }
  }
  return m
}
