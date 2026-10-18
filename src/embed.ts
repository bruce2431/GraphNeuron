/**
 * BGE 嵌入适配。
 *
 * 原型复用 Pj16 Floria 现网 embedder（只读 import，不修改 Pj16 文件）：
 * 同一模型 bge-small-zh-v1.5 / CLS 池化 / L2 归一，保证与活引擎口径一致。
 * Pj16 不可用时降级为 512 维 feature-hash（仅保证可跑，不用于效果结论）。
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import type { MemEntry } from './types.ts'

const DIM = 512

function findModelCache(): string {
  let dir = import.meta.dir
  for (let i = 0; i < 10; i++) {
    const p = join(dir, '.claude', 'neturon', 'cache', 'models')
    if (existsSync(p)) return p
    dir = join(dir, '..')
  }
  return join(import.meta.dir, '..', 'data', 'models')
}

type EncodeFn = (texts: string[], cacheDir: string) => Promise<number[][]>

let encodeImpl: EncodeFn | null = null

async function getEncode(): Promise<EncodeFn> {
  if (encodeImpl) return encodeImpl
  try {
    const mod = await import('../../../../Pj16-CodeAgent构建/Floria/src/tools/neturon/embedder.ts')
    encodeImpl = mod.encode as EncodeFn
  } catch {
    encodeImpl = hashEncode
  }
  return encodeImpl
}

function fnv1a(s: string): number {
  let h = 0x811c9dc5
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return h >>> 0
}

/** 降级嵌入：字符 unigram + bigram 散列，零依赖。 */
function hashVec(text: string): number[] {
  const v = new Array<number>(DIM).fill(0)
  const s = text.toLowerCase()
  const toks: string[] = []
  const runs = s.match(/[\u3000-\u9fff\uf900-\ufaff]+|[a-z0-9]+/g) ?? []
  for (const run of runs) {
    if (/^[a-z0-9]+$/.test(run)) {
      toks.push(run)
      continue
    }
    const chars = [...run]
    for (let i = 0; i < chars.length; i++) {
      toks.push(chars[i]!)
      if (i + 1 < chars.length) toks.push(chars[i]! + chars[i + 1]!)
    }
  }
  for (const t of toks) {
    const h = fnv1a(t)
    v[h % DIM]! += (h >>> 31) & 1 ? -1 : 1
  }
  let norm = 0
  for (const x of v) norm += x * x
  norm = Math.sqrt(norm) || 1
  for (let i = 0; i < DIM; i++) v[i]! /= norm
  return v
}

async function hashEncode(texts: string[]): Promise<number[][]> {
  return texts.map(hashVec)
}

export function textHash(text: string): string {
  return Bun.hash(text).toString(36)
}

interface EmbedCache {
  model: string
  dim: number
  items: Record<string, number[]>
}

function cachePath(dataDir: string): string {
  return join(dataDir, 'embeddings.json')
}

export function loadEmbedCache(dataDir: string): EmbedCache {
  const p = cachePath(dataDir)
  if (existsSync(p)) {
    try {
      const c = JSON.parse(readFileSync(p, 'utf-8')) as EmbedCache
      if (c.dim === DIM && c.items) return c
    } catch {
      /* 损坏则重建 */
    }
  }
  return { model: 'Xenova/bge-small-zh-v1.5', dim: DIM, items: {} }
}

export function saveEmbedCache(dataDir: string, cache: EmbedCache): void {
  mkdirSync(dataDir, { recursive: true })
  writeFileSync(cachePath(dataDir), JSON.stringify(cache))
}

/** 为所有条目的 blocks 取块级向量（缓存按文本哈希，跨条目共享）。 */
export async function buildBlockVectors(
  entries: MemEntry[],
  dataDir: string,
): Promise<{ vectors: Map<string, Float32Array[]>; fromCache: number; encoded: number }> {
  const cache = loadEmbedCache(dataDir)
  const missing: string[] = []
  const seen = new Set<string>()
  for (const e of entries) {
    if (e.deprecatedBy) continue
    for (const b of e.blocks) {
      const h = textHash(b)
      if (!cache.items[h] && !seen.has(h)) {
        seen.add(h)
        missing.push(b)
      }
    }
  }
  const encode = await getEncode()
  if (missing.length) {
    const vecs = await encode(missing, findModelCache())
    missing.forEach((b, i) => {
      cache.items[textHash(b)] = Array.from(vecs[i] ?? new Array(DIM).fill(0))
    })
    saveEmbedCache(dataDir, cache)
  }
  const vectors = new Map<string, Float32Array[]>()
  for (const e of entries) {
    if (e.deprecatedBy) continue
    vectors.set(e.id, e.blocks.map((b) => new Float32Array(cache.items[textHash(b)] ?? new Array(DIM).fill(0))))
  }
  return { vectors, fromCache: cache.items ? Object.keys(cache.items).length - missing.length : 0, encoded: missing.length }
}

/** 查询向量（不入缓存）。 */
export async function embedQuery(query: string): Promise<Float32Array> {
  const encode = await getEncode()
  const [v] = await encode([query], findModelCache())
  return new Float32Array(v ?? new Array(DIM).fill(0))
}

export { DIM }
