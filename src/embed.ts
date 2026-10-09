/**
 * GraphNeuron — 可插拔 Embedder
 *
 * 默认实现：本地 feature-hash 向量（零依赖、确定性），面向中文按「字符 unigram+bigram」、
 * 面向拉丁按「词小写」。检索的种子匹配用它；最终检索单元是蔓延后的子图。
 *
 * 接口固定，后续可替换为 BGE / 外部 embedding API（实现 embed(texts): Promise<number[][]>）。
 */

import type { GEvent } from './schema.ts'
import { verbalize } from './schema.ts'

export interface Embedder {
  readonly dim: number
  embed(texts: string[]): Promise<number[][]>
  embedSync(texts: string[]): number[][]
}

const DEFAULT_DIM = 256

function tokenize(text: string): string[] {
  const out: string[] = []
  const s = text.toLowerCase()
  // 拉丁/数字词
  const latin = s.match(/[a-z0-9]+/g) ?? []
  out.push(...latin)
  // 中文等：字符 unigram + bigram
  const cjkRuns = s.match(/[\u3000-\u9fff\uf900-\ufaff]+/g) ?? []
  for (const run of cjkRuns) {
    const chars = [...run]
    for (let i = 0; i < chars.length; i++) {
      out.push(chars[i])
      if (i + 1 < chars.length) out.push(chars[i] + chars[i + 1])
    }
  }
  return out
}

function fnv1a(str: string): number {
  let h = 0x811c9dc5
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return h >>> 0
}

/** 把文本散列成 L2 归一化的稀疏→稠密向量 */
export function hashEmbed(text: string, dim = DEFAULT_DIM): number[] {
  const v = new Array<number>(dim).fill(0)
  for (const tok of tokenize(text)) {
    const h = fnv1a(tok)
    const idx = h % dim
    const sign = (h >>> 31) & 1 ? -1 : 1
    v[idx] += sign
  }
  let norm = 0
  for (const x of v) norm += x * x
  norm = Math.sqrt(norm) || 1
  for (let i = 0; i < dim; i++) v[i] /= norm
  return v
}

export function cosine(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length)
  let dot = 0
  for (let i = 0; i < n; i++) dot += a[i] * b[i]
  return dot // 双方均已归一化
}

/** 结构指纹：把 move/chara/pol 拼成加权文本后再哈希（保证结构相似的事件向量接近） */
export function eventFingerprint(e: GEvent): string {
  const parts: string[] = []
  parts.push(`M:${e.move}`)
  for (const c of e.chara1) parts.push(`A:${c}`)
  for (const c of e.chara2) parts.push(`P:${c}`)
  parts.push(e.pol === '-' ? 'POL:-' : 'POL:+')
  if (e.mod) parts.push(`MOD:${e.mod}`)
  if (e.location) parts.push(`LOC:${e.location}`)
  return parts.join(' ')
}

/** 结构化事件向量（结构指纹为主，口语化文本为辅——结构权重更高） */
export function eventVector(e: GEvent, dim = DEFAULT_DIM, resolveChild?: (id: string) => GEvent | undefined): number[] {
  const a = hashEmbed(eventFingerprint(e), dim)
  const b = hashEmbed(verbalize(e, resolveChild), dim)
  const w = 0.7 // 结构权重
  const out = new Array<number>(dim)
  for (let i = 0; i < dim; i++) out[i] = w * a[i] + (1 - w) * b[i]
  let norm = 0
  for (const x of out) norm += x * x
  norm = Math.sqrt(norm) || 1
  for (let i = 0; i < dim; i++) out[i] /= norm
  return out
}

export class LocalEmbedder implements Embedder {
  readonly dim: number
  constructor(dim = DEFAULT_DIM) {
    this.dim = dim
  }
  embedSync(texts: string[]): number[][] {
    return texts.map((t) => hashEmbed(t, this.dim))
  }
  async embed(texts: string[]): Promise<number[][]> {
    return this.embedSync(texts)
  }
}
