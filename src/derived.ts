/**
 * derived 块层（反哺第三级：门控文本改写）。
 *
 * 约束：
 *   - append-only：原 mem.blocks 永不修改；每次改写生成一条 derived 记录。
 *   - 只有结构诊断命中的候选才进入这里（调用方负责筛选）。
 *   - LLM 一律走宿主接口（host.ts），不读本文件里的任何 API key。
 *   - 写入后 status=pending，须经 A/B 校验才可由读侧采用（accepted）。
 */

import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import type { LlmCall } from './host.ts'
import type { FeedbackCandidate } from './feedback.ts'
import type { MemEntry } from './types.ts'

export interface DerivedBlock {
  derived_from: string
  source_blocks: number[]
  blocks: string[]
  reason: string
  kind: FeedbackCandidate['kind']
  created_at: string
  /** pending → accepted / rejected；只有 accepted 可被读侧采用 */
  status: 'pending' | 'accepted' | 'rejected'
  validation: Record<string, unknown> | null
}

interface DerivedFile {
  version: 1
  items: DerivedBlock[]
}

export function derivedPath(neuronPath: string): string {
  return join(neuronPath, 'l1.cog', 'block_derived.json')
}

export function loadDerived(neuronPath: string): DerivedBlock[] {
  const p = derivedPath(neuronPath)
  if (!existsSync(p)) return []
  try {
    const f = JSON.parse(readFileSync(p, 'utf-8')) as DerivedFile
    return Array.isArray(f.items) ? f.items : []
  } catch {
    return []
  }
}

export function appendDerived(neuronPath: string, items: DerivedBlock[]): number {
  const all = [...loadDerived(neuronPath), ...items]
  const p = derivedPath(neuronPath)
  writeFileSync(p, JSON.stringify({ version: 1, items: all } satisfies DerivedFile, null, 2))
  return all.length
}

const SYSTEM = '你是记忆块重写模块。只输出 JSON，不要解释、不要 markdown 围栏。'

function buildPrompt(entry: MemEntry, cand: FeedbackCandidate): string {
  const blocks = entry.blocks.map((b, i) => `[${i}] ${b}`).join('\n')
  const action =
    cand.kind === 'multi_topic'
      ? `这条记忆包含多个话题。按语义把它拆成若干原子块：
- 每个块只讲一件事，可独立理解
- 块首块（block[0]）必须是查询形自然句，便于检索命中
- 保留原文全部事实，禁止新增、推测或改变事实
- 主语用真实姓名，禁止「对方/两人/他/她」等代词`
      : `这条记忆的检索锚点（block[0]）与真实查询不匹配。重写 block[0]：
- 用下列真实查询的措辞概括记忆里可被检索的那件事
- 只改首块措辞，其余块原样保留
- 禁止新增事实`

  return `原始记忆 ${cand.memoryId}
原始块：
${blocks}

真实检索查询：
${(cand.evidence.queries as string[] | undefined)?.join('\n') ?? '（无）'}

候选原因：${cand.reason}

${action}

输出 JSON：
{"blocks": ["块1", "块2", "..."], "reason": "一句话说明"}`
}

function extractJson(text: string): Record<string, unknown> | null {
  let content = text.trim()
  if (content.startsWith('```')) {
    const parts = content.split('```')
    content = (parts[1] ?? content).trim()
    if (content.startsWith('json')) content = content.slice(4).trim()
  }
  const m = /\{[\s\S]*\}/.exec(content)
  try {
    return JSON.parse(m ? m[0] : content) as Record<string, unknown>
  } catch {
    return null
  }
}

/** 对单个候选调用宿主 LLM 生成 derived 块（不写盘）。 */
export async function generateDerived(
  llm: LlmCall,
  entry: MemEntry,
  cand: FeedbackCandidate,
): Promise<DerivedBlock> {
  const text = await llm(SYSTEM, buildPrompt(entry, cand))
  const parsed = extractJson(text)
  const blocks = Array.isArray(parsed?.blocks)
    ? (parsed!.blocks as unknown[]).map((b) => String(b).trim()).filter(Boolean)
    : []
  if (!blocks.length) throw new Error(`LLM 未返回有效 blocks: ${text.slice(0, 160)}`)

  return {
    derived_from: entry.id,
    source_blocks: entry.blocks.map((_, i) => i),
    blocks,
    reason: String(parsed?.reason ?? ''),
    kind: cand.kind,
    created_at: new Date().toISOString(),
    status: 'pending',
    validation: null,
  }
}
