/** 反哺循环：usage 信号 / 结构诊断 / 版本接口 */

import { describe, test, expect, afterAll } from 'bun:test'
import { existsSync, mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { buildUsageSignal } from '../src/usage.ts'
import { diagnoseFeedback, type FeedbackCandidate } from '../src/feedback.ts'
import { generateDerived } from '../src/derived.ts'
import { defaultLjjNeuronPath } from '../src/load.ts'
import {
  listCogVersions, readCogVersionManifest, restoreCogVersion, snapshotCogVersion,
} from '../src/version.ts'
import type { MemEntry, PrecogRecord } from '../src/types.ts'

const tmpRoot = join(import.meta.dir, '..', 'data', 'ljj', 'test', 'version-neuron')

afterAll(() => {
  if (existsSync(tmpRoot)) rmSync(tmpRoot, { recursive: true, force: true })
})

function record(
  id: string,
  query: string,
  results: Array<{ id: string; accuracy: string }>,
): PrecogRecord {
  return {
    record_id: id,
    query,
    keywords: [],
    top_k: results.length,
    results: results.map((r) => ({ ...r, summary: '' })),
    description: '',
  }
}

describe('usage 信号', () => {
  test('accuracy 分级、窗口配对、false 不计边', () => {
    const now = new Date(2026, 7, 1, 10, 0, 0).getTime()
    const sig = buildUsageSignal(
      [
        record('PCX_2026-08-01-10:00:00', '猫', [
          { id: 'A', accuracy: 'true' },
          { id: 'B', accuracy: 'revelant' },
          { id: 'C', accuracy: 'false' },
          { id: 'D', accuracy: 'true' },
        ]),
      ],
      { now, coHitWindow: 3 },
    )
    const a = sig.stats.get('A')!
    const b = sig.stats.get('B')!
    const c = sig.stats.get('C')!
    const d = sig.stats.get('D')!
    expect(a.trueHits).toBe(1)
    expect(a.weighted).toBeCloseTo(1, 5)
    expect(b.relevantHits).toBe(1)
    expect(b.weighted).toBeCloseTo(0.5, 5)
    expect(c.falseHits).toBe(1)
    expect(c.weighted).toBe(0)
    expect(d.trueHits).toBe(1) // 统计用全量结果
    expect(sig.edges.length).toBe(1) // 只有窗口内的 A-B
    expect(sig.edges[0]!.source).toBe('A')
    expect(sig.edges[0]!.target).toBe('B')
    expect(sig.edges[0]!.weight).toBeCloseTo(0.5, 5)
  })

  test('半衰期衰减', () => {
    const t0 = new Date(2026, 0, 1).getTime()
    const t180 = t0 + 180 * 86_400_000
    const r = record('PCX_2026-01-01-00:00:00', 'q', [{ id: 'A', accuracy: 'true' }])
    expect(buildUsageSignal([r], { now: t0 }).stats.get('A')!.weighted).toBeCloseTo(1, 5)
    expect(buildUsageSignal([r], { now: t180 }).stats.get('A')!.weighted).toBeCloseTo(0.5, 5)
  })
})

describe('结构诊断', () => {
  test('多话题 mem 与重复块可被识别', () => {
    const entries: MemEntry[] = [
      {
        id: 'A', blocks: ['猫喜欢晒太阳', '猫粮快吃完了'], text: '猫喜欢晒太阳 猫粮快吃完了',
        source: 'QQ', revelant: [], coreFiles: [], deprecatedBy: null, time: null,
      },
      {
        id: 'B', blocks: ['猫喜欢晒太阳'], text: '猫喜欢晒太阳',
        source: 'QQ', revelant: [], coreFiles: [], deprecatedBy: null, time: null,
      },
    ]
    const vectors = new Map<string, Float32Array[]>([
      ['A', [new Float32Array([1, 0]), new Float32Array([0, 1])]],
      ['B', [new Float32Array([1, 0])]],
    ])
    const graph = {
      nodes: [
        { id: 'A', title: 'A', time: null },
        { id: 'B', title: 'B', time: null },
      ],
      edges: [{ source: 'A', target: 'B', type: 'entity' as const, weight: 3, evidence: '共享实体: 猫' }],
    }
    const communities = [{
      id: 'C01', title: '猫', size: 2, modularity: 0.5, keywords: ['猫'],
      members: ['A', 'B'], timeSpan: null, text: '', neighbors: [],
    }]
    const now = new Date(2026, 7, 1).getTime()
    const usage = buildUsageSignal(
      [
        record('PCX_2026-08-01-10:00:00', '猫晒太阳', [
          { id: 'A', accuracy: 'true' },
          { id: 'B', accuracy: 'true' },
        ]),
        record('PCX_2026-08-02-10:00:00', '猫粮', [
          { id: 'A', accuracy: 'true' },
          { id: 'B', accuracy: 'revelant' },
        ]),
      ],
      { now },
    )
    const report = diagnoseFeedback(entries, vectors, graph, communities, usage, { dataset: 'test' })
    const kinds = new Set(report.candidates.map((c) => c.kind))
    expect(kinds.has('multi_topic')).toBe(true)
    expect(kinds.has('duplicate_block')).toBe(true)
    const multi = report.candidates.find((c) => c.kind === 'multi_topic')!
    expect(multi.memoryId).toBe('A')
    expect(multi.action).toBe('derived_split')
  })
})

describe('derived 门控改写', () => {
  test('注入式 LLM → derived 块（pending，不改原块）', async () => {
    const entry: MemEntry = {
      id: 'A', blocks: ['猫喜欢晒太阳', '猫粮快吃完了'], text: '猫喜欢晒太阳 猫粮快吃完了',
      source: 'QQ', revelant: [], coreFiles: [], deprecatedBy: null, time: null,
    }
    const cand: FeedbackCandidate = {
      kind: 'multi_topic', memoryId: 'A', score: 1, action: 'derived_split',
      reason: '块间 cos 低', evidence: { queries: ['猫晒太阳', '猫粮'] },
    }
    const d = await generateDerived(
      async () => '```json\n{"blocks": ["猫晒太阳时喜欢趴在窗台", "猫粮快吃完了"], "reason": "拆分"}\n```',
      entry,
      cand,
    )
    expect(d.derived_from).toBe('A')
    expect(d.blocks).toHaveLength(2)
    expect(d.status).toBe('pending')
    expect(d.source_blocks).toEqual([0, 1])
  })
})

describe('版本接口', () => {
  test('李京瑾既有版本可列出且 manifest 可读', () => {
    const root = defaultLjjNeuronPath()
    const vs = listCogVersions(root)
    expect(vs.length).toBeGreaterThanOrEqual(10)
    expect(vs[0]).toBe('v0')
    const last = readCogVersionManifest(root, vs[vs.length - 1]!)
    expect(last).not.toBeNull()
    expect(last!.snapshotted.length).toBeGreaterThan(0)
  })

  test('快照 → 修改 → 恢复', async () => {
    const neuron = tmpRoot
    mkdirSync(join(neuron, 'l1.cog'), { recursive: true })
    writeFileSync(join(neuron, 'l1.cog', 'cog.json'), '{"v":1}')
    const snap = snapshotCogVersion(neuron, { params: { test: true } })
    expect(snap.version).toBe('v0')
    expect(listCogVersions(neuron)).toEqual(['v0'])

    writeFileSync(join(neuron, 'l1.cog', 'cog.json'), '{"v":2}')
    const restored = restoreCogVersion(neuron, 'v0')
    expect(restored).toEqual(['cog.json'])
    expect(await Bun.file(join(neuron, 'l1.cog', 'cog.json')).text()).toBe('{"v":1}')
    expect(existsSync(join(neuron, 'l1.cog', 'cog.json.bak'))).toBe(true)
  })
})
