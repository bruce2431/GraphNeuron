/** MemGraph 原型验证：数据装载 / 图与社群 / mix 检索 / 分粒度 recall / precog 边界 */

import { describe, test, expect, beforeAll, afterAll } from 'bun:test'
import { existsSync, mkdirSync, unlinkSync } from 'node:fs'
import { join } from 'node:path'
import { Database } from 'bun:sqlite'
import { defaultNeuronPath, loadMemories } from '../src/load.ts'
import { buildRuntime, recall, defaultDataDir, type Runtime } from '../src/recall.ts'
import { rank, MIX_WEIGHTS } from '../src/retrieval.ts'
import { EVAL_QUERIES, PARAPHRASE_QUERIES } from '../src/evalQueries.ts'
import { embedQuery } from '../src/embed.ts'

const dataDir = defaultDataDir()
const testDb = join(dataDir, 'test', 'precog-isolation.db')

let rt: Runtime
let activeIds: Set<string>

beforeAll(async () => {
  mkdirSync(join(dataDir, 'test'), { recursive: true })
  if (existsSync(testDb)) unlinkSync(testDb)
  const built = await buildRuntime({ dataDir })
  rt = built.runtime
  activeIds = new Set(rt.entries.filter((e) => !e.deprecatedBy).map((e) => e.id))
}, 180_000)

afterAll(() => {
  if (existsSync(testDb)) unlinkSync(testDb)
})

describe('数据装载', () => {
  test('mem.db 可读且条目关系自洽', () => {
    const entries = loadMemories(defaultNeuronPath())
    expect(entries.length).toBeGreaterThan(100)
    const active = entries.filter((e) => !e.deprecatedBy)
    expect(active.length).toBeGreaterThan(0)
    expect(active.length).toBeLessThanOrEqual(entries.length)
    expect(entries.reduce((s, e) => s + e.blocks.length, 0)).toBeGreaterThanOrEqual(entries.length)
    for (const e of active) {
      expect(e.blocks.length).toBeGreaterThan(0)
      expect(e.id.startsWith('PJ11_MEM_')).toBe(true)
      expect(e.time).not.toBeNull()
    }
  })

  test('每条活跃记忆都有块级向量', () => {
    for (const id of activeIds) {
      const vs = rt.vectors.get(id)
      expect(vs).toBeDefined()
      expect(vs!.length).toBeGreaterThan(0)
      for (const v of vs!) expect(v.length).toBe(512)
    }
  })
})

describe('记忆图与认知社群', () => {
  test('图节点 = 活跃记忆，边带类型与证据', () => {
    expect(rt.graph.nodes.length).toBe(activeIds.size)
    const types = new Set(['reference', 'entity', 'semantic', 'temporal', 'usage'])
    for (const e of rt.graph.edges) {
      expect(types.has(e.type)).toBe(true)
      expect(e.evidence.length).toBeGreaterThan(0)
      expect(activeIds.has(e.source)).toBe(true)
      expect(activeIds.has(e.target)).toBe(true)
    }
  })

  test('社群覆盖全部活跃记忆且文本结构可读', () => {
    const { communities } = rt.communities
    expect(communities.length).toBeGreaterThan(1)
    const covered = new Set(communities.flatMap((c) => c.members))
    for (const id of activeIds) expect(covered.has(id)).toBe(true)
    for (const c of communities) {
      expect(c.id).toMatch(/^C\d{2}$/)
      expect(c.size).toBeGreaterThanOrEqual(2)
      expect(c.text).toContain('【核心记忆】')
      expect(c.text).toContain('【内部关系】')
      expect(c.text).toContain('关键词:')
      expect(c.title.length).toBeGreaterThan(0)
      for (const m of c.members) expect(activeIds.has(m)).toBe(true)
    }
    expect(rt.communities.modularity).toBeGreaterThan(0.2)
  })

  test('社群级边桥接词非空', () => {
    for (const e of rt.communities.edges) {
      expect(e.source).not.toBe(e.target)
      expect(e.weight).toBeGreaterThan(0)
      expect(e.bridge.length).toBeGreaterThan(0)
    }
  })
})

describe('mem 检索', () => {
  async function topIds(query: string, k = 5): Promise<string[]> {
    const ctx = {
      entries: rt.entries,
      vectors: rt.vectors,
      qVec: await embedQuery(query),
      usage: rt.usage,
    }
    return rank(ctx, query, 'mix', k).map((r) => r.id)
  }

  test('mix 权重来自网格搜索最优', () => {
    expect(MIX_WEIGHTS.dense + MIX_WEIGHTS.bm25 + MIX_WEIGHTS.kw).toBeCloseTo(1, 6)
    expect(MIX_WEIGHTS.dense).toBe(0.6)
  })

  test('精确措辞查询命中', async () => {
    const q = EVAL_QUERIES[0]!
    const ids = await topIds(q.query)
    expect(ids[0]).toBe(q.expected[0])
  })

  test('改写查询进入 top-3', async () => {
    const q = PARAPHRASE_QUERIES[0]!
    const ids = await topIds(q.query)
    expect(ids.slice(0, 3)).toContain(q.expected[0])
  })

  test('已废弃记忆不出现在结果中', async () => {
    const q = EVAL_QUERIES[2]!
    const ids = await topIds(q.query, 8)
    const deprecated = new Set(rt.entries.filter((e) => e.deprecatedBy).map((e) => e.id))
    for (const id of ids) expect(deprecated.has(id)).toBe(false)
  })
})

describe('分粒度 recall', () => {
  test('mem 写 precog，其余模式不写且 precog=null', async () => {
    const query = 'GraphNeuron 事件图引擎怎么启动'

    for (const mode of ['graph', 'cog', 'community'] as const) {
      const res = await recall(mode, query, rt, { topK: 3, precogDb: testDb })
      expect(res.mode).toBe(mode)
      expect(res.precog).toBeNull()
      expect(existsSync(testDb)).toBe(false)
    }

    const mem = await recall('mem', query, rt, { topK: 3, precogDb: testDb })
    expect(mem.precog).not.toBeNull()
    expect(existsSync(testDb)).toBe(true)
    const db = new Database(testDb, { readonly: true })
    try {
      const rows = db.query('SELECT status, query, results FROM precog').all() as Array<{
        status: string
        query: string
        results: string
      }>
      expect(rows.length).toBe(1)
      expect(rows[0]!.status).toBe('pre')
      expect(rows[0]!.query).toBe(query)
      const results = JSON.parse(rows[0]!.results) as Array<{ id: string; accuracy: string }>
      expect(results.length).toBe(3)
      for (const r of results) expect(r.accuracy).toBe('')
    } finally {
      db.close()
    }
  }, 60_000)

  test('graph 返回 PPR 子图与主路径', async () => {
    const res = await recall('graph', '预览页 canvas 放大后画面乱成一团', rt, { topK: 5 })
    expect(res.results.length).toBeGreaterThan(0)
    expect(res.subgraph).toBeDefined()
    const nodeIds = new Set(res.subgraph!.nodes.map((n) => n.id))
    expect(nodeIds.size).toBe(res.results.length)
    for (const e of res.subgraph!.edges) {
      expect(nodeIds.has(e.source)).toBe(true)
      expect(nodeIds.has(e.target)).toBe(true)
      expect(e.evidence.length).toBeGreaterThan(0)
    }
    expect(res.path!.length).toBeGreaterThan(0)
    expect(res.path!.every((p) => nodeIds.has(p.id))).toBe(true)
  }, 60_000)

  test('cog 返回命中的社群概念，成员内排序', async () => {
    const res = await recall('cog', '预览页手机加载不出来', rt, { topK: 3 })
    expect(res.concepts).toBeDefined()
    expect(res.concepts!.length).toBeGreaterThan(0)
    const memberIds = new Set(res.concepts!.flatMap((c) => c.members))
    for (const r of res.results) expect(memberIds.has(r.id)).toBe(true)
    for (const c of res.concepts!) expect(c.cos).toBeGreaterThanOrEqual(0.35)
  }, 60_000)

  test('community 只返回社群文本', async () => {
    const res = await recall('community', 'LoRAMoE 社群专家训练', rt, { topK: 2 })
    expect(res.results.length).toBe(0)
    expect(res.communities!.length).toBeGreaterThan(0)
    for (const c of res.communities!) {
      expect(c.text).toContain('【核心记忆】')
      expect(c.cos).toBeGreaterThanOrEqual(-1)
    }
  }, 60_000)
})
