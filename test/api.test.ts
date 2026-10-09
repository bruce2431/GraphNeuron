/**
 * GraphNeuron — API 调用测试（bun test）
 *
 * 用户指定测试方式 = API 调用：这里起一个真实 Bun.serve，用 fetch 打全部端点。
 * 运行：`bun test`
 */

import { afterAll, beforeAll, describe, expect, test } from 'bun:test'
import { createApp } from '../src/api.ts'

let server: ReturnType<typeof Bun.serve>
let base = ''

async function call(path: string, body?: unknown, method = 'POST') {
  const res = await fetch(base + path, {
    method,
    headers: { 'content-type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  return { status: res.status, data: await res.json() }
}

beforeAll(() => {
  const app = createApp()
  server = Bun.serve({ port: 0, fetch: app.handle })
  base = `http://localhost:${server.port}`
})

afterAll(() => server?.stop(true))

describe('GraphNeuron API', () => {
  test('health', async () => {
    const { status, data } = await call('/health', undefined, 'GET')
    expect(status).toBe(200)
    expect(data.name).toBe('GraphNeuron')
  })

  test('ingest 嵌套事件（携带 m₁(m₂)）+ 因果边', async () => {
    const { status, data } = await call('/ingest', {
      events: [
        {
          id: 'm_20250101090000_001',
          chara1: 'A', move: '告诉', chara2: 'B', source: 's1', location: '咖啡馆',
          content: { id: 'm_20250101085900_001', chara1: 'C', move: '表白', chara2: 'D', source: 's1', mod: '曾经' },
        },
        { id: 'm_20250101091500_002', chara1: 'A', move: '绝交', chara2: 'C', source: 's1', location: '咖啡馆', causes: ['m_20250101090000_001'] },
        { id: 'm_20250101120000_003', chara1: 'B', move: '转告', chara2: 'D', source: 's2', content: 'A 说 C 喜欢 D' },
        { id: 'm_20250101093000_004', chara1: 'A', move: '告诉', chara2: 'C', pol: '-', source: 's1' },
      ],
    })
    expect(status).toBe(200)
    expect(data.addedCount).toBe(4) // 顶层 4 条（含 1 个嵌套子事件，子事件不单独计）
    expect(data.stats.events).toBe(5) // 图内共 5 个事件节点
    expect(data.edgeCounts.carries).toBe(1)
    expect(data.edgeCounts.causes).toBe(1)
    expect(data.edgeCounts.char).toBeGreaterThan(0)
    expect(data.edgeCounts.temporal).toBeGreaterThan(0)
  })

  test('search 蔓延：从「告诉」命中嵌套表白事件', async () => {
    const { data } = await call('/search', { query: { move: '告诉', chara1: 'A' }, k: 5, hops: 3 })
    expect(data.results.length).toBeGreaterThan(0)
    const ids = data.results.map((r: any) => r.id)
    // 携带的子事件（表白）应通过 carries 边被蔓延激活
    expect(ids).toContain('m_20250101085900_001')
    // 主路径至少两个节点
    expect(data.path.length).toBeGreaterThanOrEqual(2)
    // 子图边非空
    expect(data.subgraph.edges.length).toBeGreaterThan(0)
    // usage 闭环已回写
    expect(data.usage_bumped).toBeGreaterThan(0)
  })

  test('search 否定同源：肯定查询也能召回否定事件', async () => {
    const { data } = await call('/search', { query: { move: '告诉', chara2: 'C' }, k: 5, hops: 3 })
    const ids = data.results.map((r: any) => r.id)
    expect(ids).toContain('m_20250101093000_004') // pol='-' 的「没告诉」
  })

  test('search 时空邻近：时间边把相邻事件顺带激活', async () => {
    const { data } = await call('/search', { query: { location: '咖啡馆', time: '20250101090000' }, k: 6, hops: 3 })
    const ids = data.results.map((r: any) => r.id)
    expect(ids).toContain('m_20250101091500_002')
  })

  test('community 社群检测', async () => {
    const { data } = await call('/community', { resolutions: [1.0], minSize: 2 })
    expect(Array.isArray(data)).toBe(true)
    expect(data[0].modularity).toBeDefined()
  })

  test('404', async () => {
    const { status } = await call('/nope', {}, 'POST')
    expect(status).toBe(404)
  })
})
