/**
 * GraphNeuron HTTP API（Bun.serve）
 *
 * 供 Floria 以 API 调用。测试方式 = API 调用（见 test/api.test.ts）。
 * 启动：`bun run src/api.ts`（默认端口 8787，可用 PORT 覆盖）
 *
 * 端点：
 *   GET  /health
 *   POST /ingest    { events: EventInput[], rebuild?: boolean }
 *   POST /search    { query: QueryPattern, k?, alpha?, hops?, bumpUsage?, ... }
 *   POST /rebuild   { }                      重建结构边
 *   GET  /events                              全部事件
 *   GET  /events/:id
 *   GET  /graph                               快照
 *   POST /community { resolutions?, minSize?, edgeTypes? }
 *   POST /save      { path }
 *   POST /load      { path }
 *   POST /reset     { }
 */

import { GraphNeuron } from './store.ts'
import { search, type QueryPattern, type SearchOptions } from './retrieve.ts'
import { detectCommunities, type CommunityOptions } from './community.ts'
import type { EventInput } from './schema.ts'

export function createApp(gn: GraphNeuron = new GraphNeuron()) {
  async function handle(req: Request): Promise<Response> {
    const url = new URL(req.url)
    const path = url.pathname
    const json = (data: unknown, status = 200) =>
      new Response(JSON.stringify(data, null, 2), { status, headers: { 'content-type': 'application/json; charset=utf-8' } })
    const err = (msg: string, status = 400) => json({ error: msg }, status)

    try {
      if (req.method === 'GET' && path === '/health') {
        return json({ ok: true, name: 'GraphNeuron', events: gn.size, edges: gn.edges.size })
      }

      if (req.method === 'POST' && path === '/ingest') {
        const body = (await req.json()) as { events?: EventInput[]; rebuild?: boolean }
        if (!Array.isArray(body.events)) return err('events 必须是数组')
        const added = []
        for (const e of body.events) added.push(gn.addEvent(e).id)
        const edgeCounts = body.rebuild === false ? null : gn.buildEdges()
        return json({ added, addedCount: added.length, edgeCounts, stats: gn.snapshot().stats })
      }

      if (req.method === 'POST' && path === '/search') {
        const body = (await req.json()) as { query?: QueryPattern } & SearchOptions
        if (!body || !body.query) return err('query 必填')
        return json(search(gn, body.query, body))
      }

      if (req.method === 'POST' && path === '/rebuild') {
        return json({ edgeCounts: gn.buildEdges(), stats: gn.snapshot().stats })
      }

      if (req.method === 'GET' && path === '/events') {
        return json(gn.snapshot().events)
      }

      if (req.method === 'GET' && path.startsWith('/events/')) {
        const id = decodeURIComponent(path.slice('/events/'.length))
        const ev = gn.getEvent(id)
        return ev ? json(ev) : err(`未找到事件 ${id}`, 404)
      }

      if (req.method === 'GET' && path === '/graph') {
        return json(gn.snapshot())
      }

      if (req.method === 'POST' && path === '/community') {
        const body = (await req.json().catch(() => ({}))) as CommunityOptions
        return json(detectCommunities(gn, body ?? {}))
      }

      if (req.method === 'POST' && path === '/save') {
        const body = (await req.json()) as { path?: string }
        if (!body?.path) return err('path 必填')
        gn.save(body.path)
        return json({ saved: body.path })
      }

      if (req.method === 'POST' && path === '/load') {
        const body = (await req.json()) as { path?: string }
        if (!body?.path) return err('path 必填')
        const loaded = GraphNeuron.load(body.path)
        gn.events = loaded.events
        gn.edges = loaded.edges
        gn.actors = loaded.actors
        return json({ loaded: body.path, stats: gn.snapshot().stats })
      }

      if (req.method === 'POST' && path === '/reset') {
        gn = new GraphNeuron()
        return json({ ok: true })
      }

      return err(`未知路由: ${req.method} ${path}`, 404)
    } catch (e) {
      return err(e instanceof Error ? e.message : String(e), 500)
    }
  }

  return { handle, gn }
}

if (import.meta.main) {
  const port = Number(process.env.PORT ?? 8787)
  const app = createApp()
  const server = Bun.serve({ port, fetch: app.handle })
  console.log(`GraphNeuron API 已启动: http://localhost:${server.port}`)
}
