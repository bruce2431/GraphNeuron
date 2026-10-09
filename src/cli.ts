/**
 * GraphNeuron CLI（本地快速验证，等价于 API）
 *
 *   bun run src/cli.ts ingest <events.json>            # events.json = EventInput[]
 *   bun run src/cli.ts search '{"move":"告诉"}' [k]    # 检索
 *   bun run src/cli.ts community [resolution]
 *   bun run src/cli.ts demo                             # 内置演示数据全链路
 */

import { readFileSync } from 'node:fs'
import { GraphNeuron } from './store.ts'
import { search } from './retrieve.ts'
import { detectCommunities } from './community.ts'
import type { EventInput } from './schema.ts'

function demoEvents(): EventInput[] {
  return [
    { id: 'm_20250101090000_001', chara1: 'A', move: '告诉', chara2: 'B', source: 's1', location: '咖啡馆',
      content: { id: 'm_20250101085900_001', chara1: 'C', move: '表白', chara2: 'D', source: 's1', mod: '曾经' } },
    { id: 'm_20250101091500_002', chara1: 'A', move: '绝交', chara2: 'C', source: 's1', location: '咖啡馆', causes: ['m_20250101090000_001'] },
    { id: 'm_20250101120000_003', chara1: 'B', move: '转告', chara2: 'D', source: 's2',
      content: 'A 说 C 喜欢 D' },
    { id: 'm_20250102080000_004', chara1: 'A', move: '道歉', chara2: 'C', source: 's2', pol: '-' },
  ]
}

async function main() {
  const [cmd, ...args] = process.argv.slice(2)
  const gn = new GraphNeuron()

  if (cmd === 'demo') {
    for (const e of demoEvents()) gn.addEvent(e)
    console.log('边统计:', gn.buildEdges())
    console.log('图统计:', gn.snapshot().stats)
    const r = search(gn, { move: '告诉', chara1: 'A' }, { k: 5, hops: 3 })
    console.log('\n检索 告诉/A →')
    for (const x of r.results) console.log(`  [${x.activation}] hop=${x.hop} ${x.verbal}`)
    console.log('  路径:', r.path.join(' → '))
    console.log('  子图边:', r.subgraph.edges.map((e) => `${e.type}:${e.weight}`).join(', '))
    console.log('\n社群:')
    for (const cr of detectCommunities(gn, { resolutions: [1.0] })) {
      console.log(`  res=${cr.resolution} modularity=${cr.modularity}`)
      for (const c of cr.communities) console.log(`    [${c.size}] d=${c.density} ${c.members.map((m) => m.move).join(',')}`)
    }
    return
  }

  if (cmd === 'ingest') {
    const file = args[0]
    if (!file) throw new Error('用法: ingest <events.json>')
    const events = JSON.parse(readFileSync(file, 'utf8')) as EventInput[]
    for (const e of events) gn.addEvent(e)
    console.log(JSON.stringify({ edgeCounts: gn.buildEdges(), stats: gn.snapshot().stats }, null, 2))
    return
  }

  if (cmd === 'search') {
    const q = JSON.parse(args[0] ?? '{}')
    const k = args[1] ? Number(args[1]) : 5
    console.log(JSON.stringify(search(gn, q, { k }), null, 2))
    return
  }

  if (cmd === 'community') {
    const res = args[0] ? [Number(args[0])] : undefined
    console.log(JSON.stringify(detectCommunities(gn, { resolutions: res }), null, 2))
    return
  }

  console.log('用法: demo | ingest <file> | search <json> [k] | community [res]')
}

main()
