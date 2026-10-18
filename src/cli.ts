/** CLI：build / recall / eval / communities / stats */

import { writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { buildRuntime, loadRuntime, recall, type Dataset, type RecallMode } from './recall.ts'
import { buildBlockVectors } from './embed.ts'
import { runEval, formatEval } from './eval.ts'
import { diagnoseFeedback, formatFeedback } from './feedback.ts'
import { resolveHostLlm } from './host.ts'
import { appendDerived, generateDerived, loadDerived } from './derived.ts'
import { defaultNeuronPath, defaultLjjNeuronPath } from './load.ts'
import { listCogVersions, readCogVersionManifest, restoreCogVersion, snapshotCogVersion } from './version.ts'

function arg(name: string, fallback?: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`)
  return i >= 0 ? process.argv[i + 1] : fallback
}

const cmd = process.argv[2] ?? 'help'
const dataset = (arg('dataset', 'pj11') as Dataset)
if (dataset !== 'pj11' && dataset !== 'ljj') {
  console.error(`未知数据集: ${dataset}（可选 pj11 | ljj）`)
  process.exit(1)
}
const dsLabel = dataset === 'ljj' ? '李京瑾数据集' : 'Pj11 项目记忆'

if (cmd === 'build') {
  const { stats } = await buildRuntime({ dataset, resolution: Number(arg('resolution', '1')) })
  console.log(JSON.stringify(stats, null, 2))
} else if (cmd === 'eval') {
  const { runtime } = await buildRuntime({ dataset, resolution: Number(arg('resolution', '1')) })
  const scores = await runEval(runtime)
  console.log(formatEval(scores, dsLabel))
} else if (cmd === 'recall') {
  const mode = (process.argv[3] ?? 'mem') as RecallMode
  const query = process.argv[4] ?? ''
  if (!query) {
    console.error('用法: bun run src/cli.ts recall <mem|graph|cog|community> "查询" [--top 5] [--variant mix] [--dataset pj11|ljj]')
    process.exit(1)
  }
  const { runtime } = await buildRuntime({ dataset, resolution: Number(arg('resolution', '1')) })
  const res = await recall(mode, query, runtime, {
    topK: Number(arg('top', '5')),
    variant: (arg('variant', 'mix') as never),
  })
  console.log(JSON.stringify(res, null, 2))
} else if (cmd === 'communities') {
  const { runtime } = await buildRuntime({ dataset, resolution: Number(arg('resolution', '1')) })
  for (const c of runtime.communities.communities) {
    console.log(c.text)
    console.log('\n' + '─'.repeat(72) + '\n')
  }
} else if (cmd === 'stats') {
  const { stats } = await buildRuntime({ dataset, resolution: Number(arg('resolution', '1')) })
  console.log(JSON.stringify(stats, null, 2))
} else if (cmd === 'feedback') {
  // 读已构建的图/社群；向量从缓存补齐（不重跑社区检测）
  const runtime = loadRuntime({ dataset })
  const { vectors } = await buildBlockVectors(runtime.entries, runtime.dataDir)
  runtime.vectors = vectors
  const report = diagnoseFeedback(
    runtime.entries,
    runtime.vectors,
    runtime.graph,
    runtime.communities.communities,
    runtime.usageSignal,
    {
      dataset,
      minHits: Number(arg('min-hits', '2')),
      topPerKind: Number(arg('top', '20')),
    },
  )
  const sub = process.argv[3]

  if (sub === 'rewrite') {
    // 第三级门控改写：只处理结构诊断命中的候选，结果写 derived（pending）
    const kinds = new Set(['multi_topic', 'noisy_anchor'])
    const done = new Set(loadDerived(runtime.neuronPath).map((d) => `${d.derived_from}\u0000${d.kind}`))
    const picks = report.candidates
      .filter((c) => kinds.has(c.kind) && !done.has(`${c.memoryId}\u0000${c.kind}`))
      .slice(0, Number(arg('limit', '5')))
    if (!picks.length) {
      console.log('没有待改写的候选（或都已生成 derived）')
      process.exit(0)
    }
    const llm = await resolveHostLlm(runtime.neuronPath)
    console.log(`宿主 LLM: ${llm.provider}/${llm.model}，候选 ${picks.length} 条`)
    const byId = new Map(runtime.entries.map((e) => [e.id, e]))
    const items = []
    for (const [i, c] of picks.entries()) {
      const e = byId.get(c.memoryId)
      if (!e) continue
      process.stdout.write(`  ${i + 1}/${picks.length} ${c.memoryId} (${c.kind}) … `)
      try {
        const d = await generateDerived(llm.call, e, c)
        items.push(d)
        console.log(`ok ${d.blocks.length} 块`)
      } catch (err) {
        console.log(`失败: ${err instanceof Error ? err.message : String(err)}`)
      }
    }
    if (process.argv.includes('--dry-run')) {
      console.log(JSON.stringify(items, null, 2))
      console.log('（--dry-run：未写入）')
    } else if (items.length) {
      const total = appendDerived(runtime.neuronPath, items)
      console.log(`已写入 ${items.length} 条 derived（pending），累计 ${total}：${join(runtime.neuronPath, 'l1.cog', 'block_derived.json')}`)
    }
    process.exit(0)
  }

  const out = join(runtime.dataDir, 'feedback.json')
  writeFileSync(out, JSON.stringify(report, null, 2))
  if (process.argv.includes('--json')) console.log(JSON.stringify(report, null, 2))
  else console.log(formatFeedback(report))
  console.log(`\n报告: ${out}`)
} else if (cmd === 'version') {
  const sub = process.argv[3] ?? 'list'
  const neuronPath = dataset === 'ljj' ? defaultLjjNeuronPath() : defaultNeuronPath()
  if (sub === 'list') {
    const vs = listCogVersions(neuronPath)
    for (const v of vs) {
      const m = readCogVersionManifest(neuronPath, v)
      console.log(`${v}  ${m?.snapshot_at ?? ''}  ${(m?.snapshotted ?? []).join(',')}`)
    }
    if (!vs.length) console.log('（无版本）')
  } else if (sub === 'snapshot') {
    const r = snapshotCogVersion(neuronPath, {
      params: { dataset },
      lineage: { [new Date().toISOString()]: 'memgraph snapshot' },
    })
    console.log(`已快照 ${r.version} → ${r.dir}`)
  } else if (sub === 'restore') {
    const v = process.argv[4]
    if (!v) {
      console.error('用法: bun run src/cli.ts version restore vN')
      process.exit(1)
    }
    console.log(`已恢复 ${v}: ${restoreCogVersion(neuronPath, v).join(', ')}`)
  } else {
    console.error('用法: bun run src/cli.ts version <list|snapshot|restore vN>')
    process.exit(1)
  }
} else {
  console.log(`Pj11 MemGraph 原型

  bun run src/cli.ts build                    重建向量/图/社群
  bun run src/cli.ts eval                     跑检索变体对比
  bun run src/cli.ts recall mem "查询"         记忆检索（写 precog）
  bun run src/cli.ts recall graph "查询"       图检索（PPR）
  bun run src/cli.ts recall cog "查询"         认知检索（社群概念）
  bun run src/cli.ts recall community "查询"   社群文本检索
  bun run src/cli.ts communities              打印全部社群文本
  bun run src/cli.ts feedback                 反哺诊断（usage → 结构候选 → 计划）
  bun run src/cli.ts feedback rewrite --limit 5
                                              门控改写：宿主 LLM → derived 块（pending）
                                              可加 --dry-run 只预览不写盘
  bun run src/cli.ts version list             认知层版本列表
  bun run src/cli.ts version snapshot         快照 l1.cog → versions/vN
  bun run src/cli.ts version restore vN       恢复版本（当前文件备份 .bak）

  所有命令可加 --dataset ljj 切到唯一测试库李京瑾神经元（data/ljj-neuron/l2.mem/），
  数据产物落在 data/ljj/，与 Pj11 项目记忆的 data/ 互不覆盖。
`)
}
