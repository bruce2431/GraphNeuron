/** 检索变体对比：hit@1/3/5 + MRR + 平均延迟（精确集 / 改写集 / 合计） */

import { EVAL_QUERIES, PARAPHRASE_QUERIES, type EvalQuery } from './evalQueries.ts'
import { rank, VARIANTS, MIX_WEIGHTS_BY_DATASET, type RankContext, type Variant } from './retrieval.ts'
import { loadMemories, loadLjjMemories, loadPrecog, usageCounts, defaultNeuronPath } from './load.ts'
import { buildBlockVectors, embedQuery } from './embed.ts'
import type { Runtime } from './recall.ts'

export interface VariantScore {
  variant: Variant
  n: number
  hit1: number
  hit3: number
  hit5: number
  mrr: number
  avgMs: number
  misses: Array<{ query: string; got: string[] }>
}

export interface EvalReport {
  exact: VariantScore[]
  paraphrase: VariantScore[]
  combined: VariantScore[]
}

/** LJJ 查询集含个人信息，不入公开仓库；本地缺失时退化为空集。 */
async function loadLjjQuerySets(): Promise<{ exact: EvalQuery[]; paraphrase: EvalQuery[] }> {
  try {
    const m = await import('./evalQueriesLjj.ts')
    return { exact: m.LJJ_EXACT ?? [], paraphrase: m.LJJ_PARAPHRASE ?? [] }
  } catch {
    return { exact: [], paraphrase: [] }
  }
}

async function evalSet(rt: Runtime, queries: EvalQuery[]): Promise<VariantScore[]> {
  if (!queries.length) return []
  const ctxBase = {
    entries: rt.entries,
    vectors: rt.vectors,
    usage: rt.usage,
    mixWeights: MIX_WEIGHTS_BY_DATASET[rt.dataset],
  } as Omit<RankContext, 'qVec'>
  const out: VariantScore[] = []
  for (const variant of VARIANTS) {
    let hit1 = 0
    let hit3 = 0
    let hit5 = 0
    let mrr = 0
    let totalMs = 0
    const misses: Array<{ query: string; got: string[] }> = []
    for (const q of queries) {
      const t0 = performance.now()
      const qVec = await embedQuery(q.query)
      const rows = rank({ ...ctxBase, qVec } as RankContext, q.query, variant, 5)
      totalMs += performance.now() - t0
      const ids = rows.map((r) => r.id)
      const pos = Math.min(...q.expected.map((e) => ids.indexOf(e)).filter((i) => i >= 0), Infinity)
      if (pos === 0) hit1++
      if (pos < 3) hit3++
      if (pos < 5) hit5++
      if (Number.isFinite(pos)) mrr += 1 / (pos + 1)
      else misses.push({ query: q.query, got: ids.slice(0, 3) })
    }
    const n = queries.length
    out.push({
      variant,
      n,
      hit1: hit1 / n,
      hit3: hit3 / n,
      hit5: hit5 / n,
      mrr: mrr / n,
      avgMs: totalMs / n,
      misses,
    })
  }
  return out.sort((a, b) => b.mrr - a.mrr || b.hit1 - a.hit1)
}

export async function runEval(rt: Runtime): Promise<EvalReport> {
  const ljj = rt.dataset === 'ljj' ? await loadLjjQuerySets() : null
  const exact = ljj?.exact ?? EVAL_QUERIES
  const paraphrase = ljj?.paraphrase ?? PARAPHRASE_QUERIES
  return {
    exact: await evalSet(rt, exact),
    paraphrase: await evalSet(rt, paraphrase),
    combined: await evalSet(rt, [...exact, ...paraphrase]),
  }
}

function table(title: string, scores: VariantScore[]): string[] {
  const pct = (x: number) => `${(x * 100).toFixed(1)}%`
  const lines = [title, 'variant        hit@1   hit@3   hit@5   MRR     avg(ms)']
  for (const s of scores) {
    lines.push(
      `${s.variant.padEnd(14)} ${pct(s.hit1).padStart(6)}  ${pct(s.hit3).padStart(6)}  ${pct(s.hit5).padStart(6)}  ${s.mrr.toFixed(3).padStart(5)}  ${s.avgMs.toFixed(1).padStart(6)}`,
    )
  }
  return lines
}

export function formatEval(report: EvalReport, label = 'Pj11 项目记忆'): string {
  const lines = [
    `数据集: ${label}`,
    ...table(`== 精确措辞集（${report.exact[0]?.n ?? 0} 条） ==`, report.exact),
    '',
    ...table(`== 改写/口语集（${report.paraphrase[0]?.n ?? 0} 条） ==`, report.paraphrase),
    '',
    ...table(`== 合计（${report.combined[0]?.n ?? 0} 条） ==`, report.combined),
  ]
  const best = report.combined[0]
  if (best) {
    lines.push('')
    lines.push(`最佳: ${best.variant} (MRR=${best.mrr.toFixed(3)}, hit@1=${(best.hit1 * 100).toFixed(1)}%)`)
    if (best.misses.length) {
      lines.push(`未命中 ${best.misses.length} 条:`)
      for (const m of best.misses) lines.push(`  - ${m.query} → ${m.got.join(', ')}`)
    }
  }
  return lines.join('\n')
}

export async function loadEvalRuntime(
  neuronPath = defaultNeuronPath(),
  dataDir = '',
  dataset: Runtime['dataset'] = 'pj11',
): Promise<Runtime> {
  const entries = dataset === 'ljj' ? loadLjjMemories() : loadMemories(neuronPath)
  const { vectors } = await buildBlockVectors(entries, dataDir)
  return {
    dataset,
    neuronPath,
    dataDir,
    entries,
    vectors,
    usage: usageCounts(loadPrecog(neuronPath)),
    graph: { nodes: [], edges: [] },
    communities: { resolution: 1, modularity: 0, communities: [], edges: [] },
  }
}
