/** 权重网格搜索：dense / bm25 / kw / RRF 的组合，按查询集 MRR 排序。 */

import { EVAL_QUERIES, PARAPHRASE_QUERIES, type EvalQuery } from './evalQueries.ts'
import { scoreAll, type RankContext } from './retrieval.ts'
import { loadMemories, loadLjjMemories, loadPrecog, usageCounts, defaultNeuronPath } from './load.ts'
import { buildBlockVectors, embedQuery } from './embed.ts'
import { defaultDataDir, type Dataset } from './recall.ts'

const dataset = ((process.argv.includes('--dataset') ? process.argv[process.argv.indexOf('--dataset') + 1] : 'pj11') ?? 'pj11') as Dataset

// LJJ 查询集含个人信息，不入公开仓库；本地缺失时只调 Pj11 集。
let ljjQueries: EvalQuery[] = []
if (dataset === 'ljj') {
  try {
    const m = await import('./evalQueriesLjj.ts')
    ljjQueries = [...(m.LJJ_EXACT ?? []), ...(m.LJJ_PARAPHRASE ?? [])]
  } catch {
    console.warn('未找到 evalQueriesLjj.ts（个人信息查询集不入公开仓库）')
  }
}
const queries = dataset === 'ljj' ? ljjQueries : [...EVAL_QUERIES, ...PARAPHRASE_QUERIES]

interface RowScore {
  id: string
  cos: number
  bm25: number
  kw: number
  recency: number
}

function norm(rows: RowScore[], get: (r: RowScore) => number): Map<string, number> {
  let max = 0
  for (const r of rows) max = Math.max(max, get(r))
  return new Map(rows.map((r) => [r.id, max > 0 ? get(r) / max : 0]))
}

function evalCombo(
  perQuery: Array<{ expected: string[]; rows: RowScore[]; denseN: Map<string, number>; bm25N: Map<string, number>; kwN: Map<string, number> }>,
  w: { dense: number; bm25: number; kw: number },
): { mrr: number; hit1: number; hit3: number } {
  let mrr = 0
  let hit1 = 0
  let hit3 = 0
  for (const q of perQuery) {
    const scored = q.rows.map((r) => ({
      id: r.id,
      s: w.dense * (q.denseN.get(r.id) ?? 0) + w.bm25 * (q.bm25N.get(r.id) ?? 0) + w.kw * (q.kwN.get(r.id) ?? 0),
    })).sort((a, b) => b.s - a.s)
    const ids = scored.map((x) => x.id)
    const pos = Math.min(...q.expected.map((e) => ids.indexOf(e)).filter((i) => i >= 0), Infinity)
    if (pos === 0) hit1++
    if (pos < 3) hit3++
    if (Number.isFinite(pos)) mrr += 1 / (pos + 1)
  }
  const n = perQuery.length
  return { mrr: mrr / n, hit1: hit1 / n, hit3: hit3 / n }
}

const rt = {
  entries: dataset === 'ljj' ? loadLjjMemories() : loadMemories(defaultNeuronPath()),
  vectors: new Map<string, Float32Array[]>(),
  usage: dataset === 'ljj' ? new Map<string, number>() : usageCounts(loadPrecog(defaultNeuronPath())),
}
const dataDir = defaultDataDir(dataset)
const { vectors } = await buildBlockVectors(rt.entries, dataDir)
rt.vectors = vectors

const perQuery = []
for (const q of queries) {
  const qVec = await embedQuery(q.query)
  const rows = scoreAll({ ...rt, qVec } as RankContext, q.query).map((r) => ({
    id: r.entry.id,
    cos: r.cos,
    bm25: r.bm25,
    kw: r.kw,
    recency: r.recency,
  }))
  perQuery.push({
    expected: q.expected,
    rows,
    denseN: norm(rows, (r) => r.cos),
    bm25N: norm(rows, (r) => r.bm25),
    kwN: norm(rows, (r) => r.kw),
  })
}

const results: Array<{ w: { dense: number; bm25: number; kw: number }; mrr: number; hit1: number; hit3: number }> = []
for (let d = 0; d <= 10; d++) {
  for (let b = 0; b <= 10; b++) {
    for (let k = 0; k <= 10; k++) {
      if (d + b + k !== 10) continue
      const w = { dense: d / 10, bm25: b / 10, kw: k / 10 }
      results.push({ w, ...evalCombo(perQuery, w) })
    }
  }
}
results.sort((a, b) => b.mrr - a.mrr || b.hit1 - a.hit1)
for (const r of results.slice(0, 12)) {
  console.log(`dense=${r.w.dense.toFixed(1)} bm25=${r.w.bm25.toFixed(1)} kw=${r.w.kw.toFixed(1)}  MRR=${r.mrr.toFixed(3)} hit@1=${(r.hit1 * 100).toFixed(1)}% hit@3=${(r.hit3 * 100).toFixed(1)}%`)
}
