/**
 * Leiden 社区检测 — 纯 TS 完整实现（local move + refinement + 随机贪心加速）
 *
 * 来源：移植自 Pj16 神经元引擎 `Floria/src/tools/neturon/leiden.ts`（本项目 own code，
 * 零依赖、不 import 任何项目代码），供 GraphNeuron 的社群检测复用。
 *
 * 对照基线：Python `leidenalg.find_partition(g, RBConfigurationVertexPartition,
 *   weights=weight, resolution_parameter=γ)`（engine/core/cognition.py detect_communities 同款）。
 *
 * 算法（Traag, Waltman & van Eck 2019, "From Louvain to Leiden: guaranteeing
 * well-connected communities"）循环至稳定：
 *   1. Local Move  —— 队列加速的局部移动（随机贪心：种子随机初始序，节点移动后仅
 *                     把受影响邻居重新入队；增益并列时按种子随机择一）
 *   2. Refinement  —— 逐社群内重新细划：子簇只能并入与自身有**直接连边**的子簇
 *   3. Aggregation —— 按精化结果聚合成超节点图进入下一轮（社群内边权 → 自环）
 *
 * 确定性：mulberry32 种子随机（默认 seed=0），同种子同图结果逐字节一致。
 */

// ───────────────────────── 对外类型 ─────────────────────────

/** 加权无向边（节点用 0..nodeCount-1 下标；重复边自动合并求和；自环/非正权重丢弃并计数） */
export interface LeidenEdgeInput {
  source: number
  target: number
  weight: number
}

/** 单个分辨率下的划分结果 */
export interface LeidenPartition {
  /** 分辨率 γ */
  resolution: number
  /** 节点下标 → 社群号（0..nCommunities-1，按最小节点下标排序重编，确定性） */
  membership: Int32Array
  /** 社群数（孤立点各成单点社群） */
  nCommunities: number
  /** 该划分在原始图上的模块度 Q_γ（直算，非优化过程累积值） */
  modularity: number
}

export interface LeidenOptions {
  /** 分辨率列表，默认 [0.5, 0.8, 1.0, 1.5, 2.0]（对应 config.yaml community.resolutions） */
  resolutions?: number[]
  /** 随机种子（默认 0；同种子同图结果一致） */
  seed?: number
  /** 聚合层安全上限（默认 32；算法理论上必终止，此为兜底） */
  maxLevels?: number
}

export interface LeidenBuildStats {
  nodeCount: number
  /** 去重合并后的有效边数 */
  edgeCount: number
  droppedSelfLoops: number
  droppedNonPositive: number
  /** W = Σk_i */
  totalWeight: number
}

export interface LeidenResult {
  partitions: LeidenPartition[]
  stats: LeidenBuildStats
}

// ───────────────────────── 内部：CSR 图 ─────────────────────────

interface EdgeRec {
  a: number
  b: number
  w: number
}

/** CSR 加权无向图。自环不入邻接表，权重记 self[]（计入强度 k 与所属社群内部权重）。 */
interface Csr {
  n: number
  W: number
  k: Float64Array
  start: Int32Array
  nbr: Int32Array
  nbrW: Float64Array
  self: Float64Array
}

function csrFromEdges(n: number, edgeList: EdgeRec[], k: Float64Array, self: Float64Array): Csr {
  const deg = new Int32Array(n)
  for (const e of edgeList) {
    deg[e.a]++
    deg[e.b]++
  }
  const start = new Int32Array(n + 1)
  for (let i = 0; i < n; i++) start[i + 1] = start[i] + deg[i]
  const nbr = new Int32Array(start[n])
  const nbrW = new Float64Array(start[n])
  const cursor = Int32Array.from(start.subarray(0, n))
  for (const e of edgeList) {
    nbr[cursor[e.a]] = e.b
    nbrW[cursor[e.a]] = e.w
    cursor[e.a]++
    nbr[cursor[e.b]] = e.a
    nbrW[cursor[e.b]] = e.w
    cursor[e.b]++
  }
  let W = 0
  for (let i = 0; i < n; i++) W += k[i]
  return { n, W, k, start, nbr, nbrW, self }
}

function buildCsr(
  nodeCount: number,
  edges: LeidenEdgeInput[],
): { csr: Csr; edgeList: EdgeRec[]; stats: LeidenBuildStats } {
  if (!Number.isInteger(nodeCount) || nodeCount < 0) {
    throw new Error(`Leiden: nodeCount 非法: ${nodeCount}`)
  }
  const n = nodeCount
  const agg = new Map<number, number>()
  let droppedSelfLoops = 0
  let droppedNonPositive = 0
  for (const e of edges) {
    if (!(e.weight > 0)) {
      droppedNonPositive++
      continue
    }
    if (e.source === e.target) {
      droppedSelfLoops++
      continue
    }
    if (
      !Number.isInteger(e.source) ||
      !Number.isInteger(e.target) ||
      e.source < 0 ||
      e.source >= n ||
      e.target < 0 ||
      e.target >= n
    ) {
      throw new Error(`Leiden: 边端点越界 [${e.source}, ${e.target}]，nodeCount=${n}`)
    }
    const a = Math.min(e.source, e.target)
    const b = Math.max(e.source, e.target)
    const key = a * n + b
    agg.set(key, (agg.get(key) ?? 0) + e.weight)
  }
  const edgeList: EdgeRec[] = []
  for (const [key, w] of agg) {
    const a = Math.floor(key / n)
    edgeList.push({ a, b: key - a * n, w })
  }
  edgeList.sort((x, y) => x.a - y.a || x.b - y.b) // 确定性
  const k = new Float64Array(n)
  for (const e of edgeList) {
    k[e.a] += e.w
    k[e.b] += e.w
  }
  const csr = csrFromEdges(n, edgeList, k, new Float64Array(n))
  const stats: LeidenBuildStats = {
    nodeCount: n,
    edgeCount: edgeList.length,
    droppedSelfLoops,
    droppedNonPositive,
    totalWeight: csr.W,
  }
  return { csr, edgeList, stats }
}

// ───────────────────────── 内部：划分状态 ─────────────────────────

/**
 * 划分状态：社群号空间 0..n-1（初始 singleton = 自身下标，社群合并不产生新号）。
 * cStrength[c] = Σk_i（成员总强度，含各自环）；cInternal[c] = 社群内边权和
 * （成员间边 + 成员自环，各计一次）。移动时按 ΔQ 公式精确维护。
 */
class Partition {
  readonly n: number
  readonly comm: Int32Array
  readonly cStrength: Float64Array
  readonly cInternal: Float64Array
  readonly cSize: Int32Array
  private readonly self: Float64Array
  private alive: number

  constructor(self?: Float64Array) {
    this.n = self.length
    this.comm = new Int32Array(this.n)
    this.cStrength = new Float64Array(this.n)
    this.cInternal = Float64Array.from(self)
    this.cSize = new Int32Array(this.n).fill(1)
    this.self = self
    this.alive = this.n
    for (let v = 0; v < this.n; v++) this.comm[v] = v
  }

  static singletons(csr: Csr): Partition {
    const p = new Partition(csr.self)
    p.cStrength.set(csr.k)
    return p
  }

  get count(): number {
    return this.alive
  }

  /** v 从当前社群移入 to（to ≠ 当前社群）；kv = v 的总强度（含自环）。 */
  move(v: number, kvA: number, kvB: number, kv: number, to: number): void {
    const from = this.comm[v]
    if (from === to) return
    this.cStrength[from] -= kv
    this.cStrength[to] += kv
    this.cInternal[from] -= kvA + this.self[v]
    this.cInternal[to] += kvB + this.self[v]
    if (--this.cSize[from] === 0) this.alive--
    this.cSize[to]++
    this.comm[v] = to
  }

  /** 分组等价（标签无关）：存在一致的社群 ↔ 社群映射即等价。 */
  sameGrouping(other: Partition): boolean {
    const map = new Map<number, number>()
    for (let v = 0; v < this.n; v++) {
      const a = this.comm[v]
      const b = other.comm[v]
      const prev = map.get(a)
      if (prev === undefined) map.set(a, b)
      else if (prev !== b) return false
    }
    return true
  }
}

// ───────────────────────── 内部：种子随机 ─────────────────────────

function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function shuffle(arr: number[], rng: () => number): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    const t = arr[i]
    arr[i] = arr[j]
    arr[j] = t
  }
}

// ───────────────────────── 阶段 1：Local Move（队列加速随机贪心） ─────────────────────────

const GAIN_EPS = 1e-12

/** 对 v 的邻居按社群分桶累积边权；返回触碰到的社群列表。 */
function bucketNeighbourComms(
  csr: Csr,
  v: number,
  commOf: (u: number) => number,
  bucketW: Float64Array,
  bucketStamp: Int32Array,
  stamp: number,
  touched: number[],
): void {
  touched.length = 0
  const { start, nbr, nbrW } = csr
  for (let i = start[v]; i < start[v + 1]; i++) {
    const c = commOf(nbr[i])
    if (bucketStamp[c] !== stamp) {
      bucketStamp[c] = stamp
      bucketW[c] = 0
      touched.push(c)
    }
    bucketW[c] += nbrW[i]
  }
}

/** Local Move：队列加速随机贪心（节点移动后仅重排受影响邻居）。返回移动次数。 */
function localMove(csr: Csr, part: Partition, gamma: number, rng: () => number): number {
  const { n, W, k, start, nbr } = csr
  if (W <= 0) return 0
  const order: number[] = new Array(n)
  for (let i = 0; i < n; i++) order[i] = i
  shuffle(order, rng)
  const queue = order.slice()
  const inQueue = new Uint8Array(n).fill(1)
  const bucketW = new Float64Array(n)
  const bucketStamp = new Int32Array(n)
  const touched: number[] = []
  const bests: number[] = []
  let stamp = 0
  let head = 0
  let moves = 0
  const W2 = W * W
  while (head < queue.length) {
    const v = queue[head++]
    inQueue[v] = 0
    const own = part.comm[v]
    stamp++
    bucketNeighbourComms(csr, v, (u) => part.comm[u], bucketW, bucketStamp, stamp, touched)
    const kv = k[v]
    const kvA = bucketStamp[own] === stamp ? bucketW[own] : 0
    const DA = part.cStrength[own]
    let bestGain = GAIN_EPS
    bests.length = 0
    for (const c of touched) {
      if (c === own) continue
      const gain = (2 * (bucketW[c] - kvA)) / W - (gamma * 2 * kv * (part.cStrength[c] - DA + kv)) / W2
      if (gain <= GAIN_EPS) continue
      if (bests.length === 0 || gain > bestGain + GAIN_EPS) {
        bestGain = gain
        bests.length = 0
        bests.push(c)
      } else if (gain >= bestGain - GAIN_EPS) {
        bests.push(c)
      }
    }
    if (bests.length > 0) {
      const to = bests.length === 1 ? bests[0] : bests[Math.floor(rng() * bests.length)]
      part.move(v, kvA, bucketW[to], kv, to)
      moves++
      for (let i = start[v]; i < start[v + 1]; i++) {
        const u = nbr[i]
        if (!inQueue[u]) {
          inQueue[u] = 1
          queue.push(u)
        }
      }
    }
  }
  return moves
}

// ───────────────────────── 阶段 2：Refinement（社群内细划，连通性约束） ─────────────────────────

function refinePartition(csr: Csr, part: Partition, gamma: number, rng: () => number): Partition {
  const { n, W, k, start, nbr } = csr
  if (W <= 0) return Partition.singletons(csr) // 无边图：无结构可分
  const sub = Partition.singletons(csr)
  const membersByComm = new Map<number, number[]>()
  for (let v = 0; v < n; v++) {
    const c = part.comm[v]
    let arr = membersByComm.get(c)
    if (!arr) {
      arr = []
      membersByComm.set(c, arr)
    }
    arr.push(v)
  }
  const bucketW = new Float64Array(n)
  const bucketStamp = new Int32Array(n)
  const touched: number[] = []
  const bests: number[] = []
  const queue: number[] = []
  let stamp = 0
  const W2 = W * W
  for (const members of membersByComm.values()) {
    if (members.length === 1) continue
    const C = part.comm[members[0]]
    shuffle(members, rng)
    queue.length = 0
    for (const v of members) queue.push(v)
    const inQueue = new Uint8Array(n)
    for (const v of members) inQueue[v] = 1
    let head = 0
    while (head < queue.length) {
      const v = queue[head++]
      inQueue[v] = 0
      const own = sub.comm[v]
      stamp++
      touched.length = 0
      for (let i = start[v]; i < start[v + 1]; i++) {
        const u = nbr[i]
        if (part.comm[u] !== C) continue // 只在社群 C 内部细划
        const sc = sub.comm[u]
        if (bucketStamp[sc] !== stamp) {
          bucketStamp[sc] = stamp
          bucketW[sc] = 0
          touched.push(sc)
        }
        bucketW[sc] += csr.nbrW[i]
      }
      const kv = k[v]
      const kvA = bucketStamp[own] === stamp ? bucketW[own] : 0
      const DA = sub.cStrength[own]
      let bestGain = GAIN_EPS
      bests.length = 0
      for (const c of touched) {
        if (c === own) continue
        const gain = (2 * (bucketW[c] - kvA)) / W - (gamma * 2 * kv * (sub.cStrength[c] - DA + kv)) / W2
        if (gain <= GAIN_EPS) continue
        if (bests.length === 0 || gain > bestGain + GAIN_EPS) {
          bestGain = gain
          bests.length = 0
          bests.push(c)
        } else if (gain >= bestGain - GAIN_EPS) {
          bests.push(c)
        }
      }
      if (bests.length > 0) {
        const to = bests.length === 1 ? bests[0] : bests[Math.floor(rng() * bests.length)]
        sub.move(v, kvA, bucketW[to], kv, to)
        for (let i = start[v]; i < start[v + 1]; i++) {
          const u = nbr[i]
          if (part.comm[u] === C && !inQueue[u]) {
            inQueue[u] = 1
            queue.push(u)
          }
        }
      }
    }
  }
  return sub
}

// ───────────────────────── 阶段 3：Aggregation ─────────────────────────

/**
 * 按给定划分聚合超节点图：超节点 = 划分社群（新号 0..m-1 按最小成员下标排序）；
 * 社群内边权 → 新节点自环（cInternal 直取，含成员自环）；跨社群边按对聚合。
 */
function aggregate(csr: Csr, part: Partition): { csr2: Csr; remap: Int32Array } {
  const { n, start, nbr, nbrW } = csr
  const remap = new Int32Array(n).fill(-1) // 旧社群号 → 新超节点号
  let m = 0
  for (let v = 0; v < n; v++) {
    const c = part.comm[v]
    if (remap[c] === -1) remap[c] = m++
  }
  const agg = new Map<number, number>()
  for (let v = 0; v < n; v++) {
    const cv = remap[part.comm[v]]
    for (let i = start[v]; i < start[v + 1]; i++) {
      const u = nbr[i]
      if (u <= v) continue // CSR 每条无向边出现两次，只取 u>v 一次
      const cu = remap[part.comm[u]]
      if (cu === cv) continue // 内部边 → 自环（走 cInternal）
      const a = Math.min(cv, cu)
      const b = Math.max(cv, cu)
      const key = a * m + b
      agg.set(key, (agg.get(key) ?? 0) + nbrW[i])
    }
  }
  const edgeList: EdgeRec[] = []
  for (const [key, w] of agg) {
    const a = Math.floor(key / m)
    edgeList.push({ a, b: key - a * m, w })
  }
  edgeList.sort((x, y) => x.a - y.a || x.b - y.b)
  const k2 = new Float64Array(m)
  const self2 = new Float64Array(m)
  const seen = new Uint8Array(n)
  for (let v = 0; v < n; v++) {
    const c = part.comm[v]
    if (seen[c]) continue
    seen[c] = 1
    k2[remap[c]] = part.cStrength[c]
    self2[remap[c]] = part.cInternal[c]
  }
  return { csr2: csrFromEdges(m, edgeList, k2, self2), remap }
}

// ───────────────────────── 驱动与输出 ─────────────────────────

function modularityOf(
  edgeList: EdgeRec[],
  k: Float64Array,
  W: number,
  n: number,
  membership: Int32Array,
  gamma: number,
): number {
  if (W <= 0) return 0
  const L = new Float64Array(n)
  const D = new Float64Array(n)
  for (const e of edgeList) {
    if (membership[e.a] === membership[e.b]) L[membership[e.a]] += e.w
  }
  for (let v = 0; v < n; v++) D[membership[v]] += k[v]
  let q = 0
  for (let c = 0; c < n; c++) {
    if (D[c] === 0) continue
    q += (2 * L[c]) / W - gamma * (D[c] / W) ** 2
  }
  return q
}

/** 社群号按最小节点下标排序重编（确定性输出）。 */
function relabelMembership(membership: Int32Array): { membership: Int32Array; nCommunities: number } {
  const remap = new Map<number, number>()
  for (let v = 0; v < membership.length; v++) {
    const c = membership[v]
    if (!remap.has(c)) remap.set(c, remap.size)
  }
  const out = new Int32Array(membership.length)
  for (let v = 0; v < membership.length; v++) out[v] = remap.get(membership[v])!
  return { membership: out, nCommunities: remap.size }
}

/** 单分辨率完整 Leiden：多轮（local move → refine → aggregate）至稳定，返回原始节点划分。 */
function leidenOnce(csr0: Csr, gamma: number, seed: number, maxLevels: number): Int32Array {
  const rng = mulberry32(seed)
  let csr = csr0
  let part = Partition.singletons(csr)
  const nodeMap = new Int32Array(csr0.n)
  for (let i = 0; i < csr0.n; i++) nodeMap[i] = i
  for (let level = 0; level < maxLevels; level++) {
    const moves = localMove(csr, part, gamma, rng)
    const refined = refinePartition(csr, part, gamma, rng)
    const changed = !refined.sameGrouping(part)
    if (moves === 0 && !changed) break
    if (level === maxLevels - 1) break
    const { csr2, remap } = aggregate(csr, refined)
    for (let i = 0; i < nodeMap.length; i++) nodeMap[i] = remap[refined.comm[nodeMap[i]]]
    csr = csr2
    part = Partition.singletons(csr)
  }
  const out = new Int32Array(csr0.n)
  for (let i = 0; i < csr0.n; i++) out[i] = part.comm[nodeMap[i]]
  return out
}

/**
 * 多分辨率 Leiden 社区检测入口。
 *
 * @param nodeCount 节点数（节点用 0..nodeCount-1 下标）
 * @param edges     加权无向边（重复边合并求和；自环/非正权重丢弃计数）
 * @param options   resolutions/seed/maxLevels
 */
export function leidenCommunities(
  nodeCount: number,
  edges: LeidenEdgeInput[],
  options: LeidenOptions = {},
): LeidenResult {
  const resolutions = options.resolutions ?? [0.5, 0.8, 1.0, 1.5, 2.0]
  const seed = options.seed ?? 0
  const maxLevels = options.maxLevels ?? 32
  if (!Array.isArray(resolutions) || resolutions.length === 0) {
    throw new Error('Leiden: resolutions 不能为空')
  }
  for (const g of resolutions) {
    if (!(typeof g === 'number' && Number.isFinite(g) && g > 0)) {
      throw new Error(`Leiden: 分辨率须为正有限数，收到 ${g}`)
    }
  }
  const { csr, edgeList, stats } = buildCsr(nodeCount, edges)
  const partitions: LeidenPartition[] = []
  for (const gamma of resolutions) {
    const raw = leidenOnce(csr, gamma, seed, maxLevels)
    const { membership, nCommunities } = relabelMembership(raw)
    partitions.push({
      resolution: gamma,
      membership,
      nCommunities,
      modularity: modularityOf(edgeList, csr.k, csr.W, csr.n, membership, gamma),
    })
  }
  return { partitions, stats }
}
