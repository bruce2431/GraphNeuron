/**
 * GraphNeuron — 公共入口
 *
 * 记忆 = 事件的图；检索 = 种子匹配 + 图蔓延 + 子图诱导。
 * 见 SubPj9 [20261009130500]-GraphNeuron设计-v0.2.md。
 */

export * from './schema.ts'
export * from './embed.ts'
export { GraphNeuron, timeFromId, type GNOptions } from './store.ts'
export { search, type QueryPattern, type SearchOptions, type SearchResult, type ScoredEvent } from './retrieve.ts'
export {
  detectCommunities,
  type Community,
  type CommunityResult,
  type CommunityOptions,
  type CommunityMember,
} from './community.ts'
export { leidenCommunities, type LeidenEdgeInput, type LeidenResult, type LeidenPartition } from './leiden.ts'
