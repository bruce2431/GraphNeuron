/**
 * 认知层版本接口（与神经元既有 l1.cog/versions/vN + manifest.json 同构）。
 *
 * 宿主版本机制就绪前由本文件维护接口；就绪后只需替换 snapshot/restore 实现，
 * 调用方（feedback / 管线）不变。
 *
 * 约定：
 *   - 目录：<neuron>/l1.cog/versions/vN/
 *   - 文件：cog.json / cog_graph.json / community.json / cog2.json（存在才快照）
 *   - manifest：version / snapshot_at / snapshotted[] / params / lineage / meta
 *   - 只增不改：已存在的版本目录拒绝覆盖；恢复前把当前文件备份为 *.bak
 */

import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

export const COG_FILES = ['cog.json', 'cog_graph.json', 'community.json', 'cog2.json'] as const

export interface CogVersionManifest {
  version: string
  snapshot_at: string
  snapshotted: string[]
  params: Record<string, unknown>
  lineage: Record<string, string>
  [k: string]: unknown
}

export interface SnapshotOptions {
  params?: Record<string, unknown>
  lineage?: Record<string, string>
  meta?: Record<string, unknown>
}

function versionsDir(neuronPath: string): string {
  return join(neuronPath, 'l1.cog', 'versions')
}

function versionNum(name: string): number {
  const m = /^v(\d+)$/.exec(name)
  return m ? Number(m[1]) : -1
}

export function listCogVersions(neuronPath: string): string[] {
  const dir = versionsDir(neuronPath)
  if (!existsSync(dir)) return []
  return readdirSync(dir)
    .filter((n) => versionNum(n) >= 0)
    .sort((a, b) => versionNum(a) - versionNum(b))
}

function nowStamp(): string {
  const d = new Date()
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}

export function snapshotCogVersion(
  neuronPath: string,
  opts: SnapshotOptions = {},
): { version: string; dir: string; manifest: CogVersionManifest } {
  const existing = listCogVersions(neuronPath)
  const next = existing.length ? `v${versionNum(existing[existing.length - 1]!) + 1}` : 'v0'
  const dir = join(versionsDir(neuronPath), next)
  mkdirSync(dir, { recursive: true })

  const snapshotted: string[] = []
  for (const f of COG_FILES) {
    const src = join(neuronPath, 'l1.cog', f)
    if (!existsSync(src)) continue
    copyFileSync(src, join(dir, f))
    snapshotted.push(f)
  }
  if (!snapshotted.length) throw new Error(`l1.cog 无可快照文件: ${join(neuronPath, 'l1.cog')}`)

  const manifest: CogVersionManifest = {
    version: next,
    snapshot_at: nowStamp(),
    snapshotted,
    params: opts.params ?? {},
    lineage: opts.lineage ?? {},
    ...(opts.meta ?? {}),
  }
  writeFileSync(join(dir, 'manifest.json'), JSON.stringify(manifest, null, 2))
  return { version: next, dir, manifest }
}

export function readCogVersionManifest(neuronPath: string, version: string): CogVersionManifest | null {
  const p = join(versionsDir(neuronPath), version, 'manifest.json')
  if (!existsSync(p)) return null
  try {
    return JSON.parse(readFileSync(p, 'utf-8')) as CogVersionManifest
  } catch {
    return null
  }
}

/** 恢复某个版本到 l1.cog（当前文件先备份 *.bak）；不删除更新版本。 */
export function restoreCogVersion(neuronPath: string, version: string): string[] {
  const dir = join(versionsDir(neuronPath), version)
  if (!existsSync(dir)) throw new Error(`版本不存在: ${version}`)
  const restored: string[] = []
  for (const f of COG_FILES) {
    const src = join(dir, f)
    if (!existsSync(src)) continue
    const dst = join(neuronPath, 'l1.cog', f)
    if (existsSync(dst)) copyFileSync(dst, `${dst}.bak`)
    copyFileSync(src, dst)
    restored.push(f)
  }
  if (!restored.length) throw new Error(`版本 ${version} 无认知文件可恢复`)
  return restored
}
