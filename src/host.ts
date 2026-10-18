/**
 * 宿主能力借用（Pj16 Floria），本文件只做接口适配，不修改宿主。
 *
 *   LLM   : config.yaml 的 llm.provider/model/temperature/thinking
 *           + 全局 credentials.json providers.<provider>（密钥不入神经元配置）
 *   YAML  : 宿主 neturon/coggraph.ts 的 readYaml（单一实现）
 *
 * 宿主按工作区相对位置查找（逐级向上找 Pj16-.../Floria/src），找不到时抛错，
 * 由调用方跳过 LLM 阶段——发布到独立仓库后仍可正常 build/eval/recall。
 */

import { existsSync } from 'node:fs'
import { join } from 'node:path'

/** 与宿主 cogname 的 LlmCall 同口径：一次请求 → 纯文本回复。 */
export type LlmCall = (system: string, user: string) => Promise<string>

export interface HostLlm {
  provider: string
  model: string
  call: LlmCall
}

const LLM_TIMEOUT_MS = 120_000
const LLM_MAX_TOKENS = 2000

/** 逐级向上查找宿主 Floria/src 目录。 */
function findHostSrc(): string {
  let dir = import.meta.dir
  for (let i = 0; i < 10; i++) {
    const p = join(dir, 'Pj16-CodeAgent构建', 'Floria', 'src')
    if (existsSync(p)) return p
    dir = join(dir, '..')
  }
  throw new Error('未找到宿主 Floria（Pj16-CodeAgent构建/Floria/src）')
}

async function loadHostModules(): Promise<{
  readYaml: (p: string) => Record<string, unknown>
  loadCredentials: () => { providers: Record<string, any> }
  resolveProtocolFetch: (o: {
    protocol: string | null | undefined
    baseUrl: string | null
    apiKey: string | null
  }) => typeof globalThis.fetch | undefined
}> {
  const src = findHostSrc()
  const coggraph = await import(`${src}/tools/neturon/coggraph.ts`)
  const pool = await import(`${src}/utils/credentials/pool.ts`)
  const protocol = await import(`${src}/services/api/protocol-fetch.ts`)
  return {
    readYaml: coggraph.readYaml,
    loadCredentials: pool.loadCredentials,
    resolveProtocolFetch: protocol.resolveProtocolFetch,
  }
}

export async function readNeuronConfig(neuronPath: string): Promise<Record<string, unknown>> {
  const p = join(neuronPath, 'config.yaml')
  if (!existsSync(p)) throw new Error(`找不到神经元配置: ${p}`)
  const { readYaml } = await loadHostModules()
  return readYaml(p)
}

/** 从神经元 config.yaml 的 llm 段解析宿主 LLM 调用器（Anthropic 协议）。 */
export async function resolveHostLlm(neuronPath: string): Promise<HostLlm> {
  const { loadCredentials, resolveProtocolFetch } = await loadHostModules()
  const cfg = await readNeuronConfig(neuronPath)
  const llm = cfg.llm as Record<string, unknown> | undefined
  if (!llm) throw new Error(`config.yaml 缺 llm 段: ${join(neuronPath, 'config.yaml')}`)

  const providerName = String(llm.provider ?? '')
  const model = String(llm.model ?? '')
  if (!providerName) throw new Error('config.yaml 缺 llm.provider（凭据借宿主 credentials.json）')
  if (!model) throw new Error('config.yaml 缺 llm.model')

  const temperature = Number(llm.temperature ?? 0.3)
  const thinking = llm.thinking
  const creds = loadCredentials()
  const provider = creds.providers[providerName]
  if (!provider) throw new Error(`credentials.json 无 provider '${providerName}'`)
  const key = provider.keys[provider.activeKeyIndex ?? 0]?.value
  if (!key) throw new Error(`provider '${providerName}' 无可用 key`)
  const baseUrl = provider.baseUrl
  if (!baseUrl) throw new Error(`provider '${providerName}' 无 baseUrl`)

  const doFetch =
    resolveProtocolFetch({
      protocol: provider.protocol ?? 'anthropic',
      baseUrl,
      apiKey: key,
    }) ?? fetch

  return {
    provider: providerName,
    model,
    call: async (system, user) => {
      const resp = await doFetch(`${baseUrl}/v1/messages`, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'x-api-key': key,
          'anthropic-version': '2023-06-01',
        },
        signal: AbortSignal.timeout(LLM_TIMEOUT_MS),
        body: JSON.stringify({
          model,
          max_tokens: LLM_MAX_TOKENS,
          temperature,
          ...(thinking && typeof thinking === 'object' ? { thinking } : {}),
          system,
          messages: [{ role: 'user', content: user }],
        }),
      })
      if (!resp.ok) throw new Error(`LLM HTTP ${resp.status}: ${(await resp.text()).slice(0, 200)}`)
      const data = (await resp.json()) as { content?: Array<{ type: string; text?: string }> }
      const text = (data.content ?? [])
        .filter((b) => b.type === 'text' && b.text)
        .map((b) => b.text as string)
        .join('')
      if (!text.trim()) throw new Error('LLM 返回空内容')
      return text
    },
  }
}
