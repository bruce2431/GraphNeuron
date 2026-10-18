/** 检索对比实验的查询集（人工按库内容构造；expected 允许多个等价命中） */

export interface EvalQuery {
  query: string
  expected: string[]
}

export const EVAL_QUERIES: EvalQuery[] = [
  { query: 'GraphNeuron 事件图引擎在哪里、怎么启动 API', expected: ['PJ11_MEM_20261009_132702'] },
  { query: '归档 zip 里重复 raw 清理省了多少空间', expected: ['PJ11_MEM_20261009_132827'] },
  { query: '李京瑾数据集 22502 个事件是怎么转出来的', expected: ['PJ11_MEM_20261009_134529'] },
  { query: '事件图可视化为什么要排除 char 边', expected: ['PJ11_MEM_20261009_140254'] },
  { query: '预览里的图画完漂到右下角，取景 bug 怎么修', expected: ['PJ11_MEM_20261009_140942'] },
  { query: 'work 模式右栏预览一直显示默认页，根因是什么', expected: ['PJ11_MEM_20261009_142742'] },
  { query: '预览页不能在页面里再画一圈卡片', expected: ['PJ11_MEM_20261009_143221'] },
  { query: '用 LLM 把聊天抽取成五维事件 schema', expected: ['PJ11_MEM_20261009_145230'] },
  { query: '事件向量怎么用结构指纹和 hash 生成', expected: ['PJ11_MEM_20261009_145230'] },
  { query: '人物别名归一，马天越和李京瑾两套昵称', expected: ['PJ11_MEM_20261009_195514'] },
  { query: '嵌套事件垃圾的根因和 NESTING_MOVES 白名单', expected: ['PJ11_MEM_20261009_195514'] },
  { query: 'd3-force 替换手写 FR 力导做事件图预览', expected: ['PJ11_MEM_20261009_202021'] },
  { query: 'LoRAMoE 社群专家 每个社群训练成 LoRA 专家', expected: ['PJ11_MEM_20260904_132127'] },
  { query: 'canvas 缺 DPR 坐标系导致放大旋转成乱麻', expected: ['PJ11_MEM_20260902_183630'] },
  { query: '拖拽产生拖影，合成泄漏怎么修', expected: ['PJ11_MEM_20260902_173046'] },
  { query: '手机远程访问预览页一直正在加载', expected: ['PJ11_MEM_20260902_171438'] },
  { query: 'viz 重皮成 Floria 黑白简约风', expected: ['PJ11_MEM_20260902_174345'] },
  { query: '预览页 file:// 打开被 CORS 拦截 fetch', expected: ['PJ11_MEM_20260902_152610'] },
]

/** 改写/口语化查询集：不复用原文措辞，用来暴露纯字面匹配的短板。 */
export const PARAPHRASE_QUERIES: EvalQuery[] = [
  { query: '图记忆引擎怎么启动，监听哪个端口', expected: ['PJ11_MEM_20261009_132702'] },
  { query: '清理压缩包里重复的原始数据，节省了多少 G', expected: ['PJ11_MEM_20261009_132827'] },
  { query: '把聊天记录转成事件图语料，一共多少条事件', expected: ['PJ11_MEM_20261009_134529'] },
  { query: '为什么共享人物会把事件连成一团，可视化要丢掉哪类边', expected: ['PJ11_MEM_20261009_140254'] },
  { query: '力导向布局收敛后画面跑到右下角怎么办', expected: ['PJ11_MEM_20261009_140942'] },
  { query: '右栏预览不显示项目页面，只显示默认页', expected: ['PJ11_MEM_20261009_142742'] },
  { query: '预览页里再套卡片导致边框和圆角重复', expected: ['PJ11_MEM_20261009_143221'] },
  { query: '让大模型从聊天里抽时间、人物、动作', expected: ['PJ11_MEM_20261009_145230'] },
  { query: '事件用结构特征加文本散列做向量', expected: ['PJ11_MEM_20261009_145230'] },
  { query: '同一个人的多个昵称怎么识别成一个人', expected: ['PJ11_MEM_20261009_195514'] },
  { query: '父事件里塞了不相关的子事件，怎么用动词白名单限制', expected: ['PJ11_MEM_20261009_195514'] },
  { query: '用 d3 的力导向库替代自己写的布局', expected: ['PJ11_MEM_20261009_202021'] },
  { query: '把每个认知社群训练成一个 LoRA 专家', expected: ['PJ11_MEM_20260904_132127'] },
  { query: '画布没有按设备像素比设置坐标变换，放大后糊成一团', expected: ['PJ11_MEM_20260902_183630'] },
  { query: '拖动节点留下残影，浏览器合成层溢出', expected: ['PJ11_MEM_20260902_173046'] },
  { query: '用手机打开预览一直转圈加载不出来', expected: ['PJ11_MEM_20260902_171438'] },
  { query: '把可视化界面改成黑白灰加一个蓝色', expected: ['PJ11_MEM_20260902_174345'] },
  { query: '本地双击网页读不到数据，因为跨域限制', expected: ['PJ11_MEM_20260902_152610'] },
  { query: '事件五维结构里内容槽可以嵌套另一件事吗', expected: ['PJ11_MEM_20261009_132702'] },
  { query: '这个引擎对外提供哪些 HTTP 接口', expected: ['PJ11_MEM_20261009_132702'] },
]
