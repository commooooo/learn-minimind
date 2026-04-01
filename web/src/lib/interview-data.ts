/** 面试页 slug → interview/ 下文件名 */
export const INTERVIEW_FILES: Record<string, string> = {
  "01": "01-项目介绍话术.md",
  "02": "02-模型架构面试题.md",
  "03": "03-训练流程面试题.md",
  "04": "04-优化与部署面试题.md",
  "05": "05-综合追问与深挖题.md",
};

export const INTERVIEW_META: Record<
  string,
  { title: string; subtitle: string }
> = {
  "01": { title: "项目介绍话术", subtitle: "30秒 / 1分钟 / 3分钟版本" },
  "02": { title: "模型架构面试题", subtitle: "Transformer · GQA · RoPE · …" },
  "03": { title: "训练流程面试题", subtitle: "预训练 · SFT · LoRA · DPO · …" },
  "04": { title: "优化与部署面试题", subtitle: "KV-Cache · MoE · 推理 · …" },
  "05": { title: "综合追问与深挖题", subtitle: "手写代码 · 场景题 · …" },
};

/** 固定顺序：上一篇 / 下一篇 */
export const INTERVIEW_ORDER = ["01", "02", "03", "04", "05"] as const;
export const INTERVIEW_SLUGS = [...INTERVIEW_ORDER];
