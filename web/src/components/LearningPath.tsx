"use client";

import { motion } from "framer-motion";
import Link from "next/link";

type Lesson = {
  id: string;
  title: string;
  quote: string;
  duration: string;
};

type Phase = {
  key: string;
  label: string;
  range: string;
  theme: string;
  description: string;
  lessons: Lesson[];
};

const PHASES: Phase[] = [
  {
    key: "p1",
    label: "Phase 1",
    range: "L01–L04",
    theme: "#10b981",
    description: "零基础入门",
    lessons: [
      {
        id: "L01",
        title: "什么是大语言模型",
        quote: "大道至简，从文字接龙说起",
        duration: "30min",
      },
      {
        id: "L02",
        title: "Transformer全景图",
        quote: "注意力就是一切",
        duration: "45min",
      },
      {
        id: "L03",
        title: "PyTorch快速上手",
        quote: "工欲善其事，必先利其器",
        duration: "60min",
      },
      {
        id: "L04",
        title: "MiniMind项目导览",
        quote: "千里之行，始于配环境",
        duration: "30min",
      },
    ],
  },
  {
    key: "p2",
    label: "Phase 2",
    range: "L05–L10",
    theme: "#3b82f6",
    description: "模型核心组件",
    lessons: [
      {
        id: "L05",
        title: "Tokenizer分词器",
        quote: "模型的第一本字典",
        duration: "45min",
      },
      {
        id: "L06",
        title: "词嵌入Embedding",
        quote: "把文字变成数字的魔法",
        duration: "30min",
      },
      {
        id: "L07",
        title: "RMSNorm归一化",
        quote: "训练稳定的守护者",
        duration: "40min",
      },
      {
        id: "L08",
        title: "RoPE旋转位置编码",
        quote: "让模型知道谁先谁后",
        duration: "50min",
      },
      {
        id: "L09",
        title: "注意力机制与GQA",
        quote: "每个词都在关注其他词",
        duration: "60min",
      },
      {
        id: "L10",
        title: "前馈网络与SwiGLU",
        quote: "知识的仓库，智慧的门控",
        duration: "40min",
      },
    ],
  },
  {
    key: "p3",
    label: "Phase 3",
    range: "L11–L16",
    theme: "#8b5cf6",
    description: "训练全流程",
    lessons: [
      {
        id: "L11",
        title: "数据处理流水线",
        quote: "数据是模型的食粮",
        duration: "45min",
      },
      {
        id: "L12",
        title: "预训练Pretrain",
        quote: "让模型学会词语接龙",
        duration: "60min",
      },
      {
        id: "L13",
        title: "监督微调SFT",
        quote: "从百科全书到对话助手",
        duration: "50min",
      },
      {
        id: "L14",
        title: "LoRA高效微调",
        quote: "四两拨千斤的微调艺术",
        duration: "50min",
      },
      {
        id: "L15",
        title: "知识蒸馏",
        quote: "青出于蓝而胜于蓝",
        duration: "40min",
      },
      {
        id: "L16",
        title: "完整模型组装",
        quote: "乐高积木拼出飞机",
        duration: "60min",
      },
    ],
  },
  {
    key: "p4",
    label: "Phase 4",
    range: "L17–L22",
    theme: "#f59e0b",
    description: "高级特性与面试",
    lessons: [
      {
        id: "L17",
        title: "DPO偏好优化",
        quote: "教模型分辨好与坏",
        duration: "50min",
      },
      {
        id: "L18",
        title: "PPO与GRPO强化学习",
        quote: "在试错中持续进化",
        duration: "60min",
      },
      {
        id: "L19",
        title: "MoE混合专家模型",
        quote: "术业有专攻",
        duration: "50min",
      },
      {
        id: "L20",
        title: "推理优化",
        quote: "又快又好地生成",
        duration: "45min",
      },
      {
        id: "L21",
        title: "部署与应用",
        quote: "从实验室走向生产",
        duration: "45min",
      },
      {
        id: "L22",
        title: "面试通关指南",
        quote: "把项目写进简历，拿下Offer",
        duration: "90min",
      },
    ],
  },
];

const listVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.06, delayChildren: 0.05 },
  },
};

const cardVariants = {
  hidden: { opacity: 0, y: 28 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { type: "spring" as const, stiffness: 380, damping: 28 },
  },
};

export default function LearningPath() {
  return (
    <section className="mx-auto w-full max-w-5xl px-4 py-16 md:px-6">
      <header className="mb-12 text-center md:mb-14">
        <h2 className="text-3xl font-semibold tracking-tight text-zinc-100 md:text-4xl">
          学习路径
        </h2>
        <p className="mt-3 text-base text-zinc-400 md:text-lg">
          22节渐进式课程，从零基础到面试通关 —{" "}
          <span className="text-[#8b5cf6]">点击卡片</span>即可在网页阅读完整讲义
        </p>
      </header>

      <div className="flex flex-col gap-14 md:gap-16">
        {PHASES.map((phase) => (
          <div key={phase.key}>
            <div className="mb-6 flex flex-wrap items-end gap-3 border-b border-zinc-800/80 pb-4">
              <span
                className="rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wider text-white"
                style={{ backgroundColor: phase.theme }}
              >
                {phase.label}
              </span>
              <span className="text-sm text-zinc-500">{phase.range}</span>
              <h3
                className="ml-auto text-lg font-medium md:text-xl"
                style={{ color: phase.theme }}
              >
                {phase.description}
              </h3>
            </div>

            <motion.ul
              className="grid gap-4 sm:grid-cols-2"
              variants={listVariants}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true, margin: "-40px" }}
            >
              {phase.lessons.map((lesson) => (
                <motion.li key={lesson.id} variants={cardVariants} layout>
                  <Link href={`/lesson/${lesson.id}`} className="block">
                  <motion.article
                    className="group relative cursor-pointer overflow-hidden rounded-2xl border border-zinc-800/90 p-5 transition-colors"
                    style={{ backgroundColor: "#1a1a2e" }}
                    whileHover={{
                      y: -4,
                      boxShadow: `0 20px 40px -12px rgba(0,0,0,0.5), 0 0 0 1px ${phase.theme}55, 0 0 28px ${phase.theme}40`,
                    }}
                    transition={{ type: "spring", stiffness: 400, damping: 25 }}
                  >
                    <div
                      className="absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100"
                      style={{
                        background: `radial-gradient(600px circle at 50% -20%, ${phase.theme}22, transparent 55%)`,
                      }}
                    />
                    <div className="relative flex items-start justify-between gap-3">
                      <span
                        className="shrink-0 rounded-lg px-2.5 py-1 font-mono text-xs font-bold text-white"
                        style={{ backgroundColor: `${phase.theme}cc` }}
                      >
                        {lesson.id}
                      </span>
                      <span
                        className="rounded-md border px-2 py-0.5 text-xs text-zinc-400"
                        style={{ borderColor: `${phase.theme}44` }}
                      >
                        {lesson.duration}
                      </span>
                    </div>
                    <h4 className="relative mt-3 text-lg font-semibold text-zinc-100">
                      {lesson.title}
                    </h4>
                    <p className="relative mt-2 text-sm italic leading-relaxed text-zinc-500">
                      「{lesson.quote}」
                    </p>
                    <p
                      className="relative mt-3 text-xs font-medium"
                      style={{ color: phase.theme }}
                    >
                      点击阅读全文 →
                    </p>
                  </motion.article>
                  </Link>
                </motion.li>
              ))}
            </motion.ul>
          </div>
        ))}
      </div>
    </section>
  );
}
