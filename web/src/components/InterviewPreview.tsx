"use client";

import { motion, AnimatePresence } from "framer-motion";
import { Briefcase, User } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

const MESSAGES: { role: "interviewer" | "you"; text: string }[] = [
  {
    role: "interviewer",
    text: "请解释一下 GQA 和 MHA 的区别？",
  },
  {
    role: "you",
    text: "GQA 是分组查询注意力，让多个 Q 头共享同一组 KV，MiniMind 中 8 个 Q 头共享 4 组 KV，在性能和效率间取得平衡。",
  },
  {
    role: "interviewer",
    text: "KV-Cache 是什么？为什么训练时不用？",
  },
  {
    role: "you",
    text: "KV-Cache 缓存历史 token 的 K 和 V，避免推理时重复计算。训练时因为所有 token 并行处理，不需要缓存。",
  },
  {
    role: "interviewer",
    text: "LoRA 的核心原理是什么？",
  },
  {
    role: "you",
    text: "冻结原始权重 W，旁路加入低秩矩阵 W'=W+BA，r=8 时仅需 2% 的参数量即可微调。",
  },
];

const TOPIC_CARDS = [
  { emoji: "📝", title: "项目介绍话术", href: "/interview/01" as const },
  { emoji: "🏗️", title: "模型架构面试题", count: "28题", href: "/interview/02" as const },
  { emoji: "🔧", title: "训练流程面试题", count: "32题", href: "/interview/03" as const },
  { emoji: "🚀", title: "优化与部署面试题", count: "22题", href: "/interview/04" as const },
  { emoji: "🔥", title: "综合追问深挖题", count: "18题", href: "/interview/05" as const },
] as const;

function ChatBubble({
  role,
  text,
}: {
  role: "interviewer" | "you";
  text: string;
}) {
  const isInterviewer = role === "interviewer";
  const bubbleBg = isInterviewer ? "#1e3a5f" : "#1a3a2e";

  return (
    <motion.div
      layout
      initial={{ opacity: 0, x: isInterviewer ? -24 : 24, scale: 0.96 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      exit={{ opacity: 0, scale: 0.96 }}
      transition={{
        type: "spring" as const,
        stiffness: 380,
        damping: 28,
      }}
      className={`flex w-full gap-3 ${isInterviewer ? "flex-row" : "flex-row-reverse"}`}
    >
      <div
        className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-white/10 ${
          isInterviewer
            ? "bg-blue-500/20 text-blue-200"
            : "bg-emerald-500/20 text-emerald-200"
        }`}
        aria-hidden
      >
        {isInterviewer ? (
          <Briefcase className="h-5 w-5" />
        ) : (
          <User className="h-5 w-5" />
        )}
      </div>
      <div className={`flex max-w-[min(100%,36rem)] flex-col ${isInterviewer ? "items-start" : "items-end"}`}>
        <span className="mb-1 text-xs text-zinc-500">
          {isInterviewer ? "面试官" : "你"}
        </span>
        <div
          className="rounded-2xl px-4 py-3 text-sm leading-relaxed text-zinc-100 shadow-lg ring-1 ring-white/5"
          style={{ backgroundColor: bubbleBg }}
        >
          {text}
        </div>
      </div>
    </motion.div>
  );
}

export function InterviewPreview() {
  const [visibleCount, setVisibleCount] = useState(1);

  useEffect(() => {
    if (visibleCount === 0) {
      const t = window.setTimeout(() => setVisibleCount(1), 400);
      return () => clearTimeout(t);
    }
    if (visibleCount < MESSAGES.length) {
      const t = window.setTimeout(() => setVisibleCount((c) => c + 1), 1000);
      return () => clearTimeout(t);
    }
    const t = window.setTimeout(() => setVisibleCount(0), 2800);
    return () => clearTimeout(t);
  }, [visibleCount]);

  return (
    <section className="w-full px-4 py-12 sm:px-6">
      <div className="mx-auto max-w-4xl">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mb-10 text-center"
        >
          <h2 className="text-3xl font-bold tracking-tight text-zinc-100 sm:text-4xl">
            面试宝典
          </h2>
          <p className="mt-2 text-base text-zinc-400 sm:text-lg">
            100+ 真实面试题，覆盖所有考点 —{" "}
            <Link
              href="/interview"
              className="text-[#8b5cf6] underline-offset-2 hover:underline"
            >
              点击进入网页阅读全文
            </Link>
          </p>
        </motion.div>

        <div className="mb-12 rounded-2xl border border-white/10 bg-zinc-900/40 p-4 sm:p-6">
          <div className="flex min-h-[280px] flex-col gap-4">
            <AnimatePresence initial={false} mode="popLayout">
              {MESSAGES.slice(0, visibleCount).map((msg, i) => (
                <ChatBubble key={`${i}-${msg.text.slice(0, 12)}`} role={msg.role} text={msg.text} />
              ))}
            </AnimatePresence>
          </div>
        </div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          <p className="mb-4 text-center text-sm font-medium text-zinc-500">
            面试专题
          </p>
          <div className="-mx-4 flex gap-4 overflow-x-auto px-4 pb-2 sm:mx-0 sm:px-0">
            {TOPIC_CARDS.map((card) => (
              <Link key={card.title} href={card.href} className="shrink-0">
                <motion.div
                  whileHover={{
                    scale: 1.03,
                    boxShadow:
                      "0 0 32px rgba(59, 130, 246, 0.35), 0 0 48px rgba(139, 92, 246, 0.2)",
                  }}
                  whileTap={{ scale: 0.99 }}
                  transition={{ type: "spring", stiffness: 400, damping: 25 }}
                  className="min-w-[200px] cursor-pointer rounded-xl border border-white/10 bg-zinc-900/80 px-4 py-4 ring-1 ring-white/5 transition-shadow"
                >
                  <div className="text-2xl">{card.emoji}</div>
                  <div className="mt-2 text-sm font-semibold text-zinc-100">
                    {card.title}
                  </div>
                  {"count" in card && card.count && (
                    <div className="mt-1 text-xs text-zinc-500">{card.count}</div>
                  )}
                  <div className="mt-2 text-xs font-medium text-[#8b5cf6]">
                    点击阅读 →
                  </div>
                </motion.div>
              </Link>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}

export default InterviewPreview;
