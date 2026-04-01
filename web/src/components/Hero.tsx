"use client";

import { useEffect, useRef, useState } from "react";
import { motion, useInView, animate } from "framer-motion";
import { Github } from "lucide-react";
import Link from "next/link";

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.11,
      delayChildren: 0.08,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 28 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.55,
      ease: [0.22, 1, 0.36, 1] as const,
    },
  },
};

const statsContainerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.09,
      delayChildren: 0,
    },
  },
};

const ctaRowVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0,
    },
  },
};

function useCountUp(
  target: number,
  enabled: boolean,
  options?: { duration?: number; formatter?: (n: number) => string }
) {
  const [display, setDisplay] = useState(0);
  const duration = options?.duration ?? (target > 5000 ? 2.3 : 1.55);
  const formatter = options?.formatter ?? ((n: number) => String(Math.round(n)));

  useEffect(() => {
    if (!enabled) return;
    const controls = animate(0, target, {
      duration,
      ease: [0.16, 1, 0.3, 1] as const,
      onUpdate: (latest) => setDisplay(latest),
    });
    return () => controls.stop();
  }, [enabled, target, duration]);

  return formatter(display);
}

function StatBlock({
  value,
  suffix,
  label,
  plus,
  formatValue,
}: {
  value: number;
  suffix: string;
  label: string;
  plus?: boolean;
  formatValue?: (n: number) => string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-40px" });
  const text = useCountUp(value, inView, {
    duration: value > 5000 ? 2.4 : 1.5,
    formatter: (n) => {
      const rounded = Math.round(n);
      const core = formatValue ? formatValue(rounded) : String(rounded);
      return plus ? `${core}+` : core;
    },
  });

  return (
    <motion.div
      ref={ref}
      variants={itemVariants}
      className="relative overflow-hidden rounded-2xl border border-white/[0.08] bg-white/[0.03] px-5 py-4 backdrop-blur-sm transition-colors hover:border-[#8b5cf6]/30 hover:bg-white/[0.05]"
    >
      <div className="font-mono text-2xl font-bold tracking-tight text-white sm:text-3xl">
        {text}
        {suffix}
      </div>
      <div className="mt-1 text-sm text-zinc-400">{label}</div>
    </motion.div>
  );
}

export function Hero() {
  return (
    <section className="relative isolate overflow-hidden bg-[#0a0a0f] px-4 pb-24 pt-20 sm:px-6 sm:pb-32 sm:pt-28 lg:px-8">
      {/* 顶部径向光晕 */}
      <div
        className="pointer-events-none absolute inset-x-0 -top-40 h-[min(520px,70vh)] opacity-[0.55]"
        aria-hidden
        style={{
          background: `
            radial-gradient(ellipse 90% 55% at 50% 0%, rgba(59, 130, 246, 0.22), transparent 58%),
            radial-gradient(ellipse 70% 45% at 72% 12%, rgba(139, 92, 246, 0.18), transparent 52%),
            radial-gradient(ellipse 55% 40% at 28% 10%, rgba(236, 72, 153, 0.12), transparent 48%)
          `,
        }}
      />
      <div
        className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-[#8b5cf6]/40 to-transparent"
        aria-hidden
      />

      <motion.div
        className="relative mx-auto max-w-5xl text-center"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        <motion.div variants={itemVariants}>
          <span className="inline-flex items-center gap-2 rounded-full border border-[#3b82f6]/25 bg-[#3b82f6]/10 px-4 py-1.5 text-xs font-medium text-[#93c5fd] sm:text-sm">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[#10b981] opacity-60" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-[#10b981]" />
            </span>
            系统化 LLM 实战教程
          </span>
        </motion.div>

        <motion.h1
          variants={itemVariants}
          className="mt-8 text-4xl font-extrabold tracking-tight sm:text-6xl sm:leading-[1.08] lg:text-7xl"
        >
          <span
            className="animate-gradient bg-gradient-to-r from-[#3b82f6] via-[#8b5cf6] to-[#ec4899] bg-clip-text text-transparent"
            style={{ WebkitBackgroundClip: "text" }}
          >
            Learn MiniMind
          </span>
        </motion.h1>

        <motion.p
          variants={itemVariants}
          className="mx-auto mt-6 max-w-2xl text-base leading-relaxed text-zinc-400 sm:text-lg"
        >
          从零基础到面试通关 — 22节课彻底搞懂大语言模型
        </motion.p>

        <motion.div
          variants={statsContainerVariants}
          className="mx-auto mt-12 grid max-w-3xl grid-cols-2 gap-3 sm:grid-cols-4 sm:gap-4"
        >
          <StatBlock value={22} suffix="" label="节课" />
          <StatBlock value={100} suffix="" label="面试题" plus />
          <StatBlock
            value={14000}
            suffix=""
            label="行内容"
            plus
            formatValue={(n) => n.toLocaleString("en-US")}
          />
          <StatBlock value={64} suffix="M" label="参数模型" />
        </motion.div>

        <motion.div
          variants={ctaRowVariants}
          className="mt-12 flex flex-col items-center justify-center gap-3 sm:flex-row sm:flex-wrap sm:gap-4"
        >
          <motion.a
            variants={itemVariants}
            href="#start"
            className="inline-flex items-center justify-center rounded-xl bg-gradient-to-r from-[#3b82f6] to-[#8b5cf6] px-8 py-3.5 text-base font-semibold text-white shadow-lg shadow-[#3b82f6]/25 transition hover:shadow-xl hover:shadow-[#8b5cf6]/30"
            whileHover={{ scale: 1.03, y: -2 }}
            whileTap={{ scale: 0.98 }}
          >
            看动画 & 学习路径 →
          </motion.a>
          <motion.div variants={itemVariants}>
            <Link
              href="/learn"
              className="inline-flex items-center justify-center rounded-xl border border-emerald-700/50 bg-emerald-950/40 px-8 py-3.5 text-base font-semibold text-emerald-100 transition hover:scale-[1.03] hover:border-emerald-500/60 hover:bg-emerald-950/70 hover:-translate-y-0.5 active:scale-[0.98]"
            >
              网页读讲义（可点击）→
            </Link>
          </motion.div>
          <motion.a
            variants={itemVariants}
            href="https://github.com/bcefghj/learn-minimind"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center justify-center gap-2 rounded-xl border border-zinc-700 bg-zinc-900/80 px-8 py-3.5 text-base font-semibold text-zinc-100 backdrop-blur-sm transition hover:border-[#8b5cf6]/50 hover:bg-zinc-800/90"
            whileHover={{ scale: 1.03, y: -2 }}
            whileTap={{ scale: 0.98 }}
          >
            <Github className="h-5 w-5 text-zinc-300" />
            本教程仓库 ⭐
          </motion.a>
        </motion.div>
        <motion.p
          variants={itemVariants}
          className="mx-auto mt-6 max-w-xl text-center text-xs text-zinc-600"
        >
          交互形式参考{" "}
          <a
            href="https://github.com/shareAI-lab/learn-claude-code"
            target="_blank"
            rel="noopener noreferrer"
            className="text-zinc-500 underline-offset-2 hover:text-[#8b5cf6] hover:underline"
          >
            learn-claude-code
          </a>
          的学习站{" "}
          <a
            href="https://learn.shareai.run"
            target="_blank"
            rel="noopener noreferrer"
            className="text-zinc-500 underline-offset-2 hover:text-[#8b5cf6] hover:underline"
          >
            learn.shareai.run
          </a>
          ；原模型项目见{" "}
          <a
            href="https://github.com/jingyaogong/minimind"
            target="_blank"
            rel="noopener noreferrer"
            className="text-zinc-500 underline-offset-2 hover:text-[#8b5cf6] hover:underline"
          >
            jingyaogong/minimind
          </a>
          。
        </motion.p>
      </motion.div>
    </section>
  );
}

export default Hero;
