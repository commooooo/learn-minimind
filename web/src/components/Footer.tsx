"use client";

import { motion } from "framer-motion";

const LINKS = [
  { label: "GitHub", href: "https://github.com/bcefghj/learn-minimind" },
  { label: "MiniMind 原项目", href: "https://github.com/jingyaogong/minimind" },
  { label: "MiniMind Wiki", href: "https://minimind.wiki" },
] as const;

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.12, delayChildren: 0.08 },
  },
};

const item = {
  hidden: { opacity: 0, y: 14 },
  show: {
    opacity: 1,
    y: 0,
    transition: { type: "spring" as const, stiffness: 300, damping: 28 },
  },
};

export function Footer() {
  return (
    <footer className="relative mt-auto w-full bg-[#0a0a0f] px-4 py-14 sm:px-6">
      <div
        className="pointer-events-none absolute inset-x-0 top-0 h-px"
        style={{
          background:
            "linear-gradient(90deg, transparent, #3b82f6, #8b5cf6, #ec4899, transparent)",
        }}
        aria-hidden
      />

      <motion.div
        className="mx-auto max-w-3xl text-center"
        variants={container}
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, margin: "-40px" }}
      >
        <motion.p
          variants={item}
          className="text-balance text-lg font-bold leading-relaxed text-zinc-100 sm:text-xl"
        >
          大道至简 — 从零开始，一步一步，你也能训练自己的大语言模型。
        </motion.p>
        <motion.p
          variants={item}
          className="mt-4 text-sm italic text-zinc-400 sm:text-base"
        >
          Star ⭐ 这个仓库，开始你的 LLM 学习之旅吧！
        </motion.p>

        <motion.nav
          variants={item}
          className="mt-10 flex flex-wrap items-center justify-center gap-x-6 gap-y-3"
          aria-label="页脚链接"
        >
          {LINKS.map((link) => (
            <a
              key={link.href}
              href={link.href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-zinc-400 underline-offset-4 transition-colors hover:text-violet-300 hover:underline"
            >
              {link.label}
            </a>
          ))}
        </motion.nav>

        <motion.p
          variants={item}
          className="mt-8 text-xs text-zinc-600"
        >
          MIT License · 2026
        </motion.p>
      </motion.div>
    </footer>
  );
}

export default Footer;
