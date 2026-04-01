import Link from "next/link";
import { LESSON_FILES, LESSON_ORDER } from "@/lib/lessons-data";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "课程目录 | Learn MiniMind",
  description: "22 节 MiniMind 课程，点击在浏览器中阅读完整 Markdown",
};

const PHASE_LABEL: Record<string, { label: string; color: string }> = {
  L01: { label: "Phase 1", color: "#10b981" },
  L02: { label: "Phase 1", color: "#10b981" },
  L03: { label: "Phase 1", color: "#10b981" },
  L04: { label: "Phase 1", color: "#10b981" },
  L05: { label: "Phase 2", color: "#3b82f6" },
  L06: { label: "Phase 2", color: "#3b82f6" },
  L07: { label: "Phase 2", color: "#3b82f6" },
  L08: { label: "Phase 2", color: "#3b82f6" },
  L09: { label: "Phase 2", color: "#3b82f6" },
  L10: { label: "Phase 2", color: "#3b82f6" },
  L11: { label: "Phase 3", color: "#8b5cf6" },
  L12: { label: "Phase 3", color: "#8b5cf6" },
  L13: { label: "Phase 3", color: "#8b5cf6" },
  L14: { label: "Phase 3", color: "#8b5cf6" },
  L15: { label: "Phase 3", color: "#8b5cf6" },
  L16: { label: "Phase 3", color: "#8b5cf6" },
  L17: { label: "Phase 4", color: "#f59e0b" },
  L18: { label: "Phase 4", color: "#f59e0b" },
  L19: { label: "Phase 4", color: "#f59e0b" },
  L20: { label: "Phase 4", color: "#f59e0b" },
  L21: { label: "Phase 4", color: "#f59e0b" },
  L22: { label: "Phase 4", color: "#f59e0b" },
};

export default function LearnIndexPage() {
  return (
    <div className="min-h-screen bg-[#0a0a0f] pb-20">
      <div className="mx-auto max-w-3xl px-4 pt-10 md:px-6">
        <Link
          href="/"
          className="mb-6 inline-block text-sm text-zinc-500 hover:text-[#8b5cf6]"
        >
          ← 返回首页动画
        </Link>
        <h1 className="text-3xl font-bold text-zinc-100">课程目录</h1>
        <p className="mt-3 text-zinc-400">
          点击任意一节课，在网页中阅读{" "}
          <code className="text-zinc-500">docs/</code> 下的完整内容（与 GitHub
          同步）。风格参考{" "}
          <a
            href="https://learn.shareai.run"
            target="_blank"
            rel="noopener noreferrer"
            className="text-[#8b5cf6] underline-offset-2 hover:underline"
          >
            learn-claude-code 学习站
          </a>
          的可点击学习路径。
        </p>

        <ul className="mt-10 space-y-2">
          {LESSON_ORDER.map((slug) => {
            const phase = PHASE_LABEL[slug];
            const file = LESSON_FILES[slug];
            return (
              <li key={slug}>
                <Link
                  href={`/lesson/${slug}`}
                  className="group flex items-center gap-4 rounded-xl border border-zinc-800/90 bg-[#1a1a2e]/80 px-4 py-3 transition hover:border-[#8b5cf6]/35 hover:bg-[#1a1a2e]"
                >
                  <span
                    className="shrink-0 rounded-md px-2 py-1 font-mono text-xs font-bold text-white"
                    style={{ backgroundColor: `${phase.color}cc` }}
                  >
                    {slug}
                  </span>
                  <span className="min-w-0 flex-1 truncate text-zinc-200 group-hover:text-white">
                    {file.replace(/\.md$/, "").replace(/^(L\d+-)/, "")}
                  </span>
                  <span className="hidden text-zinc-600 group-hover:text-[#8b5cf6] sm:inline">
                    阅读 →
                  </span>
                </Link>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
