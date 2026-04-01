import Link from "next/link";
import {
  INTERVIEW_FILES,
  INTERVIEW_META,
  INTERVIEW_ORDER,
} from "@/lib/interview-data";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "面试宝典 | Learn MiniMind",
  description: "项目介绍话术与 100+ 面试题，网页端阅读",
};

export default function InterviewIndexPage() {
  return (
    <div className="min-h-screen bg-[#0a0a0f] pb-20">
      <div className="mx-auto max-w-3xl px-4 pt-10 md:px-6">
        <Link
          href="/"
          className="mb-6 inline-block text-sm text-zinc-500 hover:text-[#8b5cf6]"
        >
          ← 返回首页
        </Link>
        <h1 className="text-3xl font-bold text-zinc-100">面试宝典</h1>
        <p className="mt-3 text-zinc-400">
          与仓库 <code className="text-zinc-500">interview/</code>{" "}
          目录一一对应，支持点击连续阅读。
        </p>

        <ul className="mt-10 space-y-3">
          {INTERVIEW_ORDER.map((slug) => {
            const meta = INTERVIEW_META[slug];
            const file = INTERVIEW_FILES[slug];
            return (
              <li key={slug}>
                <Link
                  href={`/interview/${slug}`}
                  className="block rounded-xl border border-zinc-800/90 bg-[#1a1a2e]/80 px-5 py-4 transition hover:border-emerald-600/40 hover:bg-[#1a1a2e]"
                >
                  <div className="font-mono text-xs text-zinc-500">
                    {file}
                  </div>
                  <div className="mt-1 text-lg font-semibold text-zinc-100">
                    {meta?.title ?? slug}
                  </div>
                  {meta?.subtitle ? (
                    <div className="mt-1 text-sm text-zinc-500">
                      {meta.subtitle}
                    </div>
                  ) : null}
                </Link>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
