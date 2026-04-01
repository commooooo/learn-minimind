import Link from "next/link";
import { notFound } from "next/navigation";
import MarkdownArticle from "@/components/MarkdownArticle";
import { LESSON_FILES, LESSON_ORDER } from "@/lib/lessons-data";
import { readLessonMarkdown } from "@/lib/readMarkdown";
import { titleFromMarkdown } from "@/lib/mdTitle";
import type { Metadata } from "next";

type Props = { params: Promise<{ slug: string }> };

export function generateStaticParams() {
  return LESSON_ORDER.map((slug) => ({ slug }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const raw = readLessonMarkdown(slug);
  if (!raw) return { title: "课程未找到" };
  return {
    title: `${titleFromMarkdown(raw, slug)} | Learn MiniMind`,
    description: `MiniMind 学习课程 ${slug}：${LESSON_FILES[slug] ?? ""}`,
  };
}

export default async function LessonPage({ params }: Props) {
  const { slug } = await params;
  const raw = readLessonMarkdown(slug);
  if (!raw) notFound();

  const title = titleFromMarkdown(raw, slug);
  const idx = LESSON_ORDER.indexOf(slug as (typeof LESSON_ORDER)[number]);
  const prev = idx > 0 ? LESSON_ORDER[idx - 1] : null;
  const next =
    idx >= 0 && idx < LESSON_ORDER.length - 1 ? LESSON_ORDER[idx + 1] : null;

  return (
    <div className="min-h-screen bg-[#0a0a0f] pb-20">
      <div className="mx-auto max-w-3xl px-4 pt-8 md:px-6">
        <div className="mb-8 flex flex-wrap items-center gap-3 text-sm">
          <Link
            href="/learn"
            className="text-zinc-500 transition hover:text-[#8b5cf6]"
          >
            ← 课程目录
          </Link>
          <span className="text-zinc-700">/</span>
          <span className="font-mono text-xs text-zinc-500">{slug}</span>
        </div>

        <h1 className="mb-2 text-2xl font-bold tracking-tight text-zinc-100 md:text-3xl">
          {title}
        </h1>
        <p className="mb-10 text-sm text-zinc-500">
          对应仓库文件{" "}
          <code className="rounded bg-zinc-800 px-1.5 py-0.5 text-zinc-400">
            docs/{LESSON_FILES[slug]}
          </code>
        </p>

        <MarkdownArticle content={raw} />

        <nav className="mt-16 flex flex-col gap-3 border-t border-zinc-800 pt-10 sm:flex-row sm:justify-between">
          {prev ? (
            <Link
              href={`/lesson/${prev}`}
              className="rounded-xl border border-zinc-800 bg-zinc-900/50 px-4 py-3 text-sm text-zinc-300 transition hover:border-[#8b5cf6]/40 hover:text-white"
            >
              ← 上一课 {prev}
            </Link>
          ) : (
            <span />
          )}
          {next ? (
            <Link
              href={`/lesson/${next}`}
              className="rounded-xl border border-zinc-800 bg-zinc-900/50 px-4 py-3 text-sm text-zinc-300 transition hover:border-[#8b5cf6]/40 hover:text-white sm:text-right"
            >
              下一课 {next} →
            </Link>
          ) : (
            <Link
              href="/interview"
              className="rounded-xl border border-emerald-900/50 bg-emerald-950/30 px-4 py-3 text-sm text-emerald-200/90 transition hover:border-emerald-700/50 sm:text-right"
            >
              进入面试宝典 →
            </Link>
          )}
        </nav>
      </div>
    </div>
  );
}
