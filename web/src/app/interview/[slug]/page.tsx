import Link from "next/link";
import { notFound } from "next/navigation";
import MarkdownArticle from "@/components/MarkdownArticle";
import {
  INTERVIEW_FILES,
  INTERVIEW_META,
  INTERVIEW_ORDER,
} from "@/lib/interview-data";
import { readInterviewMarkdown } from "@/lib/readMarkdown";
import { titleFromMarkdown } from "@/lib/mdTitle";
import type { Metadata } from "next";

type Props = { params: Promise<{ slug: string }> };

export function generateStaticParams() {
  return INTERVIEW_ORDER.map((slug) => ({ slug }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const raw = readInterviewMarkdown(slug);
  const meta = INTERVIEW_META[slug];
  if (!raw) return { title: "未找到" };
  return {
    title: `${titleFromMarkdown(raw, meta?.title ?? slug)} | 面试宝典`,
    description: meta?.subtitle,
  };
}

export default async function InterviewDocPage({ params }: Props) {
  const { slug } = await params;
  const raw = readInterviewMarkdown(slug);
  if (!raw) notFound();

  const meta = INTERVIEW_META[slug];
  const title = titleFromMarkdown(raw, meta?.title ?? slug);
  const order: string[] = [...INTERVIEW_ORDER];
  const idx = order.indexOf(slug);
  const prev = idx > 0 ? order[idx - 1] : null;
  const next = idx >= 0 && idx < order.length - 1 ? order[idx + 1] : null;

  return (
    <div className="min-h-screen bg-[#0a0a0f] pb-20">
      <div className="mx-auto max-w-3xl px-4 pt-8 md:px-6">
        <div className="mb-8 flex flex-wrap items-center gap-3 text-sm">
          <Link
            href="/interview"
            className="text-zinc-500 transition hover:text-[#8b5cf6]"
          >
            ← 面试宝典目录
          </Link>
        </div>

        <h1 className="mb-2 text-2xl font-bold tracking-tight text-zinc-100 md:text-3xl">
          {title}
        </h1>
        {meta?.subtitle ? (
          <p className="mb-6 text-sm text-zinc-500">{meta.subtitle}</p>
        ) : null}
        <p className="mb-10 text-sm text-zinc-500">
          源文件{" "}
          <code className="rounded bg-zinc-800 px-1.5 py-0.5 text-zinc-400">
            interview/{INTERVIEW_FILES[slug]}
          </code>
        </p>

        <MarkdownArticle content={raw} />

        <nav className="mt-16 flex flex-col gap-3 border-t border-zinc-800 pt-10 sm:flex-row sm:justify-between">
          {prev ? (
            <Link
              href={`/interview/${prev}`}
              className="rounded-xl border border-zinc-800 bg-zinc-900/50 px-4 py-3 text-sm text-zinc-300 transition hover:border-[#8b5cf6]/40 hover:text-white"
            >
              ← 上一篇
            </Link>
          ) : (
            <span />
          )}
          {next ? (
            <Link
              href={`/interview/${next}`}
              className="rounded-xl border border-zinc-800 bg-zinc-900/50 px-4 py-3 text-sm text-zinc-300 transition hover:border-[#8b5cf6]/40 hover:text-white sm:text-right"
            >
              下一篇 →
            </Link>
          ) : null}
        </nav>
      </div>
    </div>
  );
}
