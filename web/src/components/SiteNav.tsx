import Link from "next/link";
import { Github, BookOpen, MessageSquare } from "lucide-react";

export default function SiteNav() {
  return (
    <header className="sticky top-0 z-50 border-b border-zinc-800/80 bg-[#0a0a0f]/90 backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between gap-4 px-4 md:px-6">
        <Link
          href="/"
          className="font-semibold tracking-tight text-zinc-100 transition hover:text-white"
        >
          Learn <span className="text-[#8b5cf6]">MiniMind</span>
        </Link>
        <nav className="flex flex-wrap items-center gap-1 text-sm md:gap-2">
          <Link
            href="/learn"
            className="inline-flex items-center gap-1.5 rounded-lg px-3 py-2 text-zinc-400 transition hover:bg-zinc-800/80 hover:text-zinc-100"
          >
            <BookOpen className="h-4 w-4" />
            <span className="hidden sm:inline">课程目录</span>
            <span className="sm:hidden">课程</span>
          </Link>
          <Link
            href="/interview"
            className="inline-flex items-center gap-1.5 rounded-lg px-3 py-2 text-zinc-400 transition hover:bg-zinc-800/80 hover:text-zinc-100"
          >
            <MessageSquare className="h-4 w-4" />
            <span className="hidden sm:inline">面试宝典</span>
            <span className="sm:hidden">面试</span>
          </Link>
          <a
            href="https://github.com/shareAI-lab/learn-claude-code"
            target="_blank"
            rel="noopener noreferrer"
            className="hidden rounded-lg px-3 py-2 text-zinc-500 transition hover:bg-zinc-800/80 hover:text-zinc-300 md:inline"
          >
            参考：learn-claude-code
          </a>
          <a
            href="https://github.com/bcefghj/learn-minimind"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 rounded-lg px-3 py-2 text-zinc-400 transition hover:bg-zinc-800/80 hover:text-zinc-100"
          >
            <Github className="h-4 w-4" />
            <span className="hidden sm:inline">本仓库</span>
          </a>
        </nav>
      </div>
    </header>
  );
}
