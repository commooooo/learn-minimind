import Hero from "@/components/Hero";
import CodeBlock from "@/components/CodeBlock";
import TokenFlow from "@/components/TokenFlow";
import LearningPath from "@/components/LearningPath";
import Architecture from "@/components/Architecture";
import InterviewPreview from "@/components/InterviewPreview";
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <main className="min-h-screen">
      <Hero />

      <section className="max-w-5xl mx-auto px-6 py-20">
        <h2 className="text-3xl font-bold text-center mb-4 text-white">
          核心代码
        </h2>
        <p className="text-center text-zinc-400 mb-12">
          MiniMind 的 Transformer Block — 所有 LLM 的心脏
        </p>
        <CodeBlock />
      </section>

      <section className="max-w-6xl mx-auto px-6 py-20">
        <h2 className="text-3xl font-bold text-center mb-4 text-white">
          数据流可视化
        </h2>
        <p className="text-center text-zinc-400 mb-12">
          看一个 Token 如何从输入文字变成模型输出
        </p>
        <TokenFlow />
      </section>

      <section id="start">
        <LearningPath />
      </section>

      <section className="py-20">
        <Architecture />
      </section>

      <InterviewPreview />

      <Footer />
    </main>
  );
}
