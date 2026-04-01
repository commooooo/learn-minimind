import type { Metadata } from "next";
import SiteNav from "@/components/SiteNav";
import "./globals.css";

export const metadata: Metadata = {
  title: "Learn MiniMind — 从零基础到面试通关",
  description:
    "22节课彻底搞懂大语言模型，从零训练64M参数GPT，系统化学习LLM全流程",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN">
      <body className="antialiased">
        <SiteNav />
        {children}
      </body>
    </html>
  );
}
