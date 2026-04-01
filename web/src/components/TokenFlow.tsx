"use client";

import { motion } from "framer-motion";
import { Fragment, useEffect, useState } from "react";

const STAGE_COUNT = 5;
const STAGE_MS = 1400;

const STAGES = [
  { key: "input", title: "输入", color: "#10b981" },
  { key: "tokenizer", title: "Tokenizer", color: "#3b82f6" },
  { key: "embedding", title: "Embedding", color: "#8b5cf6" },
  { key: "transformer", title: "Transformer ×8", color: "#f59e0b" },
  { key: "output", title: "输出", color: "#ec4899" },
] as const;

function FlowDots({
  orientation,
  active,
}: {
  orientation: "horizontal" | "vertical";
  active: boolean;
}) {
  const dots = [0, 1, 2];
  const isH = orientation === "horizontal";

  return (
    <div
      className={
        isH
          ? "relative h-1 w-full min-h-[4px] min-w-[24px] overflow-hidden rounded-full bg-zinc-800/90"
          : "relative h-10 w-1 min-h-[32px] min-w-[4px] overflow-hidden rounded-full bg-zinc-800/90"
      }
      aria-hidden
    >
      {dots.map((i) => (
        <motion.span
          key={i}
          className="absolute rounded-full bg-zinc-100 shadow-[0_0_6px_rgba(255,255,255,0.35)]"
          style={
            isH
              ? { top: "50%", width: 6, height: 6, marginTop: -3 }
              : { left: "50%", width: 6, height: 6, marginLeft: -3 }
          }
          initial={isH ? { left: "0%", x: "-50%" } : { top: "0%", y: "-50%" }}
          animate={
            active
              ? isH
                ? { left: ["0%", "100%"], x: "-50%" }
                : { top: ["0%", "100%"], y: "-50%" }
              : isH
                ? { left: "0%", x: "-50%" }
                : { top: "0%", y: "-50%" }
          }
          transition={{
            duration: 1.35,
            repeat: active ? Infinity : 0,
            ease: "linear",
            delay: i * 0.35,
          }}
        />
      ))}
    </div>
  );
}

function StageBody({
  index,
  activeStage,
}: {
  index: number;
  activeStage: number;
}) {
  const reached = activeStage >= index;

  if (index === 0) {
    return (
      <p
        className={`mt-2 text-center text-xs leading-snug md:text-sm ${
          reached ? "text-white/95" : "text-white/35"
        }`}
      >
        天空为什么是蓝色的
      </p>
    );
  }
  if (index === 1) {
    return (
      <p
        className={`mt-2 break-all font-mono text-[10px] leading-tight md:text-xs ${
          reached ? "text-white/90" : "text-white/30"
        }`}
      >
        {reached ? "[2541, 894, 117, 3208, 12]" : "…"}
      </p>
    );
  }
  if (index === 2) {
    return (
      <p
        className={`mt-2 font-mono text-xs md:text-sm ${
          reached ? "text-white/90" : "text-white/30"
        }`}
      >
        {reached ? "[768d] 向量" : "…"}
      </p>
    );
  }
  if (index === 3) {
    return (
      <p
        className={`mt-2 text-xs md:text-sm ${
          reached ? "text-amber-100/95" : "text-white/30"
        }`}
      >
        {reached ? "Processing..." : "…"}
      </p>
    );
  }
  return (
    <p
      className={`mt-2 text-lg font-semibold md:text-xl ${
        reached ? "text-white" : "text-white/30"
      }`}
    >
      {reached ? "因" : "…"}
    </p>
  );
}

function StageCard({
  index,
  activeStage,
  title,
  color,
}: {
  index: number;
  activeStage: number;
  title: string;
  color: string;
}) {
  const isActive = activeStage === index;

  return (
    <motion.div
      className="relative z-[1] flex min-w-[140px] max-w-[220px] flex-1 flex-col rounded-2xl border border-white/10 px-3 py-3 shadow-lg md:min-w-[160px] md:px-4 md:py-4"
      style={{ backgroundColor: `${color}22` }}
      animate={
        isActive
          ? {
              scale: [1, 1.04, 1],
              boxShadow: [
                `0 0 0 0 ${color}00`,
                `0 0 28px 2px ${color}66`,
                `0 0 0 0 ${color}00`,
              ],
            }
          : { scale: 1, boxShadow: "0 10px 30px -12px rgba(0,0,0,0.45)" }
      }
      transition={
        isActive
          ? { duration: 1.2, repeat: Infinity, ease: "easeInOut" }
          : { duration: 0.35 }
      }
    >
      <div
        className="rounded-lg px-2 py-1 text-center text-xs font-semibold tracking-wide text-white md:text-sm"
        style={{ backgroundColor: color }}
      >
        {title}
      </div>
      <StageBody index={index} activeStage={activeStage} />
    </motion.div>
  );
}

export default function TokenFlow() {
  const [activeStage, setActiveStage] = useState(0);

  useEffect(() => {
    const t = window.setInterval(() => {
      setActiveStage((s) => (s + 1) % STAGE_COUNT);
    }, STAGE_MS);
    return () => window.clearInterval(t);
  }, []);

  const connectors = STAGES.length - 1;

  return (
    <div className="w-full max-w-6xl px-3 py-6 md:px-4">
      <motion.p
        className="mb-6 text-center text-base font-medium text-zinc-100 md:text-lg"
        animate={{
          opacity: [0.88, 1, 0.88],
          scale: [1, 1.015, 1],
        }}
        transition={{
          duration: 1.35,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      >
        天空为什么是蓝色的
      </motion.p>

      {/* 小屏：纵向管道 */}
      <div className="flex flex-col items-stretch gap-0 md:hidden">
        {STAGES.map((s, i) => (
          <div key={s.key} className="flex flex-col items-center">
            <StageCard
              index={i}
              activeStage={activeStage}
              title={s.title}
              color={s.color}
            />
            {i < connectors && (
              <div className="flex w-full max-w-[220px] flex-col items-center py-2">
                <FlowDots
                  orientation="vertical"
                  active={activeStage === i || activeStage === i + 1}
                />
              </div>
            )}
          </div>
        ))}
      </div>

      {/* 中大屏：横向管道 */}
      <div className="hidden w-full flex-row flex-nowrap items-stretch justify-center md:flex">
        {STAGES.map((s, i) => (
          <Fragment key={s.key}>
            <div className="flex shrink-0">
              <StageCard
                index={i}
                activeStage={activeStage}
                title={s.title}
                color={s.color}
              />
            </div>
            {i < connectors && (
              <div className="mx-1 flex min-h-[120px] min-w-[28px] flex-1 items-center md:mx-2 md:min-w-[40px]">
                <FlowDots
                  orientation="horizontal"
                  active={activeStage === i || activeStage === i + 1}
                />
              </div>
            )}
          </Fragment>
        ))}
      </div>
    </div>
  );
}
