"use client";

import { useState } from "react";
import {
  AnimatePresence,
  LayoutGroup,
  motion,
} from "framer-motion";
import { ArrowDown, ArrowUp, ChevronDown, ChevronUp } from "lucide-react";

type LayerInfo = {
  id: string;
  title: string;
  params: string;
  description: string;
};

const LAYER_DETAILS: Record<string, LayerInfo> = {
  input_ids: {
    id: "input_ids",
    title: "Input IDs",
    params: "0 params",
    description: "词表中每个 token 的整数索引序列，作为模型入口的离散表示。",
  },
  embed_tokens: {
    id: "embed_tokens",
    title: "embed_tokens",
    params: "6400×768 = 4.9M params",
    description: "将 token ID 映射为 768 维稠密向量，为后续 Transformer 提供连续表示。",
  },
  transformer_parent: {
    id: "transformer_parent",
    title: "Transformer Block ×8",
    params: "堆叠 8 层",
    description:
      "每层包含自注意力与前馈子层，残差与 RMSNorm 稳定训练；8 层结构在深度与效率间折中。",
  },
  attention: {
    id: "attention",
    title: "Attention (GQA)",
    params:
      "Q(768→768) + K(768→384) + V(768→384) + O(768→768) = 1.77M params",
    description: "分组查询注意力：减少 KV 头数以降低缓存与计算，同时保持表达能力。",
  },
  add_norm_1: {
    id: "add_norm_1",
    title: "Add & Norm",
    params: "768 params（RMSNorm）",
    description: "残差连接后与 RMSNorm，稳定注意力子层输出分布。",
  },
  ffn: {
    id: "ffn",
    title: "FFN (SwiGLU)",
    params:
      "gate(768→2048) + up(768→2048) + down(2048→768) = 4.72M params",
    description: "门控前馈网络：SwiGLU 激活增强非线性与梯度流，是参数主要集中处之一。",
  },
  add_norm_2: {
    id: "add_norm_2",
    title: "Add & Norm",
    params: "768 params（RMSNorm）",
    description: "FFN 子层后的残差与归一化，与注意力侧结构对称。",
  },
  rmsnorm_final: {
    id: "rmsnorm_final",
    title: "RMSNorm",
    params: "768 params",
    description: "最终输出前的均方根归一化，不依赖 batch 统计，推理友好。",
  },
  lm_head: {
    id: "lm_head",
    title: "lm_head",
    params: "与 embed_tokens 共享权重",
    description: "将隐藏状态投影到词表维度，得到下一 token 的 logits；与词嵌入权重绑定以节省参数。",
  },
  output_logits: {
    id: "output_logits",
    title: "Output Logits",
    params: "0 params",
    description: "词表大小维的未归一化分数，经 softmax 可得到下一 token 概率分布。",
  },
};

function ArrowBetween() {
  return (
    <div className="flex justify-center py-1">
      <ArrowUp className="h-5 w-5 text-zinc-600" aria-hidden />
    </div>
  );
}

type LayerBoxProps = {
  layerId: string;
  selected: boolean;
  onSelect: (id: string) => void;
  children: React.ReactNode;
  className?: string;
  accent?: string;
};

function LayerBox({
  layerId,
  selected,
  onSelect,
  children,
  className = "",
  accent = "#8b5cf6",
}: LayerBoxProps) {
  return (
    <motion.button
      type="button"
      layout
      onClick={() => onSelect(layerId)}
      className={`relative w-full rounded-xl border px-4 py-3 text-left text-sm font-medium text-zinc-100 transition-shadow focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-offset-[#0a0a0f] ${className}`}
      style={{
        backgroundColor: "#1a1a2e",
        borderColor: selected ? accent : "rgba(63, 63, 70, 0.9)",
        boxShadow: selected
          ? `0 0 0 1px ${accent}66, 0 0 24px ${accent}33`
          : undefined,
      }}
      whileHover={{
        boxShadow: `0 0 0 1px ${accent}55, 0 0 28px ${accent}40`,
      }}
      transition={{ type: "spring", stiffness: 420, damping: 28 }}
    >
      {children}
    </motion.button>
  );
}

export default function Architecture() {
  const [selectedId, setSelectedId] = useState<string>("embed_tokens");
  const [blockOpen, setBlockOpen] = useState(false);

  const detail =
    LAYER_DETAILS[selectedId] ??
    LAYER_DETAILS.embed_tokens;

  return (
    <section className="relative mx-auto w-full max-w-6xl px-4 py-16 md:px-6">
      <div className="pointer-events-none absolute right-4 top-4 md:right-6 md:top-6">
        <span className="pointer-events-auto rounded-full border border-zinc-700/80 bg-zinc-900/80 px-3 py-1.5 font-mono text-xs font-semibold tracking-wide text-amber-400 backdrop-blur-sm">
          64M params
        </span>
      </div>

      <h2 className="mb-10 text-center text-3xl font-semibold tracking-tight text-zinc-100 md:text-4xl">
        MiniMind 模型架构
      </h2>

      <LayoutGroup id="minimind-arch">
        <div className="flex flex-col gap-10 lg:flex-row lg:items-start lg:gap-12">
          <div className="mx-auto flex w-full max-w-md flex-col-reverse lg:mx-0 lg:max-w-sm">
            <LayerBox
              layerId="input_ids"
              selected={selectedId === "input_ids"}
              onSelect={setSelectedId}
              accent="#22d3ee"
            >
              Input IDs
            </LayerBox>
            <ArrowBetween />
            <LayerBox
              layerId="embed_tokens"
              selected={selectedId === "embed_tokens"}
              onSelect={setSelectedId}
              accent="#10b981"
            >
              embed_tokens
            </LayerBox>
            <ArrowBetween />

            <motion.div
              layout
              layoutId="transformer-shell"
              className="overflow-hidden rounded-xl border border-zinc-700/90"
              style={{ backgroundColor: "#14141f" }}
            >
              <button
                type="button"
                onClick={() => {
                  setBlockOpen((o) => !o);
                  setSelectedId("transformer_parent");
                }}
                className="flex w-full items-center justify-between gap-2 rounded-xl px-4 py-3 text-left text-sm font-medium text-zinc-100 transition-colors hover:bg-zinc-800/40"
              >
                <span>Transformer Block ×8</span>
                {blockOpen ? (
                  <ChevronUp className="h-4 w-4 shrink-0 text-zinc-400" />
                ) : (
                  <ChevronDown className="h-4 w-4 shrink-0 text-zinc-400" />
                )}
              </button>

              <AnimatePresence initial={false}>
                {blockOpen && (
                  <motion.div
                    key="block-inner"
                    layout
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ type: "spring", stiffness: 320, damping: 32 }}
                    className="border-t border-zinc-800/90"
                  >
                    <div className="p-3">
                      <p className="px-1 pb-2 text-center text-[11px] uppercase tracking-wider text-zinc-500">
                        单层结构（重复 8 次）
                      </p>
                      <div className="flex flex-col-reverse gap-2">
                        <LayerBox
                          layerId="attention"
                          selected={selectedId === "attention"}
                          onSelect={setSelectedId}
                          accent="#3b82f6"
                          className="text-xs"
                        >
                          Attention
                        </LayerBox>
                        <div className="flex justify-center py-0.5">
                          <ArrowUp className="h-4 w-4 text-zinc-600" />
                        </div>
                        <LayerBox
                          layerId="add_norm_1"
                          selected={selectedId === "add_norm_1"}
                          onSelect={setSelectedId}
                          accent="#a78bfa"
                          className="text-xs"
                        >
                          Add &amp; Norm
                        </LayerBox>
                        <div className="flex justify-center py-0.5">
                          <ArrowUp className="h-4 w-4 text-zinc-600" />
                        </div>
                        <LayerBox
                          layerId="ffn"
                          selected={selectedId === "ffn"}
                          onSelect={setSelectedId}
                          accent="#f59e0b"
                          className="text-xs"
                        >
                          FFN
                        </LayerBox>
                        <div className="flex justify-center py-0.5">
                          <ArrowUp className="h-4 w-4 text-zinc-600" />
                        </div>
                        <LayerBox
                          layerId="add_norm_2"
                          selected={selectedId === "add_norm_2"}
                          onSelect={setSelectedId}
                          accent="#a78bfa"
                          className="text-xs"
                        >
                          Add &amp; Norm
                        </LayerBox>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            <ArrowBetween />
            <LayerBox
              layerId="rmsnorm_final"
              selected={selectedId === "rmsnorm_final"}
              onSelect={setSelectedId}
              accent="#ec4899"
            >
              RMSNorm
            </LayerBox>
            <ArrowBetween />
            <LayerBox
              layerId="lm_head"
              selected={selectedId === "lm_head"}
              onSelect={setSelectedId}
              accent="#10b981"
            >
              lm_head
            </LayerBox>
            <ArrowBetween />
            <LayerBox
              layerId="output_logits"
              selected={selectedId === "output_logits"}
              onSelect={setSelectedId}
              accent="#22d3ee"
            >
              Output Logits
            </LayerBox>
          </div>

          <div className="min-h-[200px] flex-1 lg:sticky lg:top-8">
            <AnimatePresence mode="wait">
              <motion.div
                key={detail.id}
                layoutId="layer-detail"
                layout
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ type: "spring", stiffness: 380, damping: 32 }}
                className="rounded-2xl border border-zinc-700/90 p-6 shadow-xl"
                style={{
                  backgroundColor: "#1a1a2e",
                  boxShadow: "0 0 40px rgba(139, 92, 246, 0.12)",
                }}
              >
                <h3 className="text-xl font-semibold text-zinc-100">
                  {detail.title}
                </h3>
                <p className="mt-2 font-mono text-sm text-violet-300/90">
                  {detail.params}
                </p>
                <p className="mt-4 text-sm leading-relaxed text-zinc-400">
                  {detail.description}
                </p>
              </motion.div>
            </AnimatePresence>
          </div>
        </div>
      </LayoutGroup>
    </section>
  );
}
