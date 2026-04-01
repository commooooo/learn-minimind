'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { useInView } from 'framer-motion';

const CODE = `class MiniMindBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim)
        self.ffn_norm = RMSNorm(config.dim)
    
    def forward(self, x, pos_cis, mask):
        # Self-Attention + Residual
        h = x + self.attention(
            self.attention_norm(x), pos_cis, mask
        )
        # FFN + Residual  
        out = h + self.feed_forward(
            self.ffn_norm(h)
        )
        return out`;

const TYPE_INTERVAL_MS = 30;

const COLORS: Record<
  "keyword" | "name" | "string" | "comment" | "plain" | "punct",
  string
> = {
  keyword: "#c084fc",
  name: "#60a5fa",
  string: "#4ade80",
  comment: "#6b7280",
  plain: "#e4e4e7",
  punct: "#fbbf24",
};

const KEYWORDS = new Set(["class", "def", "return", "self"]);

/** 类名、函数名、常见构造调用等，按需求标蓝 */
const BLUE_IDENTIFIERS = new Set([
  "MiniMindBlock",
  "nn",
  "Module",
  "Attention",
  "FeedForward",
  "RMSNorm",
  "__init__",
  "forward",
]);

function buildCharStyles(source: string): { char: string; color: string }[] {
  const out: { char: string; color: string }[] = [];
  let i = 0;
  const n = source.length;
  const isWordChar = (ch: string) => /[a-zA-Z_]/.test(ch);

  while (i < n) {
    const c = source[i]!;

    if (c === " " || c === "\t") {
      out.push({ char: c, color: COLORS.plain });
      i++;
      continue;
    }

    if (c === "\n" || c === "\r") {
      out.push({ char: c, color: COLORS.plain });
      i++;
      continue;
    }

    if (c === "#") {
      while (i < n && source[i] !== "\n") {
        out.push({ char: source[i]!, color: COLORS.comment });
        i++;
      }
      continue;
    }

    if (c === '"' || c === "'") {
      const q = c;
      out.push({ char: c, color: COLORS.string });
      i++;
      while (i < n && source[i] !== q) {
        out.push({ char: source[i]!, color: COLORS.string });
        i++;
      }
      if (i < n) {
        out.push({ char: source[i]!, color: COLORS.string });
        i++;
      }
      continue;
    }

    if (isWordChar(c)) {
      const start = i;
      while (i < n && isWordChar(source[i]!)) i++;
      const word = source.slice(start, i);
      let color: (typeof COLORS)[keyof typeof COLORS] = COLORS.plain;
      if (KEYWORDS.has(word)) color = COLORS.keyword;
      else if (BLUE_IDENTIFIERS.has(word)) color = COLORS.name;
      for (let j = start; j < i; j++) {
        out.push({ char: source[j]!, color });
      }
      continue;
    }

    out.push({ char: c, color: COLORS.punct });
    i++;
  }

  return out;
}

export default function CodeBlock() {
  const containerRef = useRef<HTMLDivElement>(null);
  const isInView = useInView(containerRef, { once: true, margin: "0px 0px -8% 0px" });
  const charStyles = useMemo(() => buildCharStyles(CODE), []);
  const [visibleCount, setVisibleCount] = useState(0);
  const [complete, setComplete] = useState(false);
  const startedRef = useRef(false);

  useEffect(() => {
    if (!isInView || startedRef.current) return;
    startedRef.current = true;

    let count = 0;
    const id = window.setInterval(() => {
      count += 1;
      if (count >= charStyles.length) {
        setVisibleCount(charStyles.length);
        setComplete(true);
        window.clearInterval(id);
        return;
      }
      setVisibleCount(count);
    }, TYPE_INTERVAL_MS);

    return () => {
      startedRef.current = false;
      window.clearInterval(id);
    };
  }, [isInView, charStyles.length]);

  const slice = charStyles.slice(0, visibleCount);

  return (
    <div
      ref={containerRef}
      className="w-full max-w-3xl overflow-hidden rounded-xl border border-white/10 shadow-lg"
      style={{ backgroundColor: "#1a1a2e" }}
    >
      <div className="flex items-center gap-2 border-b border-white/10 px-4 py-2.5">
        <div className="flex gap-1.5">
          <span className="size-3 rounded-full bg-[#ff5f57]" aria-hidden />
          <span className="size-3 rounded-full bg-[#febc2e]" aria-hidden />
          <span className="size-3 rounded-full bg-[#28c840]" aria-hidden />
        </div>
        <span className="ml-2 font-mono text-xs text-zinc-400">model_minimind.py</span>
      </div>
      <pre className="overflow-x-auto p-4 font-mono text-[13px] leading-relaxed">
        <code>
          {slice.map((item, idx) => (
            <span key={idx} style={{ color: item.color }}>
              {item.char === "\n" ? "\n" : item.char === "\t" ? "\t" : item.char}
            </span>
          ))}
          {complete ? (
            <span
              aria-hidden
              className="ml-0.5 inline-block w-2 align-[-0.12em]"
              style={{
                height: "1.1em",
                backgroundColor: "#4ade80",
                animation: "blink 1s step-end infinite",
              }}
            />
          ) : null}
        </code>
      </pre>
    </div>
  );
}
