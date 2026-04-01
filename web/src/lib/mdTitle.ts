/** 取 Markdown 第一个 # 标题作为页面标题 */
export function titleFromMarkdown(markdown: string, fallback: string): string {
  for (const line of markdown.split("\n")) {
    const t = line.trim();
    const m = t.match(/^#{1,6}\s+(.+)/);
    if (m) return m[1].trim();
  }
  return fallback;
}
