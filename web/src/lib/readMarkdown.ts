import fs from "fs";
import path from "path";
import { INTERVIEW_FILES } from "./interview-data";
import { LESSON_FILES } from "./lessons-data";

const REPO_ROOT = path.join(process.cwd(), "..");

export function readDocFile(relativeFromRepoRoot: string): string {
  const full = path.join(REPO_ROOT, relativeFromRepoRoot);
  return fs.readFileSync(full, "utf8");
}

export function readLessonMarkdown(slug: string): string | null {
  const name = LESSON_FILES[slug];
  if (!name) return null;
  return readDocFile(path.join("docs", name));
}

export function readInterviewMarkdown(slug: string): string | null {
  const name = INTERVIEW_FILES[slug];
  if (!name) return null;
  return readDocFile(path.join("interview", name));
}
