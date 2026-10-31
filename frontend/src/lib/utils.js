import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
export function cn(...values) { return twMerge(clsx(values)); }
export function formatTime(ms = 0, detailed = false) {
  const val = Math.max(0, Number(ms) || 0);
  const seconds = Math.floor(val / 1000);
  const base = `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
  return detailed ? `${base}.${String(Math.floor(val % 1000 / 100))}` : base;
}
export function formatBytes(bytes = 0) { return bytes ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : '—'; }
export function downloadText(filename, text, type = 'application/json') {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement('a'); link.href = url; link.download = filename; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
