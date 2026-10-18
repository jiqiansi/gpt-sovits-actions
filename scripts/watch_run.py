#!/usr/bin/env python3
"""轮询一个 GitHub Actions run，把每一步的实时状态打出来。

用法: python3 watch_run.py <run_id>
按 Ctrl-C 停止。跑完会打印最终 conclusion 和 artifact 列表。
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

TOKEN = open(os.path.expanduser("~/.dsh/skills/github-actions-batch-render/token.txt")).read().strip()
REPO = "jiqiansi/gpt-sovits-actions"
API = f"https://api.github.com/repos/{REPO}"


def get(path):
    req = urllib.request.Request(API + path, headers={
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/vnd.github+json",
    })
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main():
    run_id = sys.argv[1]
    seen = {}
    last = ""
    for _ in range(2000):
        try:
            info = get(f"/actions/runs/{run_id}")
        except Exception as e:
            print("poll error:", repr(e)[:120])
            time.sleep(20)
            continue
        if info["status"] == "completed":
            print(f"\n===== RUN 完成: {info['conclusion']} =====")
            try:
                arts = get(f"/actions/runs/{run_id}/artifacts")["artifacts"]
                for a in arts:
                    print(f"  artifact: {a['name']}  {a['size_in_bytes']} bytes  expired={a['expired']}")
                if not arts:
                    print("  (没有 artifact)")
            except Exception as e:
                print("  artifact 查询失败", e)
            print(f"  {info['html_url']}")
            return 0 if info["conclusion"] == "success" else 1

        try:
            jobs = get(f"/actions/runs/{run_id}/jobs")["jobs"]
        except Exception as e:
            print("jobs error:", repr(e)[:120])
            time.sleep(20)
            continue

        lines = []
        for job in jobs:
            lines.append(f"[{job['status']}] {job['name']}")
            for st in job.get("steps", []):
                mark = {"completed": "✓", "in_progress": "▶", "queued": "·"}.get(st["status"], "?")
                dur = ""
                if st.get("started_at") and st.get("completed_at"):
                    t0 = time.mktime(time.strptime(st["started_at"], "%Y-%m-%dT%H:%M:%SZ"))
                    t1 = time.mktime(time.strptime(st["completed_at"], "%Y-%m-%dT%H:%M:%SZ"))
                    dur = f" ({int(t1 - t0)}s)"
                err = "  <<< FAILED" if st.get("conclusion") == "failure" else ""
                lines.append(f"   {mark} {st['name']}{dur}{err}")
        txt = "\n".join(lines)
        if txt != last:
            print(f"\n--- {time.strftime('%H:%M:%S')} ---")
            print(txt)
            last = txt
        time.sleep(25)


if __name__ == "__main__":
    sys.exit(main())
