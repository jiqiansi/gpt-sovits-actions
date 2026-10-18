#!/usr/bin/env python3
"""git push 的兜底通道：走 GitHub Git Data API（api.github.com）。

什么时候用它：
  本机到 github.com 的直连会间歇性断，代理 7890 也常挂，
  而 api.github.com 往往还通着。这时 `git push` 走投无路，
  但 API 通道能把同样的提交推上去。

做法是把一次 commit 拆成 Git 的四个对象：
  blob（文件内容） -> tree（目录结构） -> commit（提交对象） -> ref（把 main 指过去）

用法:
    python3 scripts/push_via_api.py "commit message"

与本地 HEAD 的差异按 `git diff --cached` 算（内部用 git 读对象，不依赖网络）。
"""
import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

REPO = "jiqiansi/gpt-sovits-actions"
BASE = f"https://api.github.com/repos/{REPO}"


def token():
    return open(os.path.expanduser("~/.dsh/skills/github-actions-batch-render/token.txt")).read().strip()


def api(method, path, payload=None):
    url = path if path.startswith("http") else BASE + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {token()}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "curl/8",
    })
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            body = r.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        print(f"!! {method} {url} -> {e.code}\n{e.read().decode()[:600]}", file=sys.stderr)
        raise


def git(*args, text=False):
    """本地读 git 对象，完全不联网。默认按字节返回 —— 仓库里可能有二进制
    （音频、图片），走文本模式会被 locale 编码搞坏，所以一律字节 + base64。"""
    r = subprocess.run(["git", *args], capture_output=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} 失败: {r.stderr.decode('utf8', 'replace')[:400]}")
    return r.stdout


def main():
    if len(sys.argv) < 2:
        print("用法: push_via_api.py \"commit message\"", file=sys.stderr)
        return 2
    message = sys.argv[1]

    head = git("rev-parse", "HEAD").decode().strip()
    parent = api("GET", "/commits/main")["sha"]
    print(f"local  HEAD : {head[:10]}")
    print(f"remote main : {parent[:10]}")

    # 相对远端新增/修改的文件（git 自己算，比手写可靠）
    changed = git("diff", "--name-only", parent, head).decode().split("\n")
    changed = [c for c in changed if c]
    if not changed:
        print("远端已是最新，无需推送")
        return 0
    print(f"待推送 {len(changed)} 个文件:")
    for c in changed:
        print("  -", c)

    # ---- blob ----
    base_tree = api("GET", f"/git/trees/{parent}?recursive=1")
    blobs = {}
    for path in changed:
        raw = git("cat-file", "-p", f"{head}:{path}".encode("utf8"))
        res = api("POST", "/git/blobs", {
            "content": base64.b64encode(raw).decode(),
            "encoding": "base64",
        })
        blobs[path] = res["sha"]
        print(f"  blob {path} -> {res['sha'][:8]} ({len(raw)} bytes)")

    # ---- tree：只列出改动项，未动的由 base_tree 保留 ----
    tree_entries = [{"path": p, "mode": "100644", "type": "blob", "sha": sha}
                    for p, sha in blobs.items()]
    tree = api("POST", "/git/trees", {"base_tree": base_tree["sha"], "tree": tree_entries})
    print(f"tree -> {tree['sha'][:8]}")

    # ---- commit ----
    commit = api("POST", "/git/commits", {
        "message": message,
        "tree": tree["sha"],
        "parents": [parent],
    })
    print(f"commit -> {commit['sha'][:8]}")

    # ---- ref：若期间被别人推过，这里会 422，直接报出来让人重跑 ----
    try:
        api("PATCH", "/git/refs/heads/main", {"sha": commit["sha"]})
    except urllib.error.HTTPError as e:
        if e.code == 422:
            print("!! main 在这期间被更新了，远端 ref 没动。重跑一次本脚本即可。", file=sys.stderr)
            return 1
        raise
    print(f"PUSHED main -> {commit['sha'][:8]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
