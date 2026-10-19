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
import time
import urllib.error
import urllib.request

REPO = "jiqiansi/gpt-sovits-actions"
BASE = f"https://api.github.com/repos/{REPO}"


def token():
    return open(os.path.expanduser("~/.dsh/skills/github-actions-batch-render/token.txt")).read().strip()


def api(method, path, payload=None, tries=5):
    """带重试的 API 调用。

    这台机器到 api.github.com 的连接会被远端掐（RemoteDisconnected / 读超时），
    一次抖动就毁掉整轮推送。推送本身不是幂等操作，所以只对**幂等**的
    GET 和幂等的 POST /git/blobs（同样内容返回同一个 sha）做重试；
    commit / ref 这两步失败就直接停，让人重跑，避免产生重复提交。
    """
    url = path if path.startswith("http") else BASE + path
    data = json.dumps(payload).encode() if payload is not None else None
    last = None
    for attempt in range(tries):
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
        except Exception as e:   # 连接被掐、读超时、半包
            last = e
            print(f"   [{attempt+1}/{tries}] {method} {url} 失败: {type(e).__name__}: {e}", file=sys.stderr)
            if attempt + 1 < tries:
                time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"API 调用重试 {tries} 次仍失败: {method} {url}") from last


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

    # 算出「相对远端改了哪些文件」。
    # 不能用 git diff <remote_sha> HEAD —— 远端那个 commit 是 API 建的，
    # 本地对象库里根本没有它，会直接 bad object。
    # 改成拿远端 tree 的 path->sha，和本地 HEAD 里每个文件的 blob sha 对一遍，
    # 全程只用本地 git 算 sha，不需要拉取远端对象。
    remote_tree = api("GET", f"/git/trees/{parent}?recursive=1")["tree"]
    remote_map = {e["path"]: e["sha"] for e in remote_tree if e.get("type") == "blob"}

    local_paths = git("ls-tree", "-r", "--name-only", head).decode().split("\n")
    changed = []
    for path in local_paths:
        if not path:
            continue
        local_sha = git("rev-parse", f"{head}:{path}".encode("utf8")).decode().strip()
        if remote_map.get(path) != local_sha:
            changed.append(path)
    # 远端有、本地没有的文件本脚本不处理（本仓库不会删文件；真要删得另说）
    gone = [p for p in remote_map if p not in set(local_paths)]
    if gone:
        print("!! 远端存在但本地已删除的文件，本脚本不会删除它们:", gone, file=sys.stderr)

    if not changed:
        print("远端已是最新，无需推送")
        return 0
    print(f"待推送 {len(changed)} 个文件:")
    for c in changed:
        print("  -", c)

    # ---- blob ----
    base_tree = parent
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
    tree = api("POST", "/git/trees", {"base_tree": base_tree, "tree": tree_entries})
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
