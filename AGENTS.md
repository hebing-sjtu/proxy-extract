# 与这个仓库协作的约定

## 改完就推

这个仓库的代码是给别的机器跑的——双节点 Docker，靠 `git pull` 取更新。所以改完不要停在
工作区：**自己 commit 并 push**，不用问。一个没推上去的修复对节点来说等于不存在，而
"我改好了"和节点上真的有这份代码之间的落差，已经造成过一次在旧脚本上重复排查同一个
错误的浪费。

## 给节点的命令自带 `git pull`

凡是写给操作者、要在节点上执行的命令，如果依赖刚改的代码，就把 `git pull` 直接写进命令
里，而不是假设对方记得先拉：

```bash
cd /workspace/fastvideo_datapipe && git pull && scripts/setup_docker_env.sh --with-flicker
```
