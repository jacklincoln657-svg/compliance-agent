# 构建沙箱镜像（在仓库根或本目录执行均可）
# Windows PowerShell 示例：
#   cd cyber_lobster_project\docker
#   docker build -t cyberlobster-sandbox:py311 -f Dockerfile.sandbox .

# 根目录 .env（可选）：
#   SANDBOX_BACKEND=auto|docker|local
#   SANDBOX_DOCKER_IMAGE=cyberlobster-sandbox:py311
#
# auto：检测到 Docker 且镜像可用则走容器，否则本机子进程
# docker：优先容器；创建/镜像失败时自动回退本机并在工具返回里提示
# local：强制本机（无 Docker 时开发最稳）
