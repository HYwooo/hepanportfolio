# Dockerfile to create a reusable Python 3.13 + Playwright environment

# 使用官方 Python 3.13 slim 镜像作为基础
FROM python:3.13-slim-bookworm

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 设置一个工作目录，尽管我们不会在这里放代码
WORKDIR /app

# 复制依赖文件。这是唯一需要从你仓库中获取的文件。
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 安装 Playwright 的系统级依赖和浏览器
RUN playwright install chromium --with-deps
# 启动容器后什么都不做
CMD ["tail", "-f", "/dev/null"] # 保持容器运行
