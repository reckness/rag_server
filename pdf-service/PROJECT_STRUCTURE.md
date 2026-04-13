# 企业级文档转PDF服务项目结构

## 项目概述

这是一个企业级的文档转PDF服务，支持多种文件格式的转换，包括Word、Excel、TXT、HTML、Markdown等。该服务采用FastAPI框架构建，可在Windows和Linux环境下运行，支持API调用，可与PageIndex/RAGFlow等知识库系统集成。


## 目录结构

```
doc2pdf-service/
├── app/                      # 应用主目录
│   ├── api/                  # API接口层
│   │   └── convert_api.py    # 文件转换接口
│   ├── core/                 # 核心配置
│   │   ├── config.py         # 配置管理
│   │   └── logger.py         # 日志管理
│   ├── services/             # 业务服务层
│   │   ├── converter.py      # 主转换逻辑
│   │   ├── libreoffice.py    # LibreOffice调用
│   │   └── markdown_converter.py # Markdown转HTML
│   ├── utils/                # 工具函数
│   │   ├── file_utils.py     # 文件处理工具
│   │   └── cmd_utils.py      # 命令执行工具
│   └── main.py               # FastAPI应用入口
├── data/                     # 数据目录
│   ├── upload/               # 上传文件存储
│   └── pdf/                  # 转换结果存储
├── scripts/                  # 脚本目录
│   └── batch_convert.py      # 批量转换脚本
├── Dockerfile                # Docker构建文件
├── Dockerfile.base           # 基础镜像构建文件
├── docker-compose.yml        # Docker Compose配置
├── requirements.txt          # Python依赖
└── README.md                 # 项目说明文档
```


## 核心文件说明

### 1. API接口层
- **app/api/convert_api.py**：提供文件上传和转换的RESTful API接口，接收文件并返回转换后的PDF路径。

### 2. 业务服务层
- **app/services/converter.py**：核心转换逻辑，根据文件类型调用不同的转换服务。
- **app/services/libreoffice.py**：LibreOffice调用封装，负责将文档转换为PDF。
- **app/services/markdown_converter.py**：Markdown转HTML服务，处理Markdown文件。

### 3. 核心配置
- **app/core/config.py**：应用配置管理，包含上传目录、PDF存储目录等配置项。
- **app/core/logger.py**：日志配置，记录应用运行状态和错误信息。

### 4. 工具函数
- **app/utils/file_utils.py**：文件处理工具，提供目录创建、文件扩展名获取等功能。
- **app/utils/cmd_utils.py**：命令执行工具，封装子进程调用逻辑。

### 5. 应用入口
- **app/main.py**：FastAPI应用入口，注册API路由并启动服务。

### 6. 脚本工具
- **scripts/batch_convert.py**：批量转换脚本，支持批量处理目录下的文件。

### 7. 部署配置
- **Dockerfile**：Docker镜像构建文件，包含Python环境和LibreOffice安装。
- **docker-compose.yml**：Docker Compose配置，定义服务运行参数。
- **requirements.txt**：Python依赖包列表。


## 技术栈

| 类别 | 技术/工具 | 用途 |
|------|-----------|------|
| 后端框架 | FastAPI | 构建RESTful API服务 |
| 文档转换 | LibreOffice | 核心文档转PDF功能 |
| Markdown处理 | markdown | Markdown转HTML |
| 容器化 | Docker | 服务部署和运行 |
| 编排工具 | Docker Compose | 多容器管理 |
| 依赖管理 | pip | Python包管理 |


## API接口

### 转换接口
- **URL**：`/convert`
- **方法**：`POST`
- **参数**：`file`（上传文件）
- **返回**：`{"pdf": "data/pdf/文件名.pdf"}`

### 调用示例
```bash
# curl命令
curl -F "file=@test.docx" http://localhost:18081/convert

# 响应示例
{
  "pdf": "data/pdf/test.pdf"
}
```


## 支持的文件格式

- **Word**：doc, docx
- **Excel**：xls, xlsx
- **PowerPoint**：ppt, pptx
- **文本**：txt, csv
- **网页**：html
- **Markdown**：md


## 部署方式

### 1. 本地部署
```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 2. Docker部署
```bash
# 构建镜像
docker build -t doc2pdf-service .

# 运行容器
docker run -p 18081:8000 doc2pdf-service
```

### 3. Docker Compose部署
```bash
# 启动服务
docker-compose up -d

# 停止服务
docker-compose down
```


## 生产环境优化

### 1. LibreOffice Socket模式
生产环境建议使用LibreOffice Socket模式以提高转换速度：
```bash
soffice --headless \
--accept="socket,host=127.0.0.1,port=2002;urp;"
```

### 2. 中文支持
通过在Dockerfile中安装中文字体和配置语言环境，确保中文文档转换正确：
```dockerfile
RUN apt-get install -y \
    fonts-wqy-zenhei \
    fonts-wqy-microhei \
    language-pack-zh-hans
```

### 3. 缓存优化
利用Docker的缓存机制，优化构建过程，避免每次构建都下载LibreOffice。


## 企业架构扩展

对于企业级应用，建议升级为以下架构：

```
doc-preprocess-service/
├── converter-service      # 文档转PDF（当前项目）
├── parser-service         # PageIndex解析
├── vector-service         # 向量化
├── storage-service        # MinIO存储
└── api-gateway            # API网关
```

### 完整流程
1. 文档上传
2. 转换服务（当前项目）
3. PDF生成
4. PageIndex解析
5. JSON树结构
6. 向量数据库存储


## 适用场景

- **PageIndex**：文档解析和索引
- **RAGFlow**：检索增强生成
- **企业知识库**：文档管理和检索
- **文档管理系统**：批量文档处理


## 配置说明

### 环境变量
- `TZ`：时区设置（默认：Asia/Shanghai）
- `LANG`：语言环境（默认：zh_CN.UTF-8）

### 服务配置
- **端口**：18081（可在docker-compose.yml中修改）
- **数据目录**：./data（映射到容器内的/app/data）
- **重启策略**：unless-stopped（服务异常时自动重启）


## 故障排查

### 常见问题
1. **LibreOffice未找到**：确保LibreOffice已正确安装或Docker镜像已包含LibreOffice
2. **中文乱码**：检查中文字体是否已安装，语言环境是否正确配置
3. **转换失败**：检查源文件格式是否支持，文件路径是否正确
4. **服务启动失败**：检查端口是否被占用，依赖是否已安装

### 日志查看
```bash
# Docker容器日志
docker-compose logs -f
```


## 总结

本项目是一个功能完整的企业级文档转PDF服务，具备以下特点：
- 支持多种文件格式转换
- 跨平台运行（Windows/Linux）
- 易于部署和扩展
- 可与知识库系统集成
- 生产环境优化

通过Docker容器化部署，可快速在企业环境中部署和使用，为文档预处理提供可靠的服务支持。