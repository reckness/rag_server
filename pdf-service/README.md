# 企业级文档转PDF服务

## 项目简介

这是一个企业级的文档转PDF服务，支持多种文件格式的转换，包括Word、Excel、TXT、HTML、Markdown等。

## 支持的文件格式

- doc
- docx
- xls
- xlsx
- ppt
- pptx
- txt
- html
- md
- csv

## 项目结构

```
doc2pdf-service
│
├── app
│   ├── api
│   │   └── convert_api.py        # 文件转换接口
│   │
│   ├── core
│   │   ├── config.py             # 配置
│   │   └── logger.py             # 日志
│   │
│   ├── services
│   │   ├── converter.py          # 主转换逻辑
│   │   ├── libreoffice.py        # LibreOffice调用
│   │   └── markdown_converter.py # md转html
│   │
│   ├── utils
│   │   ├── file_utils.py         # 文件处理
│   │   └── cmd_utils.py          # shell命令封装
│   │
│   └── main.py                   # FastAPI启动
│
├── scripts
│   └── batch_convert.py          # 批量转换脚本
│
├── data
│   ├── upload                    # 上传文件
│   └── pdf                       # 转换结果
│
├── requirements.txt
├── Dockerfile
└── README.md
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### API使用

#### 上传文件进行转换

```bash
curl -F "file=@test.docx" http://localhost:8000/convert
```

#### 响应示例

```json
{
  "pdf": "data/pdf/test.pdf"
}
```

## 部署

### Docker部署

```bash
docker build -t doc2pdf-service .
docker run -p 8000:8000 doc2pdf-service
```

## 生产环境优化

### LibreOffice Socket模式

生产环境建议使用LibreOffice Socket模式以提高转换速度：

```bash
soffice --headless \
--accept="socket,host=127.0.0.1,port=2002;urp;"
```

然后使用unoconv库进行连接，可提高转换速度5-10倍。

## 企业架构升级

如果需要接入PageIndex/RAGFlow，建议升级为以下架构：

```
doc-preprocess-service
│
├── converter-service      # 文档转PDF
├── parser-service         # PageIndex解析
├── vector-service         # 向量化
├── storage-service        # MinIO
└── api-gateway
```

## 完整流程

1. 文档上传
2. 转换服务
3. PDF
4. PageIndex解析
5. JSON树结构
6. 向量数据库

## 适用场景

- PageIndex
- RAGFlow
- 企业知识库
- 文档管理系统