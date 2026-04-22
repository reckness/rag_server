import uuid

def generate_uuid():
    """生成UUID作为ID，去掉连字符"""
    return str(uuid.uuid4()).replace('-', '')