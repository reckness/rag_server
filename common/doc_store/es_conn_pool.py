#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import logging
import time
from elasticsearch import Elasticsearch

from common.config import ELASTICSEARCH_HOST, ELASTICSEARCH_USER, ELASTICSEARCH_PASSWORD
from common.decorator import singleton

ATTEMPT_TIME = 2


@singleton
class ElasticSearchConnectionPool:

    def __init__(self):
        # 直接使用从 config.py 导入的配置
        self.ES_CONFIG = {
            "hosts": ELASTICSEARCH_HOST,
            "username": ELASTICSEARCH_USER,
            "password": ELASTICSEARCH_PASSWORD,
            "verify_certs": False
        }

        for _ in range(ATTEMPT_TIME):
            try:
                if self._connect():
                    break
            except Exception as e:
                logging.warning(f"{str(e)}. Waiting Elasticsearch {self.ES_CONFIG['hosts']} to be healthy.")
                time.sleep(5)

        if not hasattr(self, "es_conn") or not self.es_conn or not hasattr(self, "info"):
            msg = f"Elasticsearch {self.ES_CONFIG['hosts']} is unhealthy in 10s."
            logging.error(msg)
            raise Exception(msg)

    def _connect(self):
        # 处理 hosts 配置
        hosts = self.ES_CONFIG["hosts"]
        if isinstance(hosts, str):
            hosts = hosts.split(",")
        
        # 自定义请求头，解决版本兼容问题
        headers = {
            "Accept": "application/vnd.elasticsearch+json; compatible-with=8",
            "Content-Type": "application/json"
        }
        
        self.es_conn = Elasticsearch(
            hosts,
            basic_auth=(self.ES_CONFIG["username"], self.ES_CONFIG[
                "password"]) if "username" in self.ES_CONFIG and "password" in self.ES_CONFIG else None,
            verify_certs= self.ES_CONFIG.get("verify_certs", False),
            request_timeout=600,
            headers=headers )
        if self.es_conn:
            self.info = self.es_conn.info()
            # 检查 Elasticsearch 版本
            v = self.info.get("version", {"number": "8.11.3"})
            v = v["number"].split(".")[0]
            if int(v) < 8:
                msg = f"Elasticsearch version must be greater than or equal to 8, current version: {v}"
                logging.error(msg)
                raise Exception(msg)
            return True
        return False

    def get_conn(self):
        return self.es_conn

    def refresh_conn(self):
        try:
            if self.es_conn and self.es_conn.info():
                return self.es_conn
        except Exception:
            pass
        # close current if exist
        if self.es_conn:
            self.es_conn.close()
        self._connect()
        return self.es_conn

    def __del__(self):
        if hasattr(self, "es_conn") and self.es_conn:
            self.es_conn.close()


ES_CONN = ElasticSearchConnectionPool()
