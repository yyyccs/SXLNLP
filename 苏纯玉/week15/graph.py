import re
import json
from py2neo import Graph
import mysql.connector
from mysql.connector import pooling
from loguru import logger
from collections import defaultdict

class DatabaseManager:
    db_config = {
        "host": "127.0.0.1",
        "user": "root",
        "password": "123456",
        "database": "badou",
        "auth_plugin" : "mysql_native_password",
    }

    # 创建连接池
    connection_pool = pooling.MySQLConnectionPool(
        pool_name="my_pool",
        pool_size=20,
        **db_config
    )

    @classmethod
    def get_connection(cls):
        return cls.connection_pool.get_connection()

    @classmethod
    def query_data(cls, sql, *args):
        # 在类方法中获取连接
        connection = cls.get_connection()
        cursor = connection.cursor()

        try:
            cursor.execute(sql, *args)
            result = cursor.fetchall()
            return result
        finally:
            cursor.close()
            connection.close()  # 在方法结束时释放连接

    @classmethod
    def insert_data(cls, sql, *args):
        # 在类方法中获取连接
        connection = cls.get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(sql, *args)
            connection.commit()
        finally:
            cursor.close()
            connection.close()  # 在方法结束时释放连接

    @classmethod
    def insert_many_data(cls, sql, *args):
        # 在类方法中获取连接
        connection = cls.get_connection()
        cursor = connection.cursor()
        try:
            cursor.executemany(sql, *args)
            connection.commit()
        finally:
            cursor.close()
            connection.close()  # 在方法结束时释放连接

    @classmethod
    def update_data(cls, sql, *args):
        # 在类方法中获取连接
        connection = cls.get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(sql, *args)
            connection.commit()
        finally:
            cursor.close()
            connection.close()  # 在方法结束时释放连接

    @classmethod
    def close_pool(cls):
        cls.connection_pool.close()

rel_data =list()
attr_data = list()
label_data = {}

with open("triplets_head_rel_tail.txt", encoding="utf8") as f:
    for line in f:
        head, relation, tail = line.strip().split("\t")
        for label in ["歌曲", "专辑", "电影", "电视剧"]:
            if label in head:
                head = re.sub(r'（.*?）', '', head)
                label_data[head] = label
        head = re.sub(r'（.*?）', '', head)
        # relation = re.sub(r'（.*?）', '', relation)
        # tail = re.sub(r'（.*?）', '', tail)
        # logger.info(f"head: {head}, relation: {relation}, tail: {tail}")
        rel_data.append((head, relation, tail))
# DatabaseManager.insert_many_data('insert into rel_tab (head,rel, tail) values (%s, %s, %s)', rel_data)

with open("triplets_enti_attr_value.txt", encoding="utf8") as f:
    for line in f:
        enti, attr, val = line.strip().split("\t")
        for label in ["歌曲", "专辑", "电影", "电视剧"]:
            if label in enti:
                enti = re.sub(r'（.*?）', '', enti)
                label_data[enti] = label
        enti = re.sub(r'（.*?）', '', enti)
        # attr = re.sub(r'（.*?）', '', attr)
        val = re.sub(r'\(.*?\)', '', val)

        # logger.info(f"enti: {enti}, attr: {attr}, val: {val}")
        attr_data.append((enti, attr, val))
# DatabaseManager.insert_many_data('insert into attr_tab (enti,attr, value) values (%s, %s, %s)', attr_data)

data = defaultdict(set)
for enti, label in label_data.items():
    data["enti"].add(enti)
    data["labels"].add(label)

[data['rel'].add(i[0]) for i in DatabaseManager.query_data('select rel from rel_tab')]
[data['enti'].add(i[0]) for i in DatabaseManager.query_data('select tail from rel_tab')]
[data['enti'].add(i[0]) for i in DatabaseManager.query_data('select enti from attr_tab')]
[data['attr'].add(i[0]) for i in DatabaseManager.query_data('select attr from attr_tab')]

data = dict((x, list(y)) for x, y in data.items())

with open("schema.json", "w", encoding="utf8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)