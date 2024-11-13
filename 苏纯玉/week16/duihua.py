import re
import json
import pandas as pd
import os
from loguru import logger


class Duihua:
    def __init__(self):
        self.node_info = {}
        self.slot_info = {}
        self.load_data()

    def load_data(self):
        self.load_scenarios()
        self.load_slots()

    def load_scenarios(self):
        sc_nam = 'sc_'
        with open('scenario-买衣服.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        for node in data:
            node_id = node['id']
            node_id = sc_nam + str(node_id)
            if 'childnode' in node:
                for i, child in enumerate(node['childnode']):
                    node['childnode'][i] = sc_nam + str(child)
            self.node_info[node_id] = node
        # logger.info(self.node_info)

    def load_slots(self):
        df = pd.read_excel('slot_fitting_templet.xlsx')
        for index, row in df.iterrows():
            slot = row['slot']
            query = row['query']
            value = row['values']
            # logger.info(slot, query, value)
            self.slot_info[slot] = [query,value]

    def get_sentence_similarity(self, string1, string2):
        # print("计算  %s %s"%(string1, string2))
        jaccard_distance = len(set(string1) & set(string2)) / len(set(string1) | set(string2))
        return jaccard_distance
    def get_node_score(self, memory, info):
        query = memory['query']
        intents = info['intent']
        scores = []
        for intent in intents:
            sentence_sim = self.get_sentence_similarity(query, intent)
            scores.append(sentence_sim)
        return  max(scores)

    def get_intent(self, memory):
        max_score = -1
        best_node = None
        for node in memory['available_node']:
            info = self.node_info[node]
            score = self.get_node_score(memory, info)
            if score > max_score:
                max_score = score
                best_node = node
        memory['best_node'] = best_node
        memory['max_score'] = max_score
        return memory

    def get_slot(self, memory):
        best_node = memory['best_node']
        slots = self.node_info[best_node].get('slot', [])
        for slot in slots:
            _, val = self.slot_info[slot]
            if re.search(val, memory['query']):
                memory[slot] = re.search(val, memory['query']).group()
        return memory

    def nlu(self, memory):
        #意图识别
        memory = self.get_intent(memory)
        #槽位抽取
        memory = self.get_slot(memory)
        return memory

    def dst(self, memory):
        #对话状态跟踪
        best_node = memory['best_node']
        slots = self.node_info[best_node].get('slot', [])
        for slot in slots:
            if slot not in memory:
                memory['need_slot'] = slot
                return memory
        memory['need_slot'] = None
        return memory

    def pol(self, memory):
        if memory['need_slot']:
            memory['action'] = 'ask'
            memory['available_node'] = [memory['best_node']]
        else:
            memory['action'] = 'answer'
            memory['available_node'] = self.node_info[memory['best_node']].get('childnode', [])
        return memory

    def nlg(self, memory):
        if memory['action'] == 'answer':
            memory['response'] = self.node_info[memory['best_node']]['response']
        else:
            slot = memory['need_slot']
            query, _ = self.slot_info[slot]
            memory['response'] = query
        return memory

    def query(self, query, memory):
        memory['query'] = query
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.pol(memory)
        memory = self.nlg(memory)
        return memory


if __name__ == "__main__":
    dh = Duihua()
    memory = {'available_node': ['sc_node1']}
    while True:
        # query = '我要买短袖'
        query = input("请输入问题：")
        memory = dh.query(query, memory)
        print('bot_resp:'+memory['response'])
