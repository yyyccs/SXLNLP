from graph import *
import pandas as pd
import itertools

class QA:
    def __init__(self):
        self.qa_match = []
        schema_path = "schema.json"
        self.load(schema_path)
        print("知识图谱问答系统加载完毕！\n===============")

    def load(self, schema_path):
        with open(schema_path, encoding="utf8") as f:
            schema = json.load(f)
        self.rel_set = set(schema["rel"])
        self.enti_set = set(schema["enti"])
        self.label_set = set(schema["labels"])
        self.attr_set = set(schema["attr"])
        match_sql = [('%ENT%的%ATT%是什么', "select `value` from attr_tab where attr = '%ATT%' and enti = '%ENT%'", {"%ENT%":1, "%ATT%":1}),
                     ('%ENT0%的%ENT1%什么关系', "select rel from rel_tab where (head = %ENTI0% AND tail = %ENTI1%) OR (head=%ENTI1% and tail = %ENTI0%)", {"%ENT%":2})]
        for i in match_sql:
            self.qa_match.append({
                'question': i[0],
                'sql': i[1],
                'check_num':i[2]
            })


    def parse_sentence(self, sentence):
        entitys = self.get_mention_entitys(sentence)
        relations = self.get_mention_relations(sentence)
        labels = self.get_mention_labels(sentence)
        attributes = self.get_mention_attributes(sentence)
        return {"%ENT%":entitys,
                "%REL%":relations,
                "%LAB%":labels,
                "%ATT%":attributes}

    #获取问题中谈到的实体，可以使用基于词表的方式，也可以使用NER模型
    def get_mention_entitys(self, sentence):
        return re.findall("|".join(self.enti_set), sentence)

    # 获取问题中谈到的关系，也可以使用各种文本分类模型
    def get_mention_relations(self, sentence):
        return re.findall("|".join(self.rel_set), sentence)

    # 获取问题中谈到的属性
    def get_mention_attributes(self, sentence):
        return re.findall("|".join(self.attr_set), sentence)

    # 获取问题中谈到的标签
    def get_mention_labels(self, sentence):
        return re.findall("|".join(self.label_set), sentence)

    def check_sql_info_valid(self, info, sql_check):
        for key, required_count in sql_check.items():
            if len(info.get(key, [])) < required_count:
                return False
        return True

    def get_combinations(self, cypher_check, info):
        # 初始化 slot_values 列表，用于存储每个槽位的组合
        slot_values = []

        # 生成每个槽位的组合
        for key, required_count in cypher_check.items():
            if len(info[key]) < required_count:
                raise ValueError(f"Not enough elements in {key} to satisfy the required count {required_count}")
            # 生成组合并将其转换为列表
            combinations = list(itertools.combinations(info[key], required_count))
            # 如果 required_count > 1，则需要考虑不同顺序的组合
            if required_count > 1:
                permutations = [list(itertools.permutations(comb)) for comb in combinations]
                slot_values.append(permutations)
            else:
                slot_values.append(combinations)

        # 生成所有槽位组合的笛卡尔积
        all_combinations = list(itertools.product(*slot_values))

        # 将每个组合中的元组展开成一个单一的字典
        result = []
        for combination in all_combinations:
            combination_dict = {}
            for i, (key, required_count) in enumerate(cypher_check.items()):
                if required_count == 1:
                    combination_dict[key] = combination[i][0]
                else:
                    for j in range(required_count):
                        combination_dict[f"{key}{j}"] = combination[i][j]
            result.append(combination_dict)

        return result

    def replace_token_in_string(self, string, combination):
        for key, value in combination.items():
            string = string.replace(key, value)
        return string

    def expand_templet(self, question, sql, check_num, info):
        combinations = self.get_combinations(check_num, info)
        templet_cpyher_pair = []
        for combination in combinations:
            replaced_templet = self.replace_token_in_string(question, combination)
            replaced_cypher = self.replace_token_in_string(sql, combination)
            # replaced_answer = self.replace_token_in_string(answer, combination)
            answer = DatabaseManager.query_data(replaced_cypher)
            templet_cpyher_pair.append([replaced_templet, replaced_cypher, answer])
        return templet_cpyher_pair

    def sentence_similarity_function(self, string1, string2):
        # print("计算  %s %s"%(string1, string2))
        jaccard_distance = len(set(string1) & set(string2)) / len(set(string1) | set(string2))
        return jaccard_distance

    def sql_match(self, info):
        temp_sql_pair = []
        for i in self.qa_match:
            question = i['question']
            sql = i['sql']
            check_num = i['check_num']
            if self.check_sql_info_valid(info, check_num):
                temp_sql_pair += self.expand_templet(question, sql, check_num, info)
        return temp_sql_pair


    def query(self, sentence):
        info = self.parse_sentence(sentence)
        temp_sql_pair = self.sql_match(info)
        result = []
        for i in temp_sql_pair:
            question = i[0]
            sql = i[1]
            answer = i[2]
            score = self.sentence_similarity_function(question, sentence)
            result.append([question, sql, answer, score])
        result = sorted(result, reverse=True, key=lambda x: x[3])
        return result[0][2]


if __name__ == "__main__":
    qa = QA()
    print(qa.query('屋顶的歌曲原唱'))
    print(qa.query('那家是不能说的秘密的出品公司'))
#知识图谱问答系统加载完毕！
#===============
#[('吴宗宪，温岚',)]
#[('安乐影片有限公司',)]
