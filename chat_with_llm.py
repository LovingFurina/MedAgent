# ccoding = utf-8
import os
from question_classifier import *
from question_parser import *
from llm_server import *
from build_medicalgraph import *
import re
from evaluation import AnswerEvaluator

entity_parser = QuestionClassifier()
kg = MedicalGraph()
MODEL_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_PROVIDER = "deepseek"
API_KEY = os.getenv("MODEL_API_KEY")
model = ModelAPI(MODEL_URL=MODEL_URL, provider=MODEL_PROVIDER, api_key=API_KEY)


class KGRAG():
    def __init__(self):
        self.cn_dict = {
            "name":"名称",
            "desc":"疾病简介",
            "cause":"疾病病因",
            "prevent":"预防措施",
            "cure_department":"治疗科室",
            "cure_lasttime":"治疗周期",
            "cure_way":"治疗方式",
            "cured_prob":"治愈概率",
            "easy_get":"易感人群",
            "belongs_to":"所属科室",
            "common_drug":"常用药品",
            "do_eat":"宜吃",
            "drugs_of":"生产药品",
            "need_check":"诊断检查",
            "no_eat":"忌吃",
            "recommand_drug":"好评药品",
            "recommand_eat":"推荐食谱",
            "has_symptom":"症状",
            "acompany_with":"并发症",
            "Check":"诊断检查项目",
            "Department":"医疗科目",
            "Disease":"疾病",
            "Drug":"药品",
            "Food":"食物",
            "Producer":"在售药品",
            "Symptom":"疾病症状"
        }
        self.entity_rel_dict = {
            "check":["name", 'need_check'],
            "department":["name", 'belongs_to'],
            "disease":["prevent", "cure_way", "name", "cure_lasttime", "cured_prob", "cause", "cure_department", "desc", "easy_get", 'recommand_eat', 'no_eat', 'do_eat', "common_drug", 'drugs_of', 'recommand_drug', 'need_check', 'has_symptom', 'acompany_with', 'belongs_to'],
            "drug":["name", "common_drug", 'drugs_of', 'recommand_drug'],
            "food":["name"],
            "producer":["name"],
            "symptom":["name", 'has_symptom'],
        }
        self.rel_weight = {
            "has_symptom": 3,
            "common_drug": 2,
            "cause": 2,
            "belongs_to": 1
        }

        self.evaluator = AnswerEvaluator()
        return

    def entity_linking(self, query):
        return entity_parser.check_medical(query)

    def link_entity_rel(self, query, entity, entity_type):
        cate = [self.cn_dict.get(i) for i in self.entity_rel_dict.get(entity_type)]
        prompt = "请判定问题：{query}所提及的是{entity}的哪几个信息，请从{cate}中进行选择，并以列表形式返回。".format(query=query, entity=entity, cate=cate)
        answer = model.chat(query=prompt, history=[])
        cls_rel = set([i for i in re.split(r"[\[。、, ;'\]]", answer)]).intersection(set(cate))
        print([prompt, answer, cls_rel])
        return cls_rel

    def recall_facts(self, cls_rel, entity_type, entity_name, depth=1):
        label_map = {
            "disease": "Disease",
            "symptom": "Symptom",
            "drug": "Drug",
            "food": "Food",
            "department": "Department",
            "check": "Check",
            "producer": "Producer"
        }
        label = label_map.get(entity_type)
        sql = f"""
            MATCH (m:{label} {{name:'{entity_name}'}})-[r]-(n)
            RETURN m,r,n
        """
        print(sql)
        ress = kg.g.run(sql).data()
        triples = set()
        for record in ress:
            m = record.get("m")
            r = record.get("r")
            n = record.get("n")
            start = m["name"]
            end = n["name"]
            rel_type = r.type if hasattr(r, "type") else "相关"
            triples.add(f"<{m['name']},{rel_type},{n['name']}>")
        #print("triples:", triples)
        # 限制最多传给 LLM 的三元组数量
        triples = list(triples)[:30]  # 建议 20~50
        return list(triples)

    def format_prompt(self, query, context):
        triples_str = "\n".join(context) if isinstance(context, (list, set)) else str(context)
        prompt = f"""
你是一名专业的医疗智能问诊助手。

用户当前描述的不适为：
「{query}」

请按照下面结构回答：

【第一步：简要安慰】
先用一句温和的语言回应用户的不适。

【第二步：危险信号排查】
列出需要立即就医的危险信号（如晕倒、肢体无力、剧烈头痛等）。
如果用户未提供相关信息，请提醒其关注这些症状。

【第三步：知识库检索结果】
以下是从医学知识图谱中检索到的相关疾病：
{triples_str}
请明确说明：
“根据医学知识库检索结果，相关疾病可能与以下疾病相关：……”
不要直接机械罗列全部疾病，应进行归类总结（如心血管系统、神经系统、代谢异常等）。

【第四步：医学机制分析（AI推理）】
结合医学常识解释头晕常见的病理机制。

【第五步：建议排查方向】
给出用户可以进行的初步自查方向（血压、血糖、贫血、耳部问题等）。

【第六步：温和结尾】
以温和语气提醒：
线上建议不能替代医生诊断，如症状持续或加重应及时就医。

请避免：
- 过度渲染疾病
- 直接下诊断结论
- 仅罗列疾病名称
输出结构必须分段清晰，如果内容较多，可以适度简洁表达，尤其是第五步和第六步。
"""
        return prompt

    def disease_reasoning(self, query):
        prompt = f"""
你是一名临床医生。
用户描述症状：
{query}
请推断可能相关的疾病名称（列出3~5个常见可能），并按可能性从高到低排序，
仅返回疾病名称列表，不要解释。
格式： ['疾病1','疾病2','疾病3']
"""
        answer = model.chat(query=prompt)
        diseases = re.findall(r"'(.*?)'", answer)
        return diseases

    def chat(self, query):
        print("step1: LLM 推断可能疾病...")
        entities = self.entity_linking(query)
        symptoms = [e for e in entities if e[1] == "symptom"]

        symptom_names = [s for s,_ in symptoms]
        kg_disease_scores = self.retrieve_disease_by_symptom_weighted(symptom_names)

        # 按症状命中数排序，取 TopK
        kg_candidates = sorted(
            kg_disease_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        kg_diseases = [d for d,_ in kg_candidates]

        llm_candidates = self.disease_reasoning(query)

        candidate_diseases = list(dict.fromkeys(kg_diseases + llm_candidates))

        if not candidate_diseases:
            print("LLM未推断出疾病，fallback纯LLM回答")
            return model.chat(query=self.format_prompt(query, []))
        print("推断疾病:", candidate_diseases)
        print("step2: 用候选疾病召回KG...")
        fact_pool = []
        for disease in candidate_diseases:
            sql = f"""
                MATCH (m:Disease {{name:'{disease}'}})-[r]-(n)
                RETURN m.name AS m_name, type(r) AS rel, n.name AS n_name
            """
            ress = kg.g.run(sql).data()
            for record in ress:
                m_name = record.get("m_name")
                rel = record.get("rel")
                n_name = record.get("n_name")
                if not (m_name and rel and n_name):
                    continue

                weight = self.rel_weight.get(rel, 1)
                triple = f"<{m_name},{rel},{n_name}>"

                fact_pool.append((triple, weight))
                
        # 去重（保留最大权重）
        fact_dict = {}
        for t, w in fact_pool:
            fact_dict[t] = max(w, fact_dict.get(t, 0))

        # 按权重排序
        facts = [
            t for t,_ in sorted(
                fact_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ][:30]
        print("step3: 二次LLM综合生成...")
        final_prompt = self.format_prompt(query, facts)
        raw_answer = model.chat(query=final_prompt)
        # 如果返回的是OpenAI格式
        if isinstance(raw_answer, dict):
            answer = raw_answer["choices"][0]["message"]["content"]
        else:
            answer = raw_answer
        evaluation = self.evaluator.evaluate(query, facts, answer)
        return {
            "answer": answer,
            "evaluation": evaluation
        }

    def retrieve_disease_by_symptom(self, symptom_name):
        sql = f"""
            MATCH (s:Symptom {{name:'{symptom_name}'}})<-[:has_symptom]-(d:Disease)
            RETURN d.name as disease
        """
        res = kg.g.run(sql).data()
        return [r["disease"] for r in res]
    
    def retrieve_disease_by_symptom_weighted(self, symptoms):
        """
        symptoms: List[str]
        return: Dict[disease, score]
        """
        disease_score = {}

        for symptom in symptoms:
            sql = f"""
                MATCH (s:Symptom {{name:'{symptom}'}})<-[:has_symptom]-(d:Disease)
                RETURN d.name as disease
            """
            res = kg.g.run(sql).data()
            for r in res:
                d = r["disease"]
                disease_score[d] = disease_score.get(d, 0) + 1  # 命中一个症状 +1

        return disease_score



if __name__ == "__main__":
    chatbot = KGRAG()
    while 1:
        query = input("USER:").strip()
        answer = chatbot.chat(query)
        print("KGRAG_BOT:", answer)