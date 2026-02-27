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
        self.session_memory = {}
        self.max_rounds = 5
        self.evaluator = AnswerEvaluator()

    def entity_linking(self, query):
        raw_entities = entity_parser.check_medical(query)

        normalized_entities = []

        # 情况1：如果返回的是 dict，例如 {"感冒": ["disease"]}
        if isinstance(raw_entities, dict):
            for name, type_list in raw_entities.items():
                for etype in type_list:
                    normalized_entities.append({
                        "name": name.strip(),
                        "type": etype.lower()
                    })

        # 情况2：如果返回的是 list of tuple，例如 [("感冒","disease")]
        elif isinstance(raw_entities, list):
            for item in raw_entities:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    normalized_entities.append({
                        "name": item[0].strip(),
                        "type": item[1].lower()
                    })

        return normalized_entities

    def link_entity_rel(self, query, entity, entity_type):
        cate = [self.cn_dict.get(i) for i in self.entity_rel_dict.get(entity_type)]
        prompt = "请判定问题：{query}所提及的是{entity}的哪几个信息，请从{cate}中进行选择，并以列表形式返回。".format(query=query, entity=entity, cate=cate)
        answer = model.chat(query=prompt, _=[])
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
    
    def classify_intent(self, query):

        scores = {
            "DRUG_QUERY": 0,
            "DISEASE_PROPERTY": 0,
            "SYMPTOM_DIAGNOSIS": 0
        }

        drug_keywords = ["吃什么药", "用什么药", "推荐药", "药物"]
        property_keywords = ["是什么", "原因", "病因", "属于", "科室"]
        symptom_keywords = ["疼", "痛", "难受", "发热", "头晕", "恶心", "咳血", "失力"]

        # DRUG 打分
        for k in drug_keywords:
            if k in query:
                scores["DRUG_QUERY"] += 3

        # PROPERTY 打分
        for k in property_keywords:
            if k in query:
                scores["DISEASE_PROPERTY"] += 2

        # SYMPTOM 打分
        if "我" in query:
            scores["SYMPTOM_DIAGNOSIS"] += 1

        for k in symptom_keywords:
            if k in query:
                scores["SYMPTOM_DIAGNOSIS"] += 2

        # 选择最高分
        best_intent = max(scores, key=scores.get)

        # 如果全是0，默认问诊
        if scores[best_intent] == 0:
            return "SYMPTOM_DIAGNOSIS"

        print("意图得分:", scores)

        return best_intent

    def format_prompt(self, query, context, history_summary=""):
        triples_str = "\n".join(context) if isinstance(context, (list, set)) else str(context)
        prompt = f"""
你是一名专业的医疗智能问诊助手。

用户当前描述的不适为：
「{query}」
以下是历史对话摘要：
{history_summary}

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

    def chat(self, query, session_id="default"):
        # 1️⃣ 初始化 session
        if session_id not in self.session_memory:
            self.session_memory[session_id] = {
                "history": [],
                "last_disease": None,
                "last_intent": None
            }

        session = self.session_memory[session_id]

        # 2️⃣ 意图识别
        intent = self.classify_intent(query)
        session["last_intent"] = intent
        print("识别意图:", intent)

        # 3️⃣ 分发处理
        if intent == "SYMPTOM_DIAGNOSIS":
            result = self.handle_symptom_diagnosis(query, session)

        elif intent == "DISEASE_PROPERTY":
            result = self.handle_property_query(query, session)

        elif intent == "DRUG_QUERY":
            result = self.handle_drug_query(query, session)

        else:
            result = {"answer": "暂未识别问题类型。", "evaluation": {}}

        # 4️⃣ 保存对话历史
        session["history"].append(f"用户: {query}")
        session["history"].append(f"助手: {result['answer']}")

        # 只保留最近10条
        session["history"] = session["history"][-10:]

        return result
    
    def handle_symptom_diagnosis(self, query, session):
        print("step1: LLM 推断可能疾病...")
        entities = self.entity_linking(query)
        symptoms = [e for e in entities if e["type"] == "symptom"]
        symptom_names = [e["name"] for e in symptoms]

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

        if candidate_diseases:
            session["last_disease"] = candidate_diseases[0]

        if not candidate_diseases:
            print("LLM未推断出疾病，fallback纯LLM回答")
            history_summary = self.compress_history(session["history"])
            return model.chat(query=self.format_prompt(query, [], history_summary))
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
        history_summary = self.compress_history(session["history"])
        print("传给LLM的历史摘要:", history_summary)
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
    
    def handle_property_query(self, query, session):

        entities = self.entity_linking(query)

        if not entities:
            return {"answer": "知识库中未识别到相关疾病。", "evaluation": {}}

        entity_name = entities[0]["name"]
        entity_type = entities[0]["type"]

        if entity_type == "disease":
            session["last_disease"] = entity_name

        if entity_type != "disease":
            return {"answer": "当前问题更适合问诊模式。", "evaluation": {}}

        # 用 LLM 判定问的是哪个属性
        rels = self.link_entity_rel(query, entity_name, entity_type)

        if not rels:
            return {"answer": "未识别到具体查询属性。", "evaluation": {}}

        facts = self.recall_facts(rels, entity_type, entity_name)

        if not facts:
            return {"answer": f"知识库中暂无 {entity_name} 的相关信息。", "evaluation": {}}

        prompt = f"""
    请根据以下医学知识库信息回答问题：
    {facts}

    问题：{query}
    请明确、简洁回答，不需要问诊结构。
    """

        answer = model.chat(query=prompt)
        return {"answer": answer, "evaluation": {}}
    
    
    def handle_drug_query(self, query, session):
        entities = self.entity_linking(query)
        print("entities:", entities)

        disease = None

        for e in entities:
            if e["type"] == "disease":
                disease = e["name"]
                break

        print("最终识别疾病:", disease)
        if not disease:
            # 尝试使用上一轮疾病
            disease = session.get("last_disease")

        if not disease:
            return {
                "answer": "未识别到具体疾病，请补充说明。",
                "evaluation": {}
            }

        sql = f"""
            MATCH (d:Disease {{name:'{disease}'}})-[:common_drug]->(n)
            RETURN n.name AS drug
        """

        res = kg.g.run(sql).data()
        drugs = [r["drug"] for r in res]

        if not drugs:
            return {
                "answer": f"知识库中暂无 {disease} 的推荐药物信息。",
                "evaluation": {}
            }

        prompt = f"""
    疾病：{disease}
    推荐药物：{drugs}

    请简洁说明这些药物的作用，并提醒用户遵医嘱。
    """

        answer = model.chat(query=prompt)

        return {"answer": answer, "evaluation": {}}

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
    
    def compress_history(self, history):
        """
        将历史对话压缩为简要摘要
        """
        if not history:
            return ""

        prompt = f"""
    请将以下对话压缩为不超过100字的医学问诊摘要，
    保留：疾病、症状、用药信息。

    对话：
    {history}

    仅返回摘要。
    """
        summary = model.chat(query=prompt)
        return summary



if __name__ == "__main__":
    chatbot = KGRAG()
    while 1:
        query = input("USER:").strip()
        answer = chatbot.chat(query)
        print("KGRAG_BOT:", answer)