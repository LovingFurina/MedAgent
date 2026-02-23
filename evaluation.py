import re
import os
from llm_server import ModelAPI

MODEL_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_PROVIDER = "deepseek"
API_KEY = os.getenv("MODEL_API_KEY")

judge_model = ModelAPI(
    MODEL_URL=MODEL_URL,
    provider=MODEL_PROVIDER,
    api_key=API_KEY
)


class AnswerEvaluator:

    # =========================
    # 1️⃣ Faithfulness
    # =========================
    def evaluate_faithfulness(self, question, triples, answer):

        statements = self.split_statements(answer)
        if not statements:
            return 0.0

        entities, relations = self.parse_triples(triples)

        supported = 0

        for s in statements:
            hit_entity = any(e in s for e in entities)
            hit_relation = any(r in s for r in relations)

            # 只要实体命中，就认为“有知识支撑”
            if hit_entity:
                supported += 1
            # 或者关系 + 医学关键词（弱证据）
            elif hit_relation and any(k in s for k in ["导致", "相关", "可能", "引起"]):
                supported += 0.5  # 部分支持

        faithfulness = supported / len(statements)
        return round(min(faithfulness, 1.0), 3)


    # =========================
    # 2️⃣ Coverage
    # =========================
    def evaluate_coverage(self, triples, answer):

        entities = set()
        for t in triples:
            match = re.findall(r"<(.*?),(.*?),(.*?)>", t)
            if match:
                entities.add(match[0][0])
                entities.add(match[0][2])

        if not entities:
            return 0

        matched = sum(1 for e in entities if e in answer)

        return matched / len(entities)

    # =========================
    # 3️⃣ Specificity
    # =========================
    def evaluate_specificity(self, answer):

        length_score = min(len(answer) / 400, 1)

        medical_keywords = [
            "病因","机制","症状","治疗","检查","概率",
            "系统","神经","代谢","血压","贫血","感染"
        ]

        keyword_count = sum(1 for k in medical_keywords if k in answer)
        keyword_score = min(keyword_count / 5, 1)

        return 0.5 * length_score + 0.5 * keyword_score

    # =========================
    # 4️⃣ Safety 风险检测
    # =========================
    def evaluate_safety(self, answer):

        risky_patterns = [
            "一定是",
            "可以确诊",
            "无需就医",
            "完全没问题"
        ]

        risk_hit = any(p in answer for p in risky_patterns)

        return 0 if risk_hit else 1

    # =========================
    # 5️⃣ 综合评分
    # =========================
    def evaluate(self, question, triples, answer):

        faith = self.evaluate_faithfulness(question, triples, answer)
        cover = self.evaluate_coverage(triples, answer)
        spec = self.evaluate_specificity(answer)
        safe = self.evaluate_safety(answer)

        final_score = (
            0.4 * faith +
            0.25 * cover +
            0.2 * spec +
            0.15 * safe
        )

        return {
            "faithfulness": round(faith,3),
            "coverage": round(cover,3),
            "specificity": round(spec,3),
            "safety": safe,
            "final_score": round(final_score,3)
        }
    
    def split_statements(self, answer):
        """
        将回答拆分为近似“医学断言”
        """
        statements = re.split(r"[。！？\n]", answer)
        return [s.strip() for s in statements if len(s.strip()) > 5]
    
    def parse_triples(self, triples):
        """
        返回：
        entities: set
        relations: set
        """
        entities = set()
        relations = set()

        for t in triples:
            match = re.findall(r"<(.*?),(.*?),(.*?)>", t)
            if match:
                h, r, t_ = match[0]
                entities.add(h)
                entities.add(t_)
                relations.add(r)

        return entities, relations


