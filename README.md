# ⚾ KNN으로 야구 데이터를 분류해본 첫 프로젝트

**실패에서 개선까지 — Learning + Experiment Log**

---

## 프로젝트 개요
이 프로젝트는 HallOfFame.csv 데이터를 기반으로,

> **“선수가 명예의 전당에 헌액되었는지 여부를 투표율·득표수를 기반으로 분류하는 머신러닝 모델을 구축하는 것”**

을 목표로 진행되었다.

초기에는 단순히 투표율 기반 라벨링을 적용했지만, 데이터 불균형 문제와 스케일링 이슈로 인해 여러 차례 수정이 필요했다. 그 과정에서 EDA, 전처리, 모델 튜닝, 시각화 자동화, 실험 기록 구조화까지 경험하며 처음으로 머신러닝 프로젝트의 “전체 파이프라인”을 구축했다.

---

## 📁 프로젝트 구조

```
BASEBALL-KNN/
│
├── Data/
│   ├── processed/
│   └── result/
│
├── plot/
│   ├── baseline_knn_k_search.png
│   └── hof_votes_vote_rate.png
│
├── src/
│   ├── __init__.py
│   ├── knn_model.py
│   └── visualize.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 🚀 실행 방법

### 1️⃣ 환경 설치 및 실행

```bash
git clone https://github.com/rohstar-ai/Baseball-KNN-Project.git
cd Baseball-KNN-Project
pip install -r requirements.txt
python main.py
```

---

## 🧠 배운 점 & 핵심 내용

* 데이터 구조가 모델에 적합한지 먼저 확인해야 했다.
* Accuracy만으로는 모델의 성능을 판단하기 어렵고, **불균형 데이터에서는 Recall, Precision, F1-score가 더 중요**하다.
* KNN은 거리 기반 모델이기 때문에 **스케일링과 데이터 분포**가 매우 중요하다.
* 반복 실험 과정:

  * **EDA → 실패 → 데이터 수정 → 모델 개선 → 최적화 → 비교 → 결과 기록**

---

## 📊 실험 결과

| Metric | Precision | Recall   | F1-score | Accuracy |
| ------ | --------- | -------- | -------- | -------- |
| Result | **0.88**  | **0.88** | **0.88** | **0.88** |

클래스 균형 조정 후 두 클래스를 고르게 예측하는 모습을 확인했다.

---

## 📎 시각화 기능

* ✔ k값 변화에 따른 Accuracy 그래프
* ✔ Scatter Plot 기반 분포 시각화
* ✔ Confusion Matrix 자동 저장
* ✔ 모든 결과는 `/plot/` 또는 `/Data/result/`에 저장

---

## 🏁 한 줄 정리

> **“실패는 방향을 알려줬고, 반복은 모델을 성장시켰다.”**

처음 하는 머신러닝 프로젝트였지만,

* 데이터 전처리
* 모델 튜닝
* 시각화 자동화
* 재현 가능한 실험 구조 설계

까지 경험할 수 있었던 의미 있는 프로젝트였다.

---

## 🔗 블로그 회고록

👉 [https://rohstar.tistory.com/entry/%F0%9F%93%98-KNN%EC%9C%BC%EB%A1%9C-%EC%95%BC%EA%B5%AC-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%A5%BC-%EB%B6%84%EB%A5%98%ED%95%B4%EB%B3%B8-%EC%B2%AB-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EC%8B%A4%ED%8C%A8%EC%97%90%EC%84%9C-%EA%B0%9C%EC%84%A0%EA%B9%8C%EC%A7%80](https://rohstar.tistory.com/entry/%F0%9F%93%98-KNN%EC%9C%BC%EB%A1%9C-%EC%95%BC%EA%B5%AC-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%A5%BC-%EB%B6%84%EB%A5%98%ED%95%B4%EB%B3%B8-%EC%B2%AB-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EC%8B%A4%ED%8C%A8%EC%97%90%EC%84%9C-%EA%B0%9C%EC%84%A0%EA%B9%8C%EC%A7%80)

---
