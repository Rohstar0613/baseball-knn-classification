# baseball-knn-classification
KNN 기반 야구 명예의 전당 데이터 분류 프로젝트 (From Failure to Improvement)
# ⚾ KNN으로 명예의 전당 후보 분류하기  
_첫 머신러닝 실험 — 실패에서 개선까지_

---

## 📌 프로젝트 소개

이 프로젝트는 KNN 알고리즘을 활용해  
**MLB Hall of Fame 후보자가 헌액될지(Y/N)를 예측하는 모델**을 만드는 과정입니다.

처음에는 `Batting.csv` 데이터를 활용해 **NL / AL 분류 모델**을 시도했으나,  
데이터 패턴 부재로 실패했고, 이후 `HallOfFame.csv`로 방향을 전환하여  
유의미한 모델 성능을 만들었습니다.

---

## 📂 폴더 구조

├── Data
│ ├── processed
│ └── result
├── plot
│ ├── baseline_knn_k_search.png
│ └── hof_votes_vote_rate.png
├── src
│ ├── knn_model.py
│ └── visualize.py
├── main.py
└── README.md

yaml
코드 복사

---

## 🚀 사용 방법

### 1️⃣ 환경 세팅
```sh
pip install -r requirements.txt
2️⃣ 실행
sh
코드 복사
python main.py

🧠 주요 학습 포인트
데이터가 정답을 예측할 수 있는 구조인지 먼저 확인해야 한다.

Accuracy만 보면 모델이 좋아 보일 수 있지만,
불균형 데이터에서는 Recall, F1-score가 더 중요하다.

EDA → 실패 → 데이터 수정 → 모델 개선 → 검증
이 반복 과정이 모델링의 핵심이라는 걸 체감했다.

📊 모델 성능 (최종 결과)
Metric	Precision	Recall	F1-score	Accuracy
Result	0.88	0.88	0.88	0.8800

➡ 클래스 균형 조정 후 모델이 의미 있게 개선됨.

🔗 실험 기록(블로그)
프로젝트 과정 전체 기록:
👉 https://rohstar.tistory.com/entry/%F0%9F%93%98-KNN%EC%9C%BC%EB%A1%9C-%EC%95%BC%EA%B5%AC-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%A5%BC-%EB%B6%84%EB%A5%98%ED%95%B4%EB%B3%B8-%EC%B2%AB-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EC%8B%A4%ED%8C%A8%EC%97%90%EC%84%9C-%EA%B0%9C%EC%84%A0%EA%B9%8C%EC%A7%80

🧩 앞으로 개선할 점
여러 모델 비교 (Logistic Regression / RandomForest / SVM)

Feature Engineering 확장

Hyperparameter 자동화(GridSearch / Bayesian Optimization)

🏁 결론
이 프로젝트는 첫 머신러닝 프로젝트로서 의미 있는 과정을 담고 있으며,
단순한 결과보다 과정, 실패, 개선에 초점을 맞췄습니다.

"모델보다 데이터를 먼저 이해하라"
이 프로젝트에서 가장 크게 느낀 문장이었습니다.
