# ⚾ KNN으로 야구 데이터를 분류해본 첫 프로젝트  
**실패에서 개선까지 — Learning + Experiment Log**

---

## 📁 프로젝트 구조

BASEBALL-KNN/
│
├── Data/
│ ├── processed/
│ └── result/
│
├── plot/
│ ├── baseline_knn_k_search.png
│ └── hof_votes_vote_rate.png
│
├── src/
│ ├── init.py
│ ├── knn_model.py
│ └── visualize.py
│
├── main.py
├── requirements.txt
└── README.md

yaml
코드 복사

---

## 🚀 실행 방법

### 1️⃣ 환경 설치

git clone https://github.com/rohstar-ai/Baseball-KNN-Project.git


파일 다운로드


cd Baseball-KNN-Project


모듈 설치


pip install -r requirements.txt


실행


python main.py


🧠 배운 점 & 핵심 내용


데이터가 모델에 적합한 구조인지 먼저 확인하는 것이 중요했다.


Accuracy만 보면 모델이 좋아 보일 수 있지만,


불균형 데이터에서는 Recall, Precision, F1-score가 더 의미 있다.


KNN 특성상 데이터 분포와 스케일링의 중요성을 체감했다.


실험 반복 과정:


EDA → 실패 → 데이터 수정 → 모델 개선 → 최적화 → 비교 → 결과 기록


📊 실험 결과


Metric	Precision	Recall	F1-score	Accuracy


Result	0.88	0.88	0.88	0.88


클래스 균형을 맞춘 후, 모델이 두 클래스를 고르게 예측하는 모습을 확인할 수 있었다.


📎 시각화 기능


✔ k값 변화에 따른 Accuracy 그래프


✔ 분포 시각화 (Scatter Plot)


✔ Confusion Matrix 자동 저장


✔ 결과 파일들은 /plot/ 또는 /Data/result/에 저장됨


🏁 한 줄 정리


“실패했지만, 그 실패는 방향을 알려줬고,


모델은 그 반복 속에서 점점 더 잘 예측하기 시작했다.”


처음 하는 머신러닝 프로젝트였지만,


데이터 전처리


모델 튜닝


시각화 자동화


재현 가능한 실험 흐름 구성


까지 경험한 매우 의미 있는 과정이었다.


🔗 블로그 정리글


👉 https://rohstar.tistory.com/entry/%F0%9F%93%98-KNN%EC%9C%BC%EB%A1%9C-%EC%95%BC%EA%B5%AC-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%A5%BC-%EB%B6%84%EB%A5%98%ED%95%B4%EB%B3%B8-%EC%B2%AB-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EC%8B%A4%ED%8C%A8%EC%97%90%EC%84%9C-%EA%B0%9C%EC%84%A0%EA%B9%8C%EC%A7%80

📩 유지보수 계획



Logistic Regression, SVM 모델 추가 예정


다른 Feature Engineering 실험


StandardScaler & PCA 적용 후 성능 비교
