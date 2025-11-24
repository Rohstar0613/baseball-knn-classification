import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def file_open():
    # 학습용, 테스트용, 시각화용(main) 데이터를 로드하여 반환
    train = pd.read_csv("Data/processed/train_20251123.csv")
    test = pd.read_csv("Data/processed/test_20251123.csv")
    main = pd.read_csv("Data/processed/main_20251123.csv") # 그래프 출력을 위한 파일
    return train, test, main


def make_k(train):
    # KNN에서 사용할 k 후보 리스트 생성 (3부터 절반까지, 홀수만)
    max_k_range = train.shape[0] // 2
    return [i for i in range(3, max_k_range, 2)]


def find_k(train, k_list):
    # 교차검증을 통해 최적의 k 값을 찾는 함수
    cross_validation_scores = []

    train = train.copy()
    # 투표율 컬럼 생성
    train["vote_rate"] = train["votes"] / train["ballots"]
    # 결측값 0으로 대체
    train = train.fillna(0)

    # 입력 변수와 타깃 변수 설정
    x_train = train[['vote_rate', 'votes']]
    y_train = train["inducted"]  

    # 주어진 k 후보들에 대해 교차검증 수행
    for k in k_list:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, x_train, y_train.values.ravel(),
                                cv=10, scoring='accuracy')
        cross_validation_scores.append(scores.mean())

    # 가장 높은 정확도를 내는 k 선택
    best_k = k_list[cross_validation_scores.index(max(cross_validation_scores))]

    return best_k, cross_validation_scores


def model_test(train, test, best_k):
    # 최적 k 값으로 모델 학습 및 예측 수행
    knn = KNeighborsClassifier(n_neighbors=best_k)

    train = train.copy()
    # 투표율 생성
    train["vote_rate"] = train["votes"] / train["ballots"]
    train = train.fillna(0)

    # 학습 데이터 분리
    x_train = train[['vote_rate', 'votes']]
    y_train = train["inducted"]  

    # 모델 학습
    knn.fit(x_train, y_train)

    test = test.copy()
    # 테스트 데이터 투표율 생성
    test["vote_rate"] = test["votes"] / test["ballots"]
    test = test.fillna(0)

    # 테스트 입력, 타깃 분리
    x_test = test[['vote_rate', 'votes']]
    y_test = test["inducted"] 

    # 예측 수행
    pred = knn.predict(x_test)

    return pred, y_test
