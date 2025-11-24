from src.visualize import *
from src.knn_model import *

def main():
    # train, test, 나누기 전 파일 열기
    train, test, main = file_open()

    # 자료 분포 그래프 그리기
    main["vote_rate"] = main["votes"] / main["ballots"]
    scatter_template(main, "votes", "vote_rate", hue_col="inducted")

    # K 리스트 만들기
    k_list = make_k(train)

    # 최적의 k 찾기
    best_k, cross_validation_scores = find_k(train, k_list)

    # k 그래프 그리기
    data_view(cross_validation_scores, k_list, filename="knn_accuracy_plot", save=True, show=True)

    # 모델 학습, 모델 테스트
    pred, y_test = model_test(train, test, best_k)

    # 성능 계산, 성능 리포트 출력
    visual_report(pred, y_test, save=True, filename="knn_baseline", k_value=best_k)

    # 화면에 떠있는 모든 그래프 닫기
    plt.close('all')


if __name__ == "__main__":
    main()