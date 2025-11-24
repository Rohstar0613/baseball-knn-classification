import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def visual_report(pred, y_test, save=False, filename="prediction", k_value=None):
    """
    예측 결과에 대한 분류 리포트, 정확도, 혼동 행렬을 출력하고
    옵션에 따라 CSV 및 이미지 파일로 저장하는 함수.

    Parameters
    ----------
    pred : array-like
        모델의 예측값
    y_test : array-like or Series
        실제 Ground Truth 레이블
    save : bool
        True일 경우 리포트/Confusion Matrix를 파일로 저장
    filename : str
        저장 파일명 prefix
    k_value : int or None
        모델 사용 K 값(KNN), 리포트 저장 시 포함
    """

    # -------------------------
    # 1) 기본 성능 출력
    # -------------------------
    report_text = classification_report(y_test, pred)
    accuracy_val = accuracy_score(y_test, pred)

    print("\n📊 Classification Report:")
    print(report_text)
    print(f"\n🎯 Final Accuracy: {accuracy_val:.4f}")

    # -------------------------
    # 2) Confusion Matrix 계산 및 시각화
    # -------------------------
    cm = confusion_matrix(y_test, pred)
    labels = sorted(list(set(y_test)))  # 클래스 레이블 자동 수집

    print("\n🧩 Confusion Matrix (raw counts):")
    print(cm)

    # 화면 출력용 Confusion Matrix
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    plt.title("Confusion Matrix")
    plt.show(block=False)
    plt.pause(0.1)

    # -------------------------
    # 3) 저장 옵션 처리
    # -------------------------
    if save:
        save_folder = "Data/result"
        os.makedirs(save_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 3-1) classification_report → CSV 저장
        report_df = pd.DataFrame(
            classification_report(y_test, pred, output_dict=True)
        ).T

        # 정확도 및 K 값 추가 저장
        report_df.loc["Final_accuracy", "score"] = accuracy_val
        report_df.loc["Model_Info", "K_value"] = k_value if k_value else "UNKNOWN"

        report_path = f"{save_folder}/{filename}_{timestamp}_REPORT.csv"
        report_df.to_csv(report_path)
        print(f"\n💾 리포트 저장 완료 → {report_path}")

        # 3-2) Confusion Matrix 숫자 버전 CSV 저장
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_csv_path = f"{save_folder}/{filename}_{timestamp}_CM.csv"
        cm_df.to_csv(cm_csv_path)
        print(f"💾 Confusion Matrix(CSV) 저장 완료 → {cm_csv_path}")

        # 3-3) Confusion Matrix 이미지 저장(PNG)
        cm_img_path = f"{save_folder}/{filename}_{timestamp}_CM.png"
        fig.savefig(cm_img_path, dpi=300, bbox_inches="tight")
        print(f"💾 Confusion Matrix(PNG) 저장 완료 → {cm_img_path}")

        plt.show(block=False)
        plt.pause(0.5)

    # 반환값: 추후 분석 가능
    return {
        "accuracy": accuracy_val,
        "k": k_value,
        "report": report_text,
        "confusion_matrix": cm,
    }


def data_view(cross_validation_scores, k_list, filename="knn_accuracy_plot", save=True, show=True):
    """KNN에서 k 값별 정확도 변화를 라인 그래프로 시각화하고 파일로 저장하는 함수."""

    fig, ax = plt.subplots(figsize=(10, 6))

    # 그래프 기본 구성
    ax.plot(k_list, cross_validation_scores)
    ax.set_xlabel("Number of K")
    ax.set_ylabel("Accuracy")
    ax.set_title("KNN Hyperparameter Tuning Results")
    ax.grid(True)

    # 이미지 저장 처리
    if save:
        save_folder = "plot"
        os.makedirs(save_folder, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"{save_folder}/{filename}_{timestamp}.png"
        fig.savefig(file_path, dpi=300, bbox_inches="tight")
        print(f"📁 이미지 저장 완료 → {file_path}")

    # 그래프 화면 표시
    if show:
        plt.show(block=False)
        plt.pause(0.5)


def scatter_template(
    df,
    x_col,
    y_col,
    hue_col=None,
    *,
    title=None,
    x_label=None,
    y_label=None,
    legend_title=None,
    filename=None,
    save=True,
    show=True,
    folder="plot"
):
    """
    통일된 스타일로 산점도(scatter plot)를 생성하고
    옵션에 따라 파일 저장 및 화면 표시까지 처리하는 함수.

    Parameters
    ----------
    df : DataFrame
        시각화할 데이터프레임
    x_col : str
        x축 컬럼명
    y_col : str
        y축 컬럼명
    hue_col : str or None
        색상 분류 컬럼
    title, x_label, y_label, legend_title : str or None
        그래프 텍스트 설정 (None이면 자동 생성)
    filename : str or None
        저장 파일명 prefix (None이면 자동 생성)
    save, show : bool
        그래프 저장/표시 여부
    folder : str
        저장 폴더명
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    # 기본 텍스트 자동 설정
    if x_label is None: x_label = x_col
    if y_label is None: y_label = y_col
    if legend_title is None: legend_title = hue_col
    if title is None: title = f"{y_label} vs {x_label}"
    if filename is None: filename = f"{y_col}_vs_{x_col}"

    # 산점도 생성
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette=["cornflowerblue", "coral"],
        alpha=0.8,
        s=60,
        ax=ax
    )

    # 그래프 옵션 설정
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if hue_col:
        ax.legend(title=legend_title)
    ax.grid(True)

    # 저장 옵션
    if save:
        os.makedirs(folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{folder}/{filename}_{timestamp}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"📁 그래프 저장 완료 → {path}")

    # 그래프 표시
    if show:
        plt.show(block=False)
        plt.pause(0.5)

